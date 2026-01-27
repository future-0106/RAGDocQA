"""测试5：完整RAG流程（GPU版+修复_type错误+回答完整自然）"""
import os
import re
import torch
import dotenv
import warnings
import pdfplumber
import shutil  # 新增：用于删除旧向量库
from typing import List
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# -------------------------- 优先加载.env文件 --------------------------
dotenv.load_dotenv(override=True)

# -------------------------- LangChain 0.2.x 适配导入 --------------------------
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.documents import Document
except ImportError as e:
    print(f"❌ LangChain导入失败：{e}")
    print("💡 执行：pip install langchain==0.2.10 langchain-community==0.2.10 --no-cache-dir")
    exit(1)

# -------------------------- 阿里百炼依赖验证 --------------------------
try:
    import dashscope
    from dashscope import Generation
except ImportError as e:
    print(f"❌ 阿里百炼依赖缺失：{e}")
    print("💡 执行：pip install dashscope==1.14.0 --no-cache-dir")
    exit(1)

# -------------------------- 全局配置（GPU+检索优化） --------------------------
LOCAL_MODEL_PATH = r"./Qwen3_Embedding_0.6B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EMBED_LENGTH = 8192

# PDF配置（改为你的目标PDF路径）
PDF_PATH = r"劳动合同法问题解答.pdf"
CHUNK_SIZE = 800  # 保留完整段落
CHUNK_OVERLAP = 100
SEARCH_K = 8  # 增大K值，弥补无score_threshold的漏检风险

# 阿里百炼配置
MODEL_NAME = "qwen-turbo"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
TEMPERATURE = 0.3  # 适当提高，避免过度简略
MAX_TOKENS = 1500  # 足够大，确保完整输出

# 向量库配置
PERSIST_DIR = r"./chroma_db"

# -------------------------- 配置验证 --------------------------
def validate_config():
    if not DASHSCOPE_API_KEY:
        raise ValueError(
            f"❌ .env文件缺失DASHSCOPE_API_KEY配置\n"
            "💡 .env文件只需添加一行：\n"
            "DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        )
    if torch.cuda.is_available():
        print(f"✅ 检测到GPU：{torch.cuda.get_device_name(0)}")
        print(f"✅ GPU显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️ 未检测到GPU，自动使用CPU运行")
    print("✅ 阿里百炼API Key配置验证通过！")

# -------------------------- 工具函数 --------------------------
def fix_cpu_compatibility():
    """修复Qwen3-Embedding兼容性（CPU/GPU通用）"""
    if not hasattr(torch.library, "register_fake"):
        def dummy_register_fake(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        torch.library.register_fake = dummy_register_fake

    if not hasattr(torch._C, "_dispatch_has_kernel_for_dispatch_key"):
        torch._C._dispatch_has_kernel_for_dispatch_key = lambda *args, **kwargs: False
    print("✅ 环境兼容修复完成")

def clean_text(text: str) -> str:
    """过滤PDF乱码，适配中文"""
    if not text:
        return ""
    # 保留中文、英文、数字、常用标点（适配你的中文文档）
    valid_chars = re.compile(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s，。！？；：""''（）【】《》、·-]')
    text = valid_chars.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_pdf(pdf_path: str) -> str:
    """用pdfplumber加载PDF"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"❌ PDF文件不存在：{pdf_path}\n"
            f"💡 请修改代码中PDF_PATH变量为实际路径（当前：{PDF_PATH}）"
        )

    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        print(f"✅ PDF加载成功！共 {page_count} 页")

        for page in pdf.pages:
            page_text = page.extract_text() or ""
            clean_page_text = clean_text(page_text)
            if clean_page_text:
                full_text += clean_page_text + "\n"

    if not full_text:
        raise ValueError("⚠️ PDF无有效文本！可能是扫描件（图片型PDF），需OCR处理")
    return full_text

def split_text(text: str) -> List[Document]:
    """分割文本（适配中文，补充_type元数据解决LangChain兼容问题）"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "：", "，", "、"],
        length_function=len
    )
    chunks = splitter.split_text(text)
    chunks = [c.strip() for c in chunks if c.strip()]

    # 核心修复：给每个Document补充_type元数据，兼容LangChain 0.2.x
    docs = []
    for idx, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": PDF_PATH,
                "chunk_idx": idx,
                "_type": "Document"  # 手动补充缺失的_type字段
            }
        )
        docs.append(doc)

    print(f"✅ PDF分割成功！共生成 {len(docs)} 个文本片段")
    return docs

# -------------------------- Qwen3-Embedding GPU版 --------------------------
class Qwen3Embeddings:
    def __init__(self, model_path: str, device: torch.device = torch.device("cpu")):
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ 模型路径不存在：{self.model_path}")

        fix_cpu_compatibility()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side='left',
            trust_remote_code=True,
            local_files_only=True
        )

        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✅ Qwen3-Embedding加载成功（{self.device}）：{self.model_path}")

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [clean_text(t) for t in texts]
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_EMBED_LENGTH,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(** batch_dict)

        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# -------------------------- 阿里百炼LLM类（优化生成参数） --------------------------
class DashScopeChatModel(BaseChatModel):
    model_name: str = MODEL_NAME
    api_key: str = DASHSCOPE_API_KEY
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        dashscope_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                dashscope_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                dashscope_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                dashscope_messages.append({"role": "system", "content": msg.content})

        if not self.api_key:
            raise ValueError("❌ DASHSCOPE_API_KEY未配置")
        dashscope.api_key = self.api_key

        try:
            response = Generation.call(
                model=self.model_name,
                messages=dashscope_messages,
                result_format='message',
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop or [],
                **kwargs
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
            else:
                content = f"API调用失败：{response.code} - {response.message}"
        except Exception as e:
            content = f"LLM调用异常：{str(e)}"

        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self):
        return "dashscope-qwen"

# -------------------------- 完整RAG类（核心修复：删除旧向量库+补充_type） --------------------------
class NaiveRAG:
    def __init__(self):
        validate_config()
        # 新增：删除旧的向量库数据，避免格式冲突
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
            print(f"✅ 已删除旧向量库：{PERSIST_DIR}")

        self.embeddings = self._init_embeddings()
        self.pdf_docs = self._load_and_split_pdf()
        self.vector_db = self._build_vector_db()
        # 修复score_threshold错误：仅保留k参数
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={"k": SEARCH_K}
        )
        self.llm = DashScopeChatModel()
        self.rag_chain = self._build_rag_chain()
        print("✅ RAG系统初始化完成！")

    def _init_embeddings(self):
        return Qwen3Embeddings(model_path=LOCAL_MODEL_PATH, device=DEVICE)

    def _load_and_split_pdf(self):
        full_text = load_pdf(PDF_PATH)
        return split_text(full_text)

    def _build_vector_db(self):
        os.makedirs(PERSIST_DIR, exist_ok=True)

        vector_db = Chroma.from_documents(
            documents=self.pdf_docs,
            embedding=self.embeddings,
            persist_directory=PERSIST_DIR
        )
        vector_db.persist()
        print(f"✅ 向量库构建成功：{PERSIST_DIR}")
        return vector_db

    def _build_rag_chain(self):
        """核心优化：Prompt明确要求“完整+重组+自然”"""
        prompt = ChatPromptTemplate.from_template("""
        请严格按照以下规则处理用户问题，必须遵守所有要求：
        1. 信息完整性：提取上下文所有与问题相关的关键信息，不遗漏任何段落、定义、属性等内容；
        2. 表述方式：不要直接复制原文，用自己的话逻辑连贯地重组信息，可适当分点（如果信息点多）；
        3. 语言风格：中文流畅、正式，符合文档表述逻辑，不添加外部知识；
        4. 无相关信息：若上下文无相关内容，直接回答“无法从文档中找到相关答案”。

        上下文：
        {context}

        用户问题：
        {question}
        """)

        # 打印检索到的上下文（方便排查漏段）
        def format_docs(docs):
            context = "\n\n".join([doc.page_content for doc in docs])
            print("\n🔍 检索到的相关上下文：")
            print("=" * 60)
            print(context)
            print("=" * 60)
            return context

        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("✅ RAG链构建成功")
        return rag_chain

    def query(self, question: str) -> str:
        print(f"\n❓ 用户问题：{question}")
        print("🔍 正在检索并生成自然完整的回答...")
        try:
            answer = self.rag_chain.invoke(question)
            print(f"\n💡 最终回答：")
            print("-" * 50)
            print(answer)
            print("-" * 50)
            return answer
        except Exception as e:
            print(f"❌ 查询失败：{str(e)}")
            # 打印详细错误栈，方便排查（新增）
            import traceback
            traceback.print_exc()
            return ""

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("📌 核心配置信息：")
    print(f"PyTorch版本：{torch.__version__}")
    print(f"运行设备：{DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU型号：{torch.cuda.get_device_name(0)}")
    print(f"生成参数：temperature={TEMPERATURE}，max_tokens={MAX_TOKENS}")
    print(f"检索参数：SEARCH_K={SEARCH_K}")
    print(f"PDF路径：{PDF_PATH}\n")

    try:
        rag_system = NaiveRAG()
        # 测试问题（匹配你的劳动合同法PDF）
        rag_system.query("试用期是否包含在劳动合同期限内？")
    except FileNotFoundError as e:
        print(f"\n❌ 文件不存在：{str(e)}")
    except ImportError as e:
        print(f"\n❌ 依赖缺失：{str(e)}")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ GPU显存不足：{str(e)}")
            print("💡 解决：减小CHUNK_SIZE或改用CPU")
        else:
            print(f"\n❌ GPU错误：{str(e)}")
    except Exception as e:
        print(f"\n❌ 运行失败：{str(e)}")
        # 打印详细错误栈
        import traceback
        traceback.print_exc()
        print("💡 请确保阿里百炼API Key有效、GPU环境配置正确、PDF路径正确！")