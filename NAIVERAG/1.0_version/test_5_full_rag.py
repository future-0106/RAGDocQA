# """测试5：完整RAG流程（修复'dict' has no 'page_content'错误，仅需.env配置阿里百炼密钥）"""
# import os
# import re
# import torch
# import dotenv
# import warnings
# import pdfplumber
# from typing import List
# import torch.nn.functional as F
# from torch import Tensor
# from transformers import AutoTokenizer, AutoModel
#
# # -------------------------- 优先加载.env文件 --------------------------
# dotenv.load_dotenv(override=True)
#
# # -------------------------- LangChain 0.2.x 适配导入（新增Document导入） --------------------------
# try:
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
#     from langchain_community.vectorstores import Chroma
#     from langchain_core.prompts import ChatPromptTemplate
#     from langchain_core.runnables import RunnablePassthrough
#     from langchain_core.output_parsers import StrOutputParser
#     from langchain_core.language_models import BaseChatModel
#     from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
#     from langchain_core.outputs import ChatGeneration, ChatResult
#     # 关键修复：导入LangChain的Document类
#     from langchain_core.documents import Document
# except ImportError as e:
#     print(f"❌ LangChain导入失败：{e}")
#     print("💡 执行：pip install langchain==0.2.10 langchain-community==0.2.10 --no-cache-dir")
#     exit(1)
#
# # -------------------------- 阿里百炼依赖验证 --------------------------
# try:
#     import dashscope
#     from dashscope import Generation
# except ImportError as e:
#     print(f"❌ 阿里百炼依赖缺失：{e}")
#     print("💡 执行：pip install dashscope==1.14.0 --no-cache-dir")
#     exit(1)
#
# # -------------------------- 全局配置（仅DASHSCOPE_API_KEY从.env读取） --------------------------
# # 嵌入模型配置
# LOCAL_MODEL_PATH = r"D:\projects\fastapi_langchain_env\NAIVERAG\model\Qwen3_Embedding_0.6B"  # 手动修改为你的模型路径
# DEVICE = torch.device("cpu")
# MAX_EMBED_LENGTH = 8192
#
# # PDF配置
# PDF_PATH = r"劳动合同法问题解答.pdf"  # 手动修改为你的PDF路径
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 50
#
# # 阿里百炼配置
# MODEL_NAME = "qwen-turbo"
# DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 仅从.env读取API Key
#
# # 向量库配置
# PERSIST_DIR = r"./chroma_db"
# SEARCH_K = 3
#
# # -------------------------- 配置验证 --------------------------
# def validate_config():
#     """仅验证阿里百炼API Key"""
#     if not DASHSCOPE_API_KEY:
#         raise ValueError(
#             f"❌ .env文件缺失DASHSCOPE_API_KEY配置\n"
#             "💡 .env文件只需添加一行：\n"
#             "DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
#         )
#     print("✅ 阿里百炼API Key配置验证通过！")
#
# # -------------------------- 工具函数 --------------------------
# def fix_cpu_compatibility():
#     """修复Qwen3-Embedding在CPU环境下的兼容性问题"""
#     if not hasattr(torch.library, "register_fake"):
#         def dummy_register_fake(*args, **kwargs):
#             def decorator(func):
#                 return func
#             return decorator
#         torch.library.register_fake = dummy_register_fake
#
#     if not hasattr(torch._C, "_dispatch_has_kernel_for_dispatch_key"):
#         torch._C._dispatch_has_kernel_for_dispatch_key = lambda *args, **kwargs: False
#     print("✅ CPU环境兼容修复完成")
#
# def clean_text(text: str) -> str:
#     """过滤PDF乱码，仅保留有效字符"""
#     if not text:
#         return ""
#     # 保留中文、英文、数字、常用标点
#     valid_chars = re.compile(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s，。！？；：""''（）【】《》、·]')
#     text = valid_chars.sub('', text)
#     # 去除多余空格/换行
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text
#
# def load_pdf(pdf_path: str) -> str:
#     """用pdfplumber加载PDF，解决中文乱码"""
#     if not os.path.exists(pdf_path):
#         raise FileNotFoundError(
#             f"❌ PDF文件不存在：{pdf_path}\n"
#             f"💡 请修改代码中PDF_PATH变量为实际路径（当前：{PDF_PATH}）"
#         )
#
#     full_text = ""
#     with pdfplumber.open(pdf_path) as pdf:
#         page_count = len(pdf.pages)
#         print(f"✅ PDF加载成功！共 {page_count} 页")
#
#         for page in pdf.pages:
#             page_text = page.extract_text() or ""
#             clean_page_text = clean_text(page_text)
#             if clean_page_text:
#                 full_text += clean_page_text + "\n"
#
#     if not full_text:
#         raise ValueError("⚠️ PDF无有效文本！可能是扫描件（图片型PDF），需OCR处理")
#     return full_text
#
#
# def split_text(text: str) -> List[Document]:
#     """分割文本（适配中文，补充_type元数据）"""
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", "。", "！", "？", "；", "：", "，", "、"],  # 中文分隔符
#         length_function=len
#     )
#     chunks = splitter.split_text(text)
#     chunks = [c.strip() for c in chunks if c.strip()]
#
#     # 核心修改：给每个Document补充_type元数据，兼容LangChain 0.2.x
#     docs = []
#     for idx, chunk in enumerate(chunks):
#         doc = Document(
#             page_content=chunk,
#             metadata={
#                 "source": PDF_PATH,
#                 "chunk_idx": idx,
#                 "_type": "Document"  # 手动补充缺失的_type字段
#             }
#         )
#         docs.append(doc)
#
#     print(f"✅ PDF分割成功！共生成 {len(docs)} 个文本片段")
#     return docs
# # -------------------------- Qwen3-Embedding 自定义实现 --------------------------
# class Qwen3Embeddings:
#     """适配Qwen3-Embedding的自定义嵌入类（兼容langchain接口）"""
#     def __init__(self, model_path: str, device: torch.device = torch.device("cpu")):
#         self.model_path = model_path
#         self.device = device
#         self.tokenizer = None
#         self.model = None
#         self._load_model()
#
#     def _load_model(self):
#         """加载本地Qwen3-Embedding模型"""
#         if not os.path.exists(self.model_path):
#             raise FileNotFoundError(
#                 f"❌ 嵌入模型路径不存在：{self.model_path}\n"
#                 f"💡 请修改代码中LOCAL_MODEL_PATH变量为实际路径（当前：{LOCAL_MODEL_PATH}）"
#             )
#
#         # CPU兼容修复
#         fix_cpu_compatibility()
#
#         # 加载Tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_path,
#             padding_side='left',
#             trust_remote_code=True,
#             local_files_only=True
#         )
#
#         # 加载Model
#         self.model = AutoModel.from_pretrained(
#             self.model_path,
#             trust_remote_code=True,
#             device_map="cpu",
#             local_files_only=True,
#             torch_dtype=torch.float32
#         )
#         self.model = self.model.to(self.device)
#         self.model.eval()
#         print(f"✅ Qwen3-Embedding加载成功：{self.model_path}")
#
#     @staticmethod
#     def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
#         """Qwen3官方池化函数"""
#         left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
#         if left_padding:
#             return last_hidden_states[:, -1]
#         else:
#             sequence_lengths = attention_mask.sum(dim=1) - 1
#             batch_size = last_hidden_states.shape[0]
#             return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
#
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """批量嵌入文档（兼容langchain接口）"""
#         texts = [clean_text(t) for t in texts]
#         batch_dict = self.tokenizer(
#             texts,
#             padding=True,
#             truncation=True,
#             max_length=MAX_EMBED_LENGTH,
#             return_tensors="pt"
#         ).to(self.device)
#
#         with torch.no_grad():
#             outputs = self.model(** batch_dict)
#
#         embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
#         embeddings = F.normalize(embeddings, p=2, dim=1)
#         return embeddings.cpu().tolist()
#
#     def embed_query(self, text: str) -> List[float]:
#         """嵌入查询文本（兼容langchain接口）"""
#         return self.embed_documents([text])[0]
#
# # -------------------------- 阿里百炼LLM类 --------------------------
# class DashScopeChatModel(BaseChatModel):
#     model_name: str = MODEL_NAME
#     api_key: str = DASHSCOPE_API_KEY
#
#     def _generate(self, messages, stop=None, run_manager=None, **kwargs):
#         # 转换消息格式
#         dashscope_messages = []
#         for msg in messages:
#             if isinstance(msg, HumanMessage):
#                 dashscope_messages.append({"role": "user", "content": msg.content})
#             elif isinstance(msg, AIMessage):
#                 dashscope_messages.append({"role": "assistant", "content": msg.content})
#             elif isinstance(msg, SystemMessage):
#                 dashscope_messages.append({"role": "system", "content": msg.content})
#
#         # 验证API Key
#         if not self.api_key:
#             raise ValueError("❌ DASHSCOPE_API_KEY未配置，请检查.env文件")
#         dashscope.api_key = self.api_key
#
#         # 调用阿里百炼API
#         try:
#             response = Generation.call(
#                 model=self.model_name,
#                 messages=dashscope_messages,
#                 result_format='message',
#                 temperature=0,
#                 stop=stop or [],
#                 **kwargs
#             )
#
#             if response.status_code == 200:
#                 content = response.output.choices[0].message.content
#             else:
#                 content = f"API调用失败：{response.code} - {response.message}"
#         except Exception as e:
#             content = f"LLM调用异常：{str(e)}"
#
#         # 返回langchain兼容结果
#         generation = ChatGeneration(message=AIMessage(content=content))
#         return ChatResult(generations=[generation])
#
#     @property
#     def _llm_type(self):
#         return "dashscope-qwen"
#
# # -------------------------- 完整RAG类 --------------------------
# class NaiveRAG:
#     def __init__(self):
#         # 验证API Key
#         validate_config()
#
#         # 初始化组件
#         self.embeddings = self._init_embeddings()
#         self.pdf_docs = self._load_and_split_pdf()
#         self.vector_db = self._build_vector_db()
#         self.retriever = self.vector_db.as_retriever(search_kwargs={"k": SEARCH_K})
#         self.llm = DashScopeChatModel()
#         self.rag_chain = self._build_rag_chain()
#         print("✅ RAG系统初始化完成！")
#
#     def _init_embeddings(self):
#         """初始化Qwen3嵌入模型"""
#         return Qwen3Embeddings(model_path=LOCAL_MODEL_PATH, device=DEVICE)
#
#     def _load_and_split_pdf(self):
#         """加载并分割PDF"""
#         full_text = load_pdf(PDF_PATH)
#         return split_text(full_text)
#
#     def _build_vector_db(self):
#         """构建Chroma向量库"""
#         # 检查向量库目录
#         os.makedirs(PERSIST_DIR, exist_ok=True)
#
#         # 构建向量库（现在传入的是Document对象，而非字典）
#         vector_db = Chroma.from_documents(
#             documents=self.pdf_docs,
#             embedding=self.embeddings,
#             persist_directory=PERSIST_DIR
#         )
#         vector_db.persist()
#         print(f"✅ 向量库构建成功：{PERSIST_DIR}")
#         return vector_db
#
#     def _build_rag_chain(self):
#         """构建RAG链（修复format_docs函数）"""
#         # 定义Prompt
#         prompt = ChatPromptTemplate.from_template("""
#         请严格基于以下上下文信息回答用户的问题，仅使用上下文里的内容，不要添加任何外部知识。
#         如果上下文没有相关信息，请直接回答：“无法从文档中找到相关答案”。
#
#         上下文：
#         {context}
#
#         用户问题：
#         {question}
#         """)
#
#         # 修复format_docs：Document对象用.page_content属性访问
#         def format_docs(docs):
#             return "\n\n".join([doc.page_content for doc in docs])
#
#         # 构建RAG链
#         rag_chain = (
#             {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
#             | prompt
#             | self.llm
#             | StrOutputParser()
#         )
#         print("✅ RAG链构建成功")
#         return rag_chain
#
#     def query(self, question: str) -> str:
#         """执行RAG查询"""
#         print(f"\n❓ 用户问题：{question}")
#         print("🔍 正在检索相关文档并生成回答...")
#         try:
#             answer = self.rag_chain.invoke(question)
#             print(f"💡 回答：{answer}")
#             return answer
#         except Exception as e:
#             print(f"❌ 查询失败：{str(e)}")
#             return ""
#
# # -------------------------- 主函数 --------------------------
# if __name__ == "__main__":
#     # 禁用无关警告
#     warnings.filterwarnings("ignore")
#
#     # 打印核心配置
#     print("📌 核心配置信息：")
#     print(f"PyTorch版本：{torch.__version__} (要求≥2.2.2)")
#     print(f"运行设备：{DEVICE}")
#     print(f"阿里百炼模型：{MODEL_NAME}")
#     print(f"本地模型路径：{LOCAL_MODEL_PATH}")
#     print(f"PDF文件路径：{PDF_PATH}\n")
#
#     try:
#         # 初始化RAG系统
#         rag_system = NaiveRAG()
#
#         # 测试查询
#         rag_system.query("什么是“民办非企业单位”？")
#
#     except FileNotFoundError as e:
#         print(f"\n❌ 文件不存在错误：{str(e)}")
#         print("💡 解决方法：手动修改代码中LOCAL_MODEL_PATH或PDF_PATH为实际路径")
#     except ImportError as e:
#         print(f"\n❌ 依赖缺失错误：{str(e)}")
#         print("💡 执行以下命令安装依赖：")
#         print("   pip install torch==2.2.2+cpu pdfplumber==0.11.4 chromadb==0.5.0")
#         print("   pip install langchain==0.2.10 langchain-community==0.2.10 dashscope==1.14.0")
#     except ValueError as e:
#         print(f"\n❌ 配置错误：{str(e)}")
#         print("💡 解决方法：检查.env文件中是否仅配置了DASHSCOPE_API_KEY")
#     except Exception as e:
#         print(f"\n❌ 运行失败：{str(e)}")
#         print("💡 请先确保阿里百炼API Key有效，且本地模型/PDF路径正确！")


