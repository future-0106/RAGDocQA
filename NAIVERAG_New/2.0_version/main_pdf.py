#!/usr/bin/env python3
"""
使用Qwen3-0.6B和Qwen3-Embedding-0.6B的完整RAG系统
支持PDF、TXT、MD文档格式，上传后立即转换为向量
"""

import os
import sys
import torch
import shutil
import time
import pdfplumber
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pydantic import Field

import warnings
from transformers import logging

# 关闭transformers的特定警告
logging.set_verbosity_error()
warnings.filterwarnings("ignore",
                        message=".*generation_config.*default values have been modified.*")

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 项目路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
MODELS_DIR = BASE_DIR / "models"

# 创建目录
for dir_path in [DATA_DIR, CHROMA_DB_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)


# ================ 导入检查 ================
def check_imports():
    """检查必要的导入"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ 使用 langchain_huggingface.HuggingFaceEmbeddings")
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print("⚠️  使用 langchain_community.HuggingFaceEmbeddings")
        except ImportError:
            print("❌ 无法导入 HuggingFaceEmbeddings")
            print("请安装: pip install langchain-huggingface 或 langchain-community")
            return False

    try:
        from langchain_chroma import Chroma
        print("✅ 使用 langchain_chroma.Chroma")
        return True
    except ImportError as e:
        print(f"❌ 无法导入 ChromaDB: {e}")
        print("请安装: pip install chromadb langchain-chroma")
        return False


if not check_imports():
    sys.exit(1)

# 现在导入需要的模块
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ================ 1. 嵌入模型适配 ================
class QwenEmbeddings:
    """适配Qwen3-Embedding-0.6B的嵌入模型"""

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = str(MODELS_DIR / "Qwen3_Embedding_0.6B")

        print(f"🔧 加载嵌入模型: {model_path}")

        # Qwen嵌入模型需要特殊的加载方式
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'trust_remote_code': True,
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 8,
            }
        )

        try:
            test_embedding = self.embeddings.embed_query("测试文本")
            print(f"✅ 嵌入模型加载成功，向量维度: {len(test_embedding)}")
        except Exception as e:
            print(f"⚠️  嵌入模型测试时出错: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)


# ================ 2. 大语言模型适配 ================
class QwenLLM(LLM):
    """适配Qwen3-0.6B的大语言模型"""

    # 使用Pydantic Field声明字段
    model_path: str = Field(default="Qwen3_0.6B", description="模型路径")
    max_new_tokens: int = Field(default=512, description="生成的最大token数")
    temperature: float = Field(default=0.1, description="温度参数")
    top_p: float = Field(default=0.9, description="top-p采样参数")
    repetition_penalty: float = Field(default=1.1, description="重复惩罚")
    do_sample: bool = Field(default=True, description="是否采样")

    # 内部使用的字段
    _tokenizer: Any = None
    _model: Any = None
    _generation_config: Any = None
    _device: str = "cpu"

    def __init__(self, **data):
        super().__init__(**data)

        print(f"🔧 加载语言模型: {self.model_path}")

        # 检查路径是否存在
        if not os.path.exists(self.model_path):
            print(f"⚠️  模型路径不存在: {self.model_path}")
            print("正在尝试从本地models目录查找...")
            local_path = MODELS_DIR / "Qwen3_0.6B"
            if local_path.exists():
                self.model_path = str(local_path)
                print(f"✅ 使用本地模型: {self.model_path}")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 使用设备: {self._device}")

        try:
            # 加载tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )

            # 如果没有pad_token，使用eos_token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # 加载模型
            torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto" if self._device == "cuda" else None,
                low_cpu_mem_usage=True
            )

            if self._device == "cpu":
                self._model = self._model.to("cpu")

            # 设置生成配置 - 更严格的参数以减少废话
            self._generation_config = GenerationConfig(
                max_new_tokens=200,  # 减少生成长度
                temperature=0.3,  # 降低温度，减少随机性
                top_p=0.9,
                repetition_penalty=1.5,  # 增加重复惩罚
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                no_repeat_ngram_size=5,  # 防止5-gram重复
                penalty_alpha=0.7,
            )

            print("✅ 语言模型加载成功！")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        return "qwen_0.6b"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs
    ) -> str:
        try:
            # 编码输入
            inputs = self._tokenizer(prompt, return_tensors="pt", padding=True)

            if self._device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            else:
                inputs = {k: v for k, v in inputs.items()}

            # 生成
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    generation_config=self._generation_config,
                    **kwargs
                )

            # 解码输出
            generated_ids = outputs[0][len(inputs["input_ids"][0]):]
            response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 处理停止词
            if stop:
                for stop_word in stop:
                    if stop_word in response:
                        response = response.split(stop_word)[0]

            # 清理输出
            response = self._clean_response(response)

            return response.strip()

        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return "抱歉，生成回答时出现了错误。"

    def _clean_response(self, response: str) -> str:
        """清理模型输出，移除所有思考过程和格式"""
        # 移除<think>标签及其内容
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

        # 移除选择题格式（A. B. C. D. 等）
        response = re.sub(r'[A-D]\.\s+.*?(?=\n[A-D]\.|\n\n|$)', '', response, flags=re.DOTALL)
        response = re.sub(r'选项[A-D]\s*[:：]?\s*', '', response)
        response = re.sub(r'正确答案?是?[：:]?\s*[A-D]', '', response)

        # 移除思考过程标记
        thought_patterns = [
            r'思考[:：].*?(?=答案[:：]|\n\n|$)',
            r'分析[:：].*?(?=答案[:：]|\n\n|$)',
            r'首先.*?然后.*?最后.*?(?=答案[:：]|\n\n|$)',
            r'根据.*?可以得出.*?(?=答案[:：]|\n\n|$)',
        ]

        for pattern in thought_patterns:
            response = re.sub(pattern, '', response, flags=re.DOTALL)

        # 移除问题分析部分
        response = re.sub(r'问题.*?问的是.*?(?=答案[:：]|\n\n|$)', '', response, flags=re.DOTALL)

        # 移除"所以"、"因此"等连接词后面的思考
        response = re.sub(r'(所以|因此|综上|由此可见|总的来说|总而言之).*?(?=答案[:：]|\n\n|$)', '', response,
                          flags=re.DOTALL)

        # 移除答案标记，保留实际内容
        response = re.sub(r'答案[：:]?\s*', '', response)

        # 移除多余的空行和空白字符
        lines = response.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # 跳过空行和明显的思考过程行
            if line and not self._is_thought_line(line):
                cleaned_lines.append(line)

        # 合并行
        cleaned_response = ' '.join(cleaned_lines)

        # 移除多余的空格
        cleaned_response = ' '.join(cleaned_response.split())

        # 如果清理后为空，返回默认信息
        if not cleaned_response or len(cleaned_response) < 10:
            return "根据文档内容，我无法生成有效的回答。请尝试更具体的问题。"

        return cleaned_response

    def _is_thought_line(self, line: str) -> bool:
        """判断一行是否是思考过程"""
        thought_keywords = [
            '思考：', '分析：', '首先', '然后', '最后',
            '所以', '因此', '综上', '由此可见', '总的来说',
            '选项', '正确答案', '应该选', '选择'
        ]

        line_lower = line.lower()
        for keyword in thought_keywords:
            if keyword in line_lower:
                return True
        return False

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }


# ================ 3. 文档处理器（支持PDF） ================
class DocumentProcessor:
    """文档处理器，支持PDF、TXT、MD格式"""

    def __init__(self, chunk_size=300, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=True
        )

    def load_pdf_document(self, file_path: str) -> List[Document]:
        """加载PDF文档"""
        documents = []

        try:
            print(f"📖 正在读取PDF文件: {Path(file_path).name}")

            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"   → PDF总页数: {total_pages}")

                for page_num, page in enumerate(pdf.pages, 1):
                    # 提取文本
                    text = page.extract_text()

                    if text:
                        # 清理文本
                        text = ' '.join(text.split())

                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": file_path,
                                "type": "pdf",
                                "page": page_num,
                                "total_pages": total_pages
                            }
                        )
                        documents.append(doc)
                        print(f"   → 处理第 {page_num}/{total_pages} 页")

            if not documents:
                print(f"⚠️  PDF文件 {Path(file_path).name} 没有提取到文本内容")

        except Exception as e:
            print(f"❌ 读取PDF文件失败 {file_path}: {e}")

        return documents

    def load_txt_document(self, file_path: str) -> List[Document]:
        """加载TXT文档"""
        documents = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            doc = Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "type": "text"
                }
            )
            documents.append(doc)

        except Exception as e:
            print(f"❌ 读取TXT文件失败 {file_path}: {e}")

        return documents

    def load_md_document(self, file_path: str) -> List[Document]:
        """加载MD文档"""
        documents = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            doc = Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "type": "markdown"
                }
            )
            documents.append(doc)

        except Exception as e:
            print(f"❌ 读取MD文件失败 {file_path}: {e}")

        return documents

    def load_documents(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            return self.load_pdf_document(file_path)
        elif file_ext == '.txt':
            return self.load_txt_document(file_path)
        elif file_ext in ['.md', '.markdown']:
            return self.load_md_document(file_path)
        else:
            print(f"⚠️  不支持的文件格式: {file_ext}，尝试按文本格式读取")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "type": "unknown"
                    }
                )
                return [doc]
            except Exception as e:
                print(f"❌ 无法读取文件: {file_path}, 错误: {e}")
                return []

    def process_file(self, file_path: str) -> List[Document]:
        """处理单个文件并返回文档块"""
        all_chunks = []

        if not Path(file_path).exists():
            print(f"❌ 文件不存在: {file_path}")
            return all_chunks

        try:
            print(f"📄 处理文件: {Path(file_path).name}")
            docs = self.load_documents(file_path)

            if docs:
                chunks = self.text_splitter.split_documents(docs)

                # 添加块信息
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "file_name": Path(file_path).name,
                        "file_type": Path(file_path).suffix.lower()[1:],
                        "processed_time": time.strftime("%Y-%m-%d %H:%M:%S")
                    })

                all_chunks.extend(chunks)
                print(f"✅ 文件处理完成，分成 {len(chunks)} 个文本块")
            else:
                print(f"❌ 文件为空或无法读取")

        except Exception as e:
            print(f"❌ 处理失败 {Path(file_path).name}: {e}")

        return all_chunks


# ================ 4. ChromaDB向量存储管理器 ================
class ChromaDBManager:
    """ChromaDB向量存储管理器"""

    def __init__(self, embedding_model, persist_directory: str = None):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory or str(CHROMA_DB_DIR)
        self.vector_store = None
        self.collection_name = "qwen_rag_collection"

        # 创建持久化目录
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        print(f"📂 ChromaDB持久化目录: {self.persist_directory}")

    def create_from_documents(self, documents: List[Document]):
        """从文档创建向量存储"""
        print("🔄 正在创建ChromaDB向量存储...")

        # 创建Chroma向量存储
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )

        print(f"✅ ChromaDB向量存储创建完成，包含 {len(documents)} 个文档块")
        return self.vector_store

    def add_documents(self, documents: List[Document]):
        """添加文档到现有向量存储"""
        print("🔄 正在添加文档到ChromaDB向量存储...")

        if not self.vector_store:
            print("⚠️  向量存储不存在，正在创建新的...")
            return self.create_from_documents(documents)

        try:
            # 添加到现有向量存储
            self.vector_store.add_documents(documents)
            print(f"✅ 成功添加 {len(documents)} 个文档块")
            return True
        except Exception as e:
            print(f"❌ 添加文档失败: {e}")
            return False

    def load(self):
        """加载向量存储"""
        try:
            if not os.path.exists(self.persist_directory):
                print(f"⚠️  持久化目录不存在: {self.persist_directory}")
                return None

            # 检查目录是否为空
            if not any(Path(self.persist_directory).iterdir()):
                print(f"⚠️  持久化目录为空: {self.persist_directory}")
                return None

            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model.embeddings,
                collection_name=self.collection_name
            )

            # 测试连接是否正常
            try:
                # 尝试一个简单的搜索来验证向量存储是否正常工作
                _ = self.vector_store.similarity_search("test", k=1)
                print(f"📂 ChromaDB向量存储已加载")
                return self.vector_store
            except Exception as e:
                print(f"❌ 向量存储加载失败（可能损坏）: {e}")
                return None

        except Exception as e:
            print(f"❌ 加载ChromaDB向量存储失败: {e}")
            return None

    def search(self, query: str, k: int = 4, score_threshold: float = 0.5):
        """搜索相似文档"""
        if not self.vector_store:
            print("⚠️  向量存储未加载")
            return []

        try:
            # ChromaDB的相似度搜索
            results = self.vector_store.similarity_search_with_relevance_scores(
                query,
                k=k
            )

            # 转换结果格式
            filtered_results = []
            for doc, score in results:
                # ChromaDB的分数是相似度（越大越相似）
                if score >= score_threshold:
                    filtered_results.append((doc, score))

            return filtered_results

        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            # 尝试不带分数的搜索
            try:
                docs = self.vector_store.similarity_search(query, k=k)
                return [(doc, 1.0) for doc in docs]
            except:
                return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self.vector_store:
            return {"error": "向量存储未加载"}

        try:
            # 获取集合中的文档数量
            collection = self.vector_store._collection
            count = collection.count() if collection else "未知"

            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "status": "已加载"
            }
        except Exception as e:
            return {"error": str(e)}


# ================ 5. RAG流水线 ================
class QwenRAGPipeline:
    """Qwen RAG流水线"""

    def __init__(self, llm, vector_store_manager):
        self.llm = llm
        self.vector_manager = vector_store_manager

        # 优化的提示模板 - 更严格，要求直接回答
        self.prompt_template = """基于以下上下文信息，直接回答问题。不要输出任何思考过程，不要输出选择题格式，直接给出答案。

上下文信息：
{context}

问题：{question}

要求：
1. 只使用上下文中的信息回答问题
2. 如果上下文没有相关信息，直接说"我不知道"
3. 不要输出任何"思考："、"分析："、"首先"、"然后"、"最后"等思考过程
4. 不要输出任何选择题格式（如A. B. C. D.）
5. 直接给出简洁明了的答案

答案："""

    def build_context(self, search_results, max_length: int = 1500) -> str:
        """构建上下文字符串"""
        context_parts = []
        current_length = 0

        for i, (doc, score) in enumerate(search_results):
            doc_text = doc.page_content

            if current_length + len(doc_text) > max_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n".join(context_parts)

    def query(self, question: str, k: int = 3, score_threshold: float = 0.3) -> Dict[str, Any]:
        """执行查询"""
        print(f"\n🔍 检索中: '{question}'")

        # 1. 检索相关文档
        search_results = self.vector_manager.search(
            question,
            k=k,
            score_threshold=score_threshold
        )

        if not search_results:
            return {
                "question": question,
                "answer": "没有在文档中找到相关信息，无法回答这个问题。",
                "sources": [],
                "context": ""
            }

        # 2. 构建上下文
        context = self.build_context(search_results)

        # 3. 构建完整提示
        full_prompt = self.prompt_template.format(
            context=context,
            question=question
        )

        # 4. 调用LLM生成回答
        print("🤖 生成回答中...")
        answer = self.llm._call(full_prompt)

        # 5. 进一步清理回答
        answer = self._post_process_answer(answer)

        # 6. 准备返回结果
        result = {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "source_info": f"{doc.metadata.get('file_name', '未知文件')} "
                                   f"(类型: {doc.metadata.get('type', '未知')})"
                }
                for doc, score in search_results
            ],
            "context_length": len(context),
            "source_count": len(search_results)
        }

        return result

    def _post_process_answer(self, answer: str) -> str:
        """后处理答案，确保格式正确"""
        # 移除答案开头的"答案："字样
        answer = re.sub(r'^答案[：:]\s*', '', answer)

        # 移除任何剩余的思考标记
        thought_patterns = [
            r'思考[:：].*',
            r'分析[:：].*',
            r'首先，.*',
            r'然后，.*',
            r'最后，.*',
            r'所以，.*',
            r'因此，.*',
            r'综上，.*',
            r'由此可见，.*',
        ]

        for pattern in thought_patterns:
            answer = re.sub(pattern, '', answer)

        # 移除多余的空格和空行
        answer = ' '.join(answer.split())

        # 如果回答以"答案"开头但没内容，重新处理
        if answer.startswith('答案') and len(answer) < 20:
            answer = "根据文档内容，我无法生成有效的回答。"

        return answer.strip()


# ================ 6. 文件上传和向量化管理器 ================
class FileVectorizationManager:
    """文件上传和向量化管理器"""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.processed_files = []

        # 确保目录存在
        self.data_dir.mkdir(exist_ok=True)

        print(f"📁 数据目录: {self.data_dir}")

    def upload_and_vectorize(self, source_path: str, processor: DocumentProcessor,
                             vector_manager: ChromaDBManager) -> Tuple[bool, str, List[Document]]:
        """上传文件并立即转换为向量，一步到位"""
        if not os.path.exists(source_path):
            print(f"❌ 源文件不存在: {source_path}")
            return False, "文件不存在", []

        try:
            filename = Path(source_path).name
            destination = self.data_dir / filename

            # 如果目标文件已存在，添加时间戳
            if destination.exists():
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                name = Path(source_path).stem
                ext = Path(source_path).suffix
                filename = f"{name}_{timestamp}{ext}"
                destination = self.data_dir / filename

            # 复制文件到数据目录
            shutil.copy2(source_path, destination)
            print(f"✅ 文件已复制到数据目录: {destination}")

            # 立即处理文件并转换为向量
            print(f"🔄 正在处理文件并转换为向量...")

            # 处理文件
            documents = processor.process_file(str(destination))

            if not documents:
                return False, "文件处理失败，无法提取文本内容", []

            # 添加到向量存储
            if vector_manager.vector_store:
                success = vector_manager.add_documents(documents)
            else:
                # 创建新的向量存储
                vector_manager.create_from_documents(documents)
                success = True

            if success:
                self.processed_files.append(str(destination))
                return True, f"文件 '{filename}' 已成功上传并转换为向量，添加到知识库", documents
            else:
                return False, "添加到向量存储失败", documents

        except Exception as e:
            print(f"❌ 上传和向量化失败: {e}")
            return False, str(e), []

    def upload_multiple_files(self, source_paths: List[str], processor: DocumentProcessor,
                              vector_manager: ChromaDBManager) -> Dict[str, Any]:
        """批量上传文件并转换为向量"""
        results = {
            "success": [],
            "failed": [],
            "total_documents": 0
        }

        for source_path in source_paths:
            success, message, documents = self.upload_and_vectorize(
                source_path, processor, vector_manager
            )

            if success:
                results["success"].append({
                    "file": Path(source_path).name,
                    "message": message,
                    "document_count": len(documents)
                })
                results["total_documents"] += len(documents)
            else:
                results["failed"].append({
                    "file": Path(source_path).name,
                    "message": message
                })

        return results

    def list_data_files(self) -> List[str]:
        """列出数据目录中的文件"""
        files = []
        for file_path in self.data_dir.glob("*"):
            if file_path.is_file():
                files.append(file_path.name)
        return files

    def delete_data_file(self, filename: str) -> bool:
        """删除数据目录中的文件"""
        file_path = self.data_dir / filename
        if not file_path.exists():
            print(f"❌ 文件不存在: {filename}")
            return False

        try:
            file_path.unlink()
            print(f"🗑️  已删除文件: {filename}")
            return True
        except Exception as e:
            print(f"❌ 删除文件失败: {e}")
            return False

    def get_file_info(self) -> List[Dict[str, Any]]:
        """获取数据目录中文件的详细信息"""
        file_info = []
        for file_path in self.data_dir.glob("*"):
            if file_path.is_file():
                info = {
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": time.strftime("%Y-%m-%d %H:%M:%S",
                                              time.localtime(file_path.stat().st_mtime)),
                    "type": file_path.suffix.lower()[1:] if file_path.suffix else "unknown"
                }
                file_info.append(info)
        return file_info


# ================ 7. 主函数 ================
def main():
    print("=" * 60)
    print("🚀 Qwen3-0.6B RAG 系统")
    print("📚 支持PDF、TXT、MD文档格式，上传后立即转换为向量")
    print("=" * 60)

    try:
        # 1. 初始化模型
        print("\n1️⃣ 初始化模型...")

        # 嵌入模型
        embedding_model = QwenEmbeddings(
            model_path=r"D:\projects\fastapi_langchain_env\NAIVERAG_test\model\Qwen3_Embedding_0.6B"
        )

        # 大语言模型 - 使用更严格的参数
        llm = QwenLLM(
            model_path=r"D:\projects\fastapi_langchain_env\NAIVERAG_test\model\Qwen3_0.6B",
            max_new_tokens=200,  # 限制生成长度
            temperature=0.3,  # 降低随机性
            top_p=0.9,
            repetition_penalty=1.5,  # 增加重复惩罚
            do_sample=True
        )

        # 2. 初始化文档处理器
        processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)

        # 3. 初始化向量存储管理器
        print("\n2️⃣ 初始化向量存储管理器...")
        vector_manager = ChromaDBManager(embedding_model)

        # 4. 初始化文件上传和向量化管理器
        print("\n3️⃣ 初始化文件上传和向量化管理器...")
        file_manager = FileVectorizationManager()

        # 5. 检查并加载向量存储
        existing_store = vector_manager.load()

        if existing_store:
            print("✅ 发现现有向量存储")
        else:
            print("📂 没有找到现有向量存储")

        # 6. 创建RAG流水线
        print("\n4️⃣ 创建RAG流水线...")
        rag_pipeline = QwenRAGPipeline(llm, vector_manager)

        # 7. 交互模式
        print("\n🎮 进入交互模式")
        print("-" * 60)
        print("命令列表:")
        print("  • 输入问题 - 基于文档内容回答问题")
        print("  • 'upload' - 上传文件并立即转换为向量")
        print("  • 'upload_multi' - 批量上传多个文件")
        print("  • 'list' - 列出所有已上传的文件")
        print("  • 'clear' - 清除屏幕")
        print("  • 'stats' - 查看向量存储统计")
        print("  • 'reload' - 重新处理所有文件并重建向量存储")
        print("  • 'quit'/'exit'/'q' - 退出程序")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n💬 请输入命令或问题: ").strip()

                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("\n👋 再见！")
                    break

                if user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("=" * 60)
                    print("🚀 Qwen3-0.6B RAG 系统")
                    print("📚 支持PDF、TXT、MD文档格式，上传后立即转换为向量")
                    print("=" * 60)
                    continue

                if user_input.lower() == 'stats':
                    stats = vector_manager.get_collection_stats()
                    print(f"\n📊 向量存储统计:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")

                    # 显示文件信息
                    file_info = file_manager.get_file_info()
                    if file_info:
                        print(f"\n📁 数据目录文件 ({len(file_info)} 个):")
                        for i, info in enumerate(file_info, 1):
                            size_kb = info['size'] / 1024
                            print(
                                f"  [{i}] {info['name']} ({size_kb:.1f} KB, {info['type']}, 修改时间: {info['modified']})")
                    continue

                if user_input.lower() == 'list':
                    files = file_manager.list_data_files()
                    print(f"\n📁 数据目录中的文件 ({len(files)} 个):")
                    if files:
                        for i, filename in enumerate(files, 1):
                            file_path = DATA_DIR / filename
                            if file_path.exists():
                                size_kb = file_path.stat().st_size / 1024
                                print(f"  [{i}] {filename} ({size_kb:.1f} KB)")
                            else:
                                print(f"  [{i}] {filename}")
                    else:
                        print("  (空)")
                    continue

                if user_input.lower() == 'upload':
                    print("\n📤 上传文件并立即转换为向量")
                    print("支持格式: PDF, TXT, MD")
                    source_path = input("请输入文件路径 (或输入 'cancel' 取消): ").strip()

                    if source_path.lower() == 'cancel':
                        print("上传已取消")
                        continue

                    if not source_path:
                        print("❌ 请输入有效的文件路径")
                        continue

                    # 上传并立即向量化
                    start_time = time.time()
                    success, message, documents = file_manager.upload_and_vectorize(
                        source_path, processor, vector_manager
                    )
                    elapsed_time = time.time() - start_time

                    if success:
                        print(f"✅ {message}")
                        print(f"⏱️  处理时间: {elapsed_time:.2f}秒")
                        print(f"📄 生成 {len(documents)} 个文本块")
                    else:
                        print(f"❌ {message}")
                    continue

                if user_input.lower() == 'upload_multi':
                    print("\n📤 批量上传多个文件并立即转换为向量")
                    print("支持格式: PDF, TXT, MD")
                    print("请输入文件路径，用逗号或分号分隔多个文件")
                    print("示例: /path/to/file1.pdf, /path/to/file2.txt")

                    input_paths = input("请输入文件路径列表: ").strip()

                    if not input_paths:
                        print("❌ 请输入有效的文件路径")
                        continue

                    # 分割文件路径
                    paths = []
                    for path in input_paths.replace(';', ',').split(','):
                        path = path.strip()
                        if path:
                            paths.append(path)

                    if not paths:
                        print("❌ 没有找到有效的文件路径")
                        continue

                    print(f"📋 发现 {len(paths)} 个文件:")
                    for i, path in enumerate(paths, 1):
                        print(f"  [{i}] {path}")

                    confirm = input(f"确认上传这 {len(paths)} 个文件? (y/N): ").strip().lower()
                    if confirm != 'y':
                        print("批量上传已取消")
                        continue

                    # 批量上传并向量化
                    start_time = time.time()
                    results = file_manager.upload_multiple_files(paths, processor, vector_manager)
                    elapsed_time = time.time() - start_time

                    print(f"\n📊 批量上传结果:")
                    print(f"⏱️  总处理时间: {elapsed_time:.2f}秒")
                    print(f"✅ 成功: {len(results['success'])} 个文件")
                    print(f"❌ 失败: {len(results['failed'])} 个文件")
                    print(f"📄 总文档块数: {results['total_documents']}")

                    if results['success']:
                        print("\n📋 成功文件:")
                        for i, success_info in enumerate(results['success'], 1):
                            print(f"  [{i}] {success_info['file']} - {success_info['document_count']} 个文档块")

                    if results['failed']:
                        print("\n❌ 失败文件:")
                        for i, failed_info in enumerate(results['failed'], 1):
                            print(f"  [{i}] {failed_info['file']} - {failed_info['message']}")
                    continue

                if user_input.lower() == 'reload':
                    print("\n🔄 重新处理所有文件并重建向量存储")
                    files = file_manager.list_data_files()

                    if not files:
                        print("❌ 数据目录中没有文件")
                        continue

                    print(f"📋 数据目录中有 {len(files)} 个文件:")
                    for i, filename in enumerate(files, 1):
                        print(f"  [{i}] {filename}")

                    confirm = input("确认重新处理所有文件并重建向量存储? (y/N): ").strip().lower()

                    if confirm == 'y':
                        # 收集所有文档
                        all_documents = []
                        for filename in files:
                            file_path = DATA_DIR / filename
                            documents = processor.process_file(str(file_path))
                            all_documents.extend(documents)
                            print(f"✅ 已处理: {filename} ({len(documents)} 个文档块)")

                        if all_documents:
                            # 删除旧的向量存储并创建新的
                            if CHROMA_DB_DIR.exists():
                                shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
                                time.sleep(1)

                            # 创建新的向量存储
                            print(f"\n🔄 正在创建新的向量存储，包含 {len(all_documents)} 个文档块...")
                            vector_manager.create_from_documents(all_documents)

                            # 显示统计信息
                            stats = vector_manager.get_collection_stats()
                            print(f"\n📊 向量存储统计:")
                            for key, value in stats.items():
                                print(f"  {key}: {value}")
                        else:
                            print("❌ 没有处理任何文档")
                    continue

                # 如果不是命令，则作为问题处理
                if not user_input:
                    continue

                # 清理用户输入（移除末尾的"-"等特殊字符）
                user_input = user_input.rstrip('-').strip()

                # 执行查询
                start_time = time.time()
                result = rag_pipeline.query(user_input, k=4)
                elapsed_time = time.time() - start_time

                # 显示结果
                print(f"\n📝 回答: {result['answer']}")
                print(f"⏱️  处理时间: {elapsed_time:.2f}秒")
                print(f"📊 检索到 {result['source_count']} 个相关文档块")

                # 显示详细来源
                if result['sources']:
                    print(f"\n🔍 检索来源:")
                    for i, source in enumerate(result['sources']):
                        print(f"\n  [{i + 1}] {source['source_info']}")
                        print(f"      相关度: {source['score']:.4f}")
                        # 清理来源内容中的特殊字符
                        clean_content = re.sub(r'[_\-\*\/\\]', ' ', source['content'])
                        print(f"      内容: {clean_content[:200]}...")

                print("\n" + "-" * 60)

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
                continue

    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


# ================ 运行 ================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-0.6B RAG系统 - 上传文件后立即转换为向量")
    parser.add_argument("--data", type=str, default="data", help="数据目录路径")
    parser.add_argument("--reset", action="store_true", help="重置向量存储（删除已有的）")

    args = parser.parse_args()

    if args.data != "data":
        DATA_DIR = Path(args.data)
        if not DATA_DIR.exists():
            print(f"❌ 指定的数据目录不存在: {args.data}")
            sys.exit(1)

    if args.reset:
        # 删除向量存储
        chroma_dir = CHROMA_DB_DIR
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
            print(f"🗑️  已删除向量存储目录: {chroma_dir}")
            time.sleep(0.5)  # 等待文件系统更新

    sys.exit(main())