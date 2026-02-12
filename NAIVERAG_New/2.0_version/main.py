# #!/usr/bin/env python3
# """
# 使用Qwen3-0.6B和Qwen3-Embedding-0.6B的完整RAG系统
# 使用ChromaDB作为向量数据库 (简化版本)
# """
#
# import os
# import sys
# import torch
# import shutil
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# from pydantic import Field
#
# import warnings
# from transformers import logging
#
# # 关闭transformers的特定警告
# logging.set_verbosity_error()
# warnings.filterwarnings("ignore",
#                         message=".*generation_config.*default values have been modified.*")
#
# # 设置环境变量
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
#
# # 项目路径
# BASE_DIR = Path(__file__).parent
# DATA_DIR = BASE_DIR / "data"
# CHROMA_DB_DIR = BASE_DIR / "chroma_db"
# MODELS_DIR = BASE_DIR / "models"
#
# # 创建目录
# for dir_path in [DATA_DIR, CHROMA_DB_DIR, MODELS_DIR]:
#     dir_path.mkdir(exist_ok=True)
#
#
# # ================ 导入检查 ================
# def check_imports():
#     """检查必要的导入"""
#     try:
#         from langchain_huggingface import HuggingFaceEmbeddings
#         print("✅ 使用 langchain_huggingface.HuggingFaceEmbeddings")
#     except ImportError:
#         try:
#             from langchain_community.embeddings import HuggingFaceEmbeddings
#             print("⚠️  使用 langchain_community.HuggingFaceEmbeddings")
#         except ImportError:
#             print("❌ 无法导入 HuggingFaceEmbeddings")
#             print("请安装: pip install langchain-huggingface 或 langchain-community")
#             return False
#
#     try:
#         from langchain_chroma import Chroma
#         print("✅ 使用 langchain_chroma.Chroma")
#         return True
#     except ImportError as e:
#         print(f"❌ 无法导入 ChromaDB: {e}")
#         print("请安装: pip install chromadb langchain-chroma")
#         return False
#
#
# if not check_imports():
#     sys.exit(1)
#
# # 现在导入需要的模块
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
# from langchain_core.language_models.llms import LLM
# from langchain_core.callbacks.manager import CallbackManagerForLLMRun
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
#
#
# # ================ 1. 嵌入模型适配 ================
# class QwenEmbeddings:
#     """适配Qwen3-Embedding-0.6B的嵌入模型"""
#
#     def __init__(self, model_path: str = None):
#         if model_path is None:
#             model_path = str(MODELS_DIR / "Qwen3_Embedding_0.6B")
#
#         print(f"🔧 加载嵌入模型: {model_path}")
#
#         # Qwen嵌入模型需要特殊的加载方式
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name=model_path,
#             model_kwargs={
#                 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#                 'trust_remote_code': True,
#             },
#             encode_kwargs={
#                 'normalize_embeddings': True,
#                 'batch_size': 8,
#             }
#         )
#
#         try:
#             test_embedding = self.embeddings.embed_query("测试文本")
#             print(f"✅ 嵌入模型加载成功，向量维度: {len(test_embedding)}")
#         except Exception as e:
#             print(f"⚠️  嵌入模型测试时出错: {e}")
#
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return self.embeddings.embed_documents(texts)
#
#     def embed_query(self, text: str) -> List[float]:
#         return self.embeddings.embed_query(text)
#
#
# # ================ 2. 大语言模型适配 ================
# class QwenLLM(LLM):
#     """适配Qwen3-0.6B的大语言模型"""
#
#     # 使用Pydantic Field声明字段
#     model_path: str = Field(default="Qwen3_0.6B", description="模型路径")
#     max_new_tokens: int = Field(default=512, description="生成的最大token数")
#     temperature: float = Field(default=0.1, description="温度参数")
#     top_p: float = Field(default=0.9, description="top-p采样参数")
#     repetition_penalty: float = Field(default=1.1, description="重复惩罚")
#     do_sample: bool = Field(default=True, description="是否采样")
#
#     # 内部使用的字段
#     _tokenizer: Any = None
#     _model: Any = None
#     _generation_config: Any = None
#     _device: str = "cpu"
#
#     def __init__(self, **data):
#         super().__init__(**data)
#
#         print(f"🔧 加载语言模型: {self.model_path}")
#
#         # 检查路径是否存在
#         if not os.path.exists(self.model_path):
#             print(f"⚠️  模型路径不存在: {self.model_path}")
#             print("正在尝试从本地models目录查找...")
#             local_path = MODELS_DIR / "Qwen3_0.6B"
#             if local_path.exists():
#                 self.model_path = str(local_path)
#                 print(f"✅ 使用本地模型: {self.model_path}")
#
#         self._device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"📱 使用设备: {self._device}")
#
#         try:
#             # 加载tokenizer
#             self._tokenizer = AutoTokenizer.from_pretrained(
#                 self.model_path,
#                 trust_remote_code=True,
#                 padding_side="left"
#             )
#
#             # 如果没有pad_token，使用eos_token
#             if self._tokenizer.pad_token is None:
#                 self._tokenizer.pad_token = self._tokenizer.eos_token
#
#             # 加载模型
#             torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
#             self._model = AutoModelForCausalLM.from_pretrained(
#                 self.model_path,
#                 trust_remote_code=True,
#                 torch_dtype=torch_dtype,
#                 device_map="auto" if self._device == "cuda" else None,
#                 low_cpu_mem_usage=True
#             )
#
#             if self._device == "cpu":
#                 self._model = self._model.to("cpu")
#
#             # 设置生成配置
#             self._generation_config = GenerationConfig(
#                 max_new_tokens=self.max_new_tokens,
#                 temperature=self.temperature,
#                 top_p=self.top_p,
#                 repetition_penalty=self.repetition_penalty,
#                 do_sample=self.do_sample,
#                 pad_token_id=self._tokenizer.pad_token_id,
#                 eos_token_id=self._tokenizer.eos_token_id
#             )
#
#             print("✅ 语言模型加载成功！")
#
#         except Exception as e:
#             print(f"❌ 模型加载失败: {e}")
#             raise
#
#     @property
#     def _llm_type(self) -> str:
#         return "qwen_0.6b"
#
#     def _call(
#             self,
#             prompt: str,
#             stop: Optional[List[str]] = None,
#             run_manager: Optional[CallbackManagerForLLMRun] = None,
#             **kwargs
#     ) -> str:
#         try:
#             # 编码输入
#             inputs = self._tokenizer(prompt, return_tensors="pt", padding=True)
#
#             if self._device == "cuda":
#                 inputs = {k: v.cuda() for k, v in inputs.items()}
#             else:
#                 inputs = {k: v for k, v in inputs.items()}
#
#             # 生成
#             with torch.no_grad():
#                 outputs = self._model.generate(
#                     **inputs,
#                     generation_config=self._generation_config,
#                     **kwargs
#                 )
#
#             # 解码输出
#             generated_ids = outputs[0][len(inputs["input_ids"][0]):]
#             response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
#
#             # 处理停止词
#             if stop:
#                 for stop_word in stop:
#                     if stop_word in response:
#                         response = response.split(stop_word)[0]
#
#             return response.strip()
#
#         except Exception as e:
#             print(f"❌ 生成失败: {e}")
#             return "抱歉，生成回答时出现了错误。"
#
#     @property
#     def _identifying_params(self) -> Dict[str, Any]:
#         return {
#             "model_path": self.model_path,
#             "max_new_tokens": self.max_new_tokens,
#             "temperature": self.temperature,
#             "top_p": self.top_p
#         }
#
#
# # ================ 3. 文档处理器 ================
# class DocumentProcessor:
#     """文档处理器"""
#
#     def __init__(self, chunk_size=300, chunk_overlap=50):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             length_function=len,
#             separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
#             keep_separator=True
#         )
#
#     def load_documents(self, file_path: str) -> List[Document]:
#         """加载文档"""
#         documents = []
#         file_ext = file_path.split('.')[-1].lower()
#
#         if file_ext == 'txt':
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 text = f.read()
#                 doc = Document(
#                     page_content=text,
#                     metadata={
#                         "source": file_path,
#                         "type": "text"
#                     }
#                 )
#                 documents.append(doc)
#         elif file_ext == 'md':
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 text = f.read()
#                 doc = Document(
#                     page_content=text,
#                     metadata={
#                         "source": file_path,
#                         "type": "markdown"
#                     }
#                 )
#                 documents.append(doc)
#         else:
#             print(f"⚠️  暂不支持 {file_ext} 格式，使用文本方式读取")
#             try:
#                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#                     text = f.read()
#                     doc = Document(
#                         page_content=text,
#                         metadata={
#                             "source": file_path,
#                             "type": "unknown"
#                         }
#                     )
#                     documents.append(doc)
#             except:
#                 print(f"❌ 无法读取文件: {file_path}")
#
#         return documents
#
#     def process_directory(self, directory_path: str) -> List[Document]:
#         """处理目录下所有文档"""
#         all_chunks = []
#
#         if not Path(directory_path).exists():
#             print(f"❌ 目录不存在: {directory_path}")
#             return all_chunks
#
#         for file_path in Path(directory_path).glob("**/*"):
#             if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.md']:
#                 try:
#                     print(f"📄 处理文件: {file_path.name}")
#                     docs = self.load_documents(str(file_path))
#                     chunks = self.text_splitter.split_documents(docs)
#
#                     # 添加块信息
#                     for i, chunk in enumerate(chunks):
#                         chunk.metadata.update({
#                             "chunk_id": i,
#                             "total_chunks": len(chunks),
#                             "file_name": file_path.name
#                         })
#
#                     all_chunks.extend(chunks)
#                     print(f"   → 分成 {len(chunks)} 个块")
#
#                 except Exception as e:
#                     print(f"❌ 处理失败 {file_path.name}: {e}")
#
#         print(f"✅ 文档处理完成，共 {len(all_chunks)} 个文本块")
#         return all_chunks
#
#
# # ================ 4. ChromaDB向量存储管理器 ================
# class ChromaDBManager:
#     """ChromaDB向量存储管理器"""
#
#     def __init__(self, embedding_model, persist_directory: str = None):
#         self.embedding_model = embedding_model
#         self.persist_directory = persist_directory or str(CHROMA_DB_DIR)
#         self.vector_store = None
#         self.collection_name = "qwen_rag_collection"
#
#         # 创建持久化目录
#         Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
#
#         print(f"📂 ChromaDB持久化目录: {self.persist_directory}")
#
#     def create_from_documents(self, documents: List[Document]):
#         """从文档创建向量存储"""
#         print("🔄 正在创建ChromaDB向量存储...")
#
#         # 创建Chroma向量存储
#         self.vector_store = Chroma.from_documents(
#             documents=documents,
#             embedding=self.embedding_model.embeddings,
#             persist_directory=self.persist_directory,
#             collection_name=self.collection_name,
#         )
#
#         print(f"✅ ChromaDB向量存储创建完成，包含 {len(documents)} 个文档块")
#         return self.vector_store
#
#     def load(self):
#         """加载向量存储"""
#         try:
#             if not os.path.exists(self.persist_directory):
#                 print(f"⚠️  持久化目录不存在: {self.persist_directory}")
#                 return None
#
#             self.vector_store = Chroma(
#                 persist_directory=self.persist_directory,
#                 embedding_function=self.embedding_model.embeddings,
#                 collection_name=self.collection_name
#             )
#
#             print(f"📂 ChromaDB向量存储已加载")
#             return self.vector_store
#
#         except Exception as e:
#             print(f"❌ 加载ChromaDB向量存储失败: {e}")
#             return None
#
#     def search(self, query: str, k: int = 4, score_threshold: float = 0.5):
#         """搜索相似文档"""
#         if not self.vector_store:
#             print("⚠️  向量存储未加载，正在尝试加载...")
#             self.load()
#
#         if not self.vector_store:
#             print("❌ 无法加载向量存储")
#             return []
#
#         try:
#             # ChromaDB的相似度搜索
#             results = self.vector_store.similarity_search_with_relevance_scores(
#                 query,
#                 k=k
#             )
#
#             # 转换结果格式
#             filtered_results = []
#             for doc, score in results:
#                 # ChromaDB的分数是相似度（越大越相似）
#                 if score >= score_threshold:
#                     filtered_results.append((doc, score))
#
#             return filtered_results
#
#         except Exception as e:
#             print(f"❌ 搜索失败: {e}")
#             # 尝试不带分数的搜索
#             try:
#                 docs = self.vector_store.similarity_search(query, k=k)
#                 return [(doc, 1.0) for doc in docs]
#             except:
#                 return []
#
#     def delete_collection(self):
#         """删除集合"""
#         try:
#             if os.path.exists(self.persist_directory):
#                 shutil.rmtree(self.persist_directory)
#                 print(f"🗑️  ChromaDB集合已删除: {self.persist_directory}")
#                 self.vector_store = None
#         except Exception as e:
#             print(f"❌ 删除集合失败: {e}")
#
#     def get_collection_stats(self) -> Dict[str, Any]:
#         """获取集合统计信息"""
#         if not self.vector_store:
#             return {"error": "向量存储未加载"}
#
#         try:
#             return {
#                 "collection_name": self.collection_name,
#                 "persist_directory": self.persist_directory,
#                 "status": "已加载"
#             }
#         except Exception as e:
#             return {"error": str(e)}
#
#
# # ================ 5. RAG流水线 ================
# class QwenRAGPipeline:
#     """Qwen RAG流水线"""
#
#     def __init__(self, llm, vector_store_manager):
#         self.llm = llm
#         self.vector_manager = vector_store_manager
#
#         # Qwen专用的提示模板
#         self.prompt_template = """基于以下上下文信息回答问题。不要输出任何思考过程，直接给出答案。
#
#         上下文信息：
#         {context}
#
#         问题：{question}
#
#         要求：
#         1. 只使用上下文中的信息
#         2. 如果上下文没有相关信息，请说"我不知道"
#         3. 不要输出<think>标签或思考过程
#         4. 回答要简洁明了
#
#         答案："""
#
#     def build_context(self, search_results, max_length: int = 1500) -> str:
#         """构建上下文字符串"""
#         context_parts = []
#         current_length = 0
#
#         for i, (doc, score) in enumerate(search_results):
#             doc_text = f"[文档{i + 1}, 相关度:{score:.3f}]\n{doc.page_content}\n"
#
#             if current_length + len(doc_text) > max_length:
#                 break
#
#             context_parts.append(doc_text)
#             current_length += len(doc_text)
#
#         return "\n".join(context_parts)
#
#     def query(self, question: str, k: int = 3, score_threshold: float = 0.3) -> Dict[str, Any]:
#         """执行查询"""
#         print(f"\n🔍 检索中: '{question}'")
#
#         # 1. 检索相关文档
#         search_results = self.vector_manager.search(
#             question,
#             k=k,
#             score_threshold=score_threshold
#         )
#
#         if not search_results:
#             return {
#                 "question": question,
#                 "answer": "没有找到相关文档，无法回答这个问题。",
#                 "sources": [],
#                 "context": ""
#             }
#
#         # 2. 构建上下文
#         context = self.build_context(search_results)
#
#         # 3. 构建完整提示
#         full_prompt = self.prompt_template.format(
#             context=context,
#             question=question
#         )
#
#         # 4. 调用LLM生成回答
#         print("🤖 生成回答中...")
#         answer = self.llm._call(full_prompt)
#
#         # 5. 准备返回结果
#         result = {
#             "question": question,
#             "answer": answer,
#             "sources": [
#                 {
#                     "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
#                     "metadata": doc.metadata,
#                     "score": float(score)
#                 }
#                 for doc, score in search_results
#             ],
#             "context_length": len(context),
#             "source_count": len(search_results)
#         }
#
#         return result
#
#
# # ================ 6. 主函数 ================
# def main():
#     print("=" * 60)
#     print("🚀 Qwen3-0.6B RAG 系统 (使用ChromaDB - 简化版)")
#     print("=" * 60)
#
#     try:
#         # 1. 初始化模型
#         print("\n1️⃣ 初始化模型...")
#
#         # 嵌入模型
#         embedding_model = QwenEmbeddings(
#             model_path=r"D:\projects\fastapi_langchain_env\NAIVERAG_test\model\Qwen3_Embedding_0.6B"
#         )
#
#         # 大语言模型
#         llm = QwenLLM(
#             model_path=r"D:\projects\fastapi_langchain_env\NAIVERAG_test\model\Qwen3_0.6B",
#             max_new_tokens=256,
#             temperature=0.3,
#             top_p=0.85,
#             repetition_penalty=1.05,
#             do_sample=True
#         )
#
#         # 2. 文档处理
#         print("\n2️⃣ 处理文档...")
#         processor = DocumentProcessor(chunk_size=200, chunk_overlap=30)
#
#         # 检查是否有数据目录
#         if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
#             documents = processor.process_directory(str(DATA_DIR))
#         else:
#             print("📝 创建示例文档...")
#             # 创建一些示例文档
#             sample_texts = [
#                 "阿里巴巴集团由马云于1999年创立，总部位于杭州。",
#                 "淘宝网是阿里巴巴集团旗下的C2C电商平台，成立于2003年。",
#                 "支付宝最初是淘宝网的支付工具，后来发展成为独立的数字支付平台。",
#                 "阿里云是阿里巴巴集团的云计算服务，提供云服务器、数据库等服务。",
#                 "达摩院是阿里巴巴的研究机构，专注于人工智能、量子计算等前沿技术。",
#                 "阿里巴巴的使命是让天下没有难做的生意。",
#                 "双十一购物节是阿里巴巴集团创办的全球最大购物节。",
#                 "菜鸟网络是阿里巴巴的物流平台，提供智能物流解决方案。"
#             ]
#
#             documents = []
#             for i, text in enumerate(sample_texts):
#                 doc = Document(
#                     page_content=text,
#                     metadata={
#                         "source": f"sample_{i}.txt",
#                         "type": "sample",
#                         "chunk_id": 0,
#                         "file_name": f"sample_{i}.txt"
#                     }
#                 )
#                 documents.append(doc)
#
#             print(f"✅ 创建了 {len(documents)} 个示例文档")
#
#         # 3. 创建ChromaDB向量存储
#         print("\n3️⃣ 创建ChromaDB向量存储...")
#         vector_manager = ChromaDBManager(embedding_model)
#
#         # 检查是否已有向量存储
#         # existing_store = vector_manager.load()
#         #
#         # if existing_store:
#         #     print("📂 发现现有的向量存储，请先删除旧存储或添加新文档")
#         #     # 这里简单起见，我们删除旧的并创建新的
#         #     confirm = input("是否删除旧的向量存储并创建新的？(y/N): ").strip().lower()
#         #     if confirm == 'y':
#         #         vector_manager.delete_collection()
#         #         vector_store = vector_manager.create_from_documents(documents)
#         #     else:
#         #         print("使用现有向量存储进行查询")
#         # else:
#         #     print("📂 创建新的向量存储...")
#         #     vector_store = vector_manager.create_from_documents(documents)
#
#         # 显示统计信息
#         # stats = vector_manager.get_collection_stats()
#         # print(f"📊 集合统计: {stats}")
#
#         # 4. 创建RAG流水线
#         print("\n4️⃣ 创建RAG流水线...")
#         rag_pipeline = QwenRAGPipeline(llm, vector_manager)
#
#         # 5. 测试查询
#         print("\n5️⃣ 测试查询...")
#         print("-" * 60)
#
#         test_questions = [
#             "阿里巴巴是什么时候成立的？",
#             "淘宝网是什么？",
#             "支付宝是做什么的？",
#         ]
#
#         for question in test_questions:
#             print(f"\n❓ 问题: {question}")
#             result = rag_pipeline.query(question, k=2)
#
#             print(f"🤖 回答: {result['answer']}")
#             print(f"📊 使用了 {result['source_count']} 个来源")
#
#             # 显示来源摘要
#             if result['sources']:
#                 print("📚 来源摘要:")
#                 for i, source in enumerate(result['sources']):
#                     print(f"  [{i + 1}] {source['content']}")
#                     print(f"     相关度: {source['score']:.3f}")
#
#             print("-" * 60)
#
#         # 6. 交互模式
#         print("\n🎮 进入交互模式 (输入 'quit' 或 '退出' 结束)")
#         print("输入 'clear' 清除屏幕")
#         print("输入 'stats' 查看向量存储统计")
#         print("输入 'delete' 删除向量存储")
#         print("-" * 60)
#
#         while True:
#             try:
#                 user_input = input("\n💬 请输入问题: ").strip()
#
#                 if user_input.lower() in ['quit', 'exit', '退出', 'q']:
#                     print("\n👋 再见！")
#                     break
#
#                 if user_input.lower() == 'clear':
#                     os.system('cls' if os.name == 'nt' else 'clear')
#                     continue
#
#                 if user_input.lower() == 'stats':
#                     stats = vector_manager.get_collection_stats()
#                     print(f"\n📊 向量存储统计:")
#                     for key, value in stats.items():
#                         print(f"  {key}: {value}")
#                     continue
#
#                 if user_input.lower() == 'delete':
#                     confirm = input("⚠️  确定要删除向量存储吗？(y/N): ").strip().lower()
#                     if confirm == 'y':
#                         vector_manager.delete_collection()
#                     continue
#
#                 if not user_input:
#                     continue
#
#                 # 执行查询
#                 result = rag_pipeline.query(user_input, k=3)
#
#                 # 显示结果
#                 print(f"\n📝 回答: {result['answer']}")
#
#                 # 显示详细来源
#                 if result['sources']:
#                     print(f"\n🔍 检索结果 ({len(result['sources'])} 个):")
#                     for i, source in enumerate(result['sources']):
#                         print(f"\n  [{i + 1}] 文件: {source['metadata'].get('file_name', '未知')}")
#                         print(f"      相关度: {source['score']:.4f}")
#                         print(f"      内容: {source['content'][:150]}...")
#
#                 print("\n" + "-" * 60)
#
#             except KeyboardInterrupt:
#                 print("\n\n👋 再见！")
#                 break
#             except Exception as e:
#                 print(f"❌ 错误: {e}")
#                 continue
#
#     except Exception as e:
#         print(f"\n❌ 系统错误: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1
#
#     return 0
#
#
# # ================ 运行 ================
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Qwen3-0.6B RAG系统 (ChromaDB版)")
#     parser.add_argument("--test", action="store_true", help="测试模型")
#     parser.add_argument("--data", type=str, default="data", help="数据目录")
#     parser.add_argument("--reset", action="store_true", help="重置向量存储")
#
#     args = parser.parse_args()
#
#     if args.test:
#         print("🧪 测试模型...")
#         # 测试嵌入模型
#         try:
#             embedding = QwenEmbeddings()
#             test_embed = embedding.embed_query("测试")
#             print(f"✅ 嵌入模型测试通过，向量长度: {len(test_embed)}")
#         except Exception as e:
#             print(f"❌ 嵌入模型测试失败: {e}")
#     else:
#         if args.data != "data":
#             DATA_DIR = Path(args.data)
#
#         if args.reset:
#             # 删除向量存储
#             chroma_dir = CHROMA_DB_DIR
#             if chroma_dir.exists():
#                 shutil.rmtree(chroma_dir)
#                 print(f"🗑️  已删除向量存储目录: {chroma_dir}")
#
#         sys.exit(main())


# !/usr/bin/env python3
"""
使用Qwen3-0.6B和Qwen3-Embedding-0.6B的完整RAG系统
使用ChromaDB作为向量数据库
"""

import os
import sys
import torch
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
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

            # 设置生成配置 - 添加防止重复生成的设置
            self._generation_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=self.do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # 防止3-gram重复
                penalty_alpha=0.6,  # 减少重复
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

            # 清理输出：移除<think>标签和多余的空行
            response = self._clean_response(response)

            return response.strip()

        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return "抱歉，生成回答时出现了错误。"

    def _clean_response(self, response: str) -> str:
        """清理模型输出"""
        # 移除<think>标签
        import re
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

        # 移除重复的提示信息
        lines = response.split('\n')
        cleaned_lines = []
        seen_lines = set()

        for line in lines:
            line = line.strip()
            # 跳过空行和重复的提示信息
            if line and "注意" not in line and "如果上下文" not in line:
                if line not in seen_lines:
                    cleaned_lines.append(line)
                    seen_lines.add(line)

        # 合并行
        cleaned_response = ' '.join(cleaned_lines)

        # 移除多余的空格
        cleaned_response = ' '.join(cleaned_response.split())

        return cleaned_response

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }


# ================ 3. 文档处理器 ================
class DocumentProcessor:
    """文档处理器"""

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

    def load_documents(self, file_path: str) -> List[Document]:
        """加载文档"""
        documents = []
        file_ext = file_path.split('.')[-1].lower()

        if file_ext == 'txt':
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
        elif file_ext == 'md':
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
        else:
            print(f"⚠️  暂不支持 {file_ext} 格式，使用文本方式读取")
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
                    documents.append(doc)
            except:
                print(f"❌ 无法读取文件: {file_path}")

        return documents

    def process_directory(self, directory_path: str) -> List[Document]:
        """处理目录下所有文档"""
        all_chunks = []

        if not Path(directory_path).exists():
            print(f"❌ 目录不存在: {directory_path}")
            return all_chunks

        for file_path in Path(directory_path).glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.md']:
                try:
                    print(f"📄 处理文件: {file_path.name}")
                    docs = self.load_documents(str(file_path))
                    chunks = self.text_splitter.split_documents(docs)

                    # 添加块信息
                    for i, chunk in enumerate(chunks):
                        chunk.metadata.update({
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "file_name": file_path.name
                        })

                    all_chunks.extend(chunks)
                    print(f"   → 分成 {len(chunks)} 个块")

                except Exception as e:
                    print(f"❌ 处理失败 {file_path.name}: {e}")

        print(f"✅ 文档处理完成，共 {len(all_chunks)} 个文本块")
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
            return {
                "collection_name": self.collection_name,
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

        # 改进的提示模板 - 更简洁明确
        self.prompt_template = """请根据以下上下文信息，直接回答问题。不要输出任何思考过程、标签或额外解释。

上下文：
{context}

问题：{question}

答案："""

    def build_context(self, search_results, max_length: int = 1500) -> str:
        """构建上下文字符串"""
        context_parts = []
        current_length = 0

        for i, (doc, score) in enumerate(search_results):
            doc_text = doc.page_content

            # 简化上下文格式
            if current_length + len(doc_text) > max_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        # 用换行符连接，但不要额外标记
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
                "answer": "没有找到相关文档，无法回答这个问题。",
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

        # 5. 准备返回结果
        result = {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                for doc, score in search_results
            ],
            "context_length": len(context),
            "source_count": len(search_results)
        }

        return result


# ================ 6. 主函数 ================
def main():
    print("=" * 60)
    print("🚀 Qwen3-0.6B RAG 系统")
    print("=" * 60)

    try:
        # 1. 初始化模型
        print("\n1️⃣ 初始化模型...")

        # 嵌入模型
        embedding_model = QwenEmbeddings(
            model_path=r"D:\projects\fastapi_langchain_env\NAIVERAG_test\model\Qwen3_Embedding_0.6B"
        )

        # 大语言模型 - 调整参数减少重复
        llm = QwenLLM(
            model_path=r"D:\projects\fastapi_langchain_env\NAIVERAG_test\model\Qwen3_0.6B",
            max_new_tokens=200,  # 减少生成长度
            temperature=0.3,
            top_p=0.85,
            repetition_penalty=1.3,  # 增加重复惩罚
            do_sample=True
        )

        # 2. 文档处理
        print("\n2️⃣ 处理文档...")
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=30)

        # 检查是否有数据目录
        if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
            documents = processor.process_directory(str(DATA_DIR))
        else:
            print("📝 创建示例文档...")
            # 创建一些示例文档
            sample_texts = [
                "阿里巴巴集团由马云于1999年创立，总部位于杭州。",
                "淘宝网是阿里巴巴集团旗下的C2C电商平台，成立于2003年。",
                "支付宝最初是淘宝网的支付工具，后来发展成为独立的数字支付平台。",
                "阿里云是阿里巴巴集团的云计算服务，提供云服务器、数据库等服务。",
                "达摩院是阿里巴巴的研究机构，专注于人工智能、量子计算等前沿技术。",
                "阿里巴巴的使命是让天下没有难做的生意。",
                "双十一购物节是阿里巴巴集团创办的全球最大购物节。",
                "菜鸟网络是阿里巴巴的物流平台，提供智能物流解决方案。"
            ]

            documents = []
            for i, text in enumerate(sample_texts):
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": f"sample_{i}.txt",
                        "type": "sample",
                        "chunk_id": 0,
                        "file_name": f"sample_{i}.txt"
                    }
                )
                documents.append(doc)

            print(f"✅ 创建了 {len(documents)} 个示例文档")

        # 3. 创建ChromaDB向量存储（根据你的要求修改）
        print("\n3️⃣ 处理ChromaDB向量存储...")
        vector_manager = ChromaDBManager(embedding_model)

        # 检查是否已有向量存储
        existing_store = vector_manager.load()

        if existing_store:
            print("✅ 使用现有向量存储")
        else:
            print("📂 没有找到现有向量存储，创建新的...")
            vector_manager.create_from_documents(documents)

        # 显示统计信息
        stats = vector_manager.get_collection_stats()
        print(f"📊 集合统计: {stats}")

        # 4. 创建RAG流水线
        print("\n4️⃣ 创建RAG流水线...")
        rag_pipeline = QwenRAGPipeline(llm, vector_manager)

        # 5. 测试查询
        print("\n5️⃣ 测试查询...")
        print("-" * 60)

        test_questions = [
            "阿里巴巴是什么时候成立的？",
            "淘宝网是什么？",
            "支付宝是做什么的？",
        ]

        for question in test_questions:
            print(f"\n❓ 问题: {question}")
            result = rag_pipeline.query(question, k=2)

            print(f"🤖 回答: {result['answer']}")
            print(f"📊 使用了 {result['source_count']} 个来源")

            # 显示来源摘要
            if result['sources']:
                print("📚 来源摘要:")
                for i, source in enumerate(result['sources']):
                    print(f"  [{i + 1}] {source['content']}")
                    print(f"     相关度: {source['score']:.3f}")

            print("-" * 60)

        # 6. 交互模式
        print("\n🎮 进入交互模式 (输入 'quit' 或 '退出' 结束)")
        print("输入 'clear' 清除屏幕")
        print("输入 'stats' 查看向量存储统计")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n💬 请输入问题: ").strip()

                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("\n👋 再见！")
                    break

                if user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue

                if user_input.lower() == 'stats':
                    stats = vector_manager.get_collection_stats()
                    print(f"\n📊 向量存储统计:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue

                if not user_input:
                    continue

                # 执行查询
                result = rag_pipeline.query(user_input, k=3)

                # 显示结果
                print(f"\n📝 回答: {result['answer']}")

                # 显示详细来源
                if result['sources']:
                    print(f"\n🔍 检索结果 ({len(result['sources'])} 个):")
                    for i, source in enumerate(result['sources']):
                        print(f"\n  [{i + 1}] 文件: {source['metadata'].get('file_name', '未知')}")
                        print(f"      相关度: {source['score']:.4f}")
                        print(f"      内容: {source['content'][:150]}...")

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

    parser = argparse.ArgumentParser(description="Qwen3-0.6B RAG系统")
    parser.add_argument("--test", action="store_true", help="测试模型")
    parser.add_argument("--data", type=str, default="data", help="数据目录")
    parser.add_argument("--reset", action="store_true", help="重置向量存储（删除已有的）")

    args = parser.parse_args()

    if args.test:
        print("🧪 测试模型...")
        # 测试嵌入模型
        try:
            embedding = QwenEmbeddings()
            test_embed = embedding.embed_query("测试")
            print(f"✅ 嵌入模型测试通过，向量长度: {len(test_embed)}")
        except Exception as e:
            print(f"❌ 嵌入模型测试失败: {e}")
    else:
        if args.data != "data":
            DATA_DIR = Path(args.data)

        if args.reset:
            # 删除向量存储
            chroma_dir = CHROMA_DB_DIR
            if chroma_dir.exists():
                shutil.rmtree(chroma_dir)
                print(f"🗑️  已删除向量存储目录: {chroma_dir}")
                time.sleep(0.5)  # 等待文件系统更新

        sys.exit(main())