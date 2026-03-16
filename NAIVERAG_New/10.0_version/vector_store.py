"""
向量存储和文件管理模块：整合向量数据库和文件上传功能
"""
import os
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import DATA_DIR
from retrieval import HybridRetriever, BM25Retriever


class ChromaDBManager:
    """ChromaDB向量存储管理器 - 适配MultiEmbeddings"""

    def __init__(self, embedding_model, persist_directory: str = None):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory or "chroma_db"
        self.vector_store = None
        self.collection_name = "qwen_rag_collection"

        # 创建持久化目录
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        print(f"📂 ChromaDB持久化目录: {self.persist_directory}")

    def create_from_documents(self, documents: List[Document]) -> List[str]:
        """从文档创建向量存储，返回文档文本列表"""
        print("🔄 正在创建ChromaDB向量存储...")
        document_texts = [doc.page_content for doc in documents]

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )

        print(f"✅ ChromaDB向量存储创建完成，包含 {len(documents)} 个文档块")
        return document_texts

    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到现有向量存储，返回文档文本列表"""
        print("🔄 正在添加文档到ChromaDB向量存储...")
        if not self.vector_store:
            print("⚠️  向量存储不存在，正在创建新的...")
            return self.create_from_documents(documents)

        try:
            self.vector_store.add_documents(documents)
            print(f"✅ 成功添加 {len(documents)} 个文档块")
            return [doc.page_content for doc in documents]
        except Exception as e:
            print(f"❌ 添加文档失败: {e}")
            return []

    def load(self):
        """加载向量存储"""
        try:
            if not os.path.exists(self.persist_directory):
                print(f"⚠️  持久化目录不存在: {self.persist_directory}")
                return None
            if not any(Path(self.persist_directory).iterdir()):
                print(f"⚠️  持久化目录为空: {self.persist_directory}")
                return None

            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )

            # 测试连接
            try:
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
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            filtered_results = []
            for doc, score in results:
                if score >= score_threshold:
                    filtered_results.append((doc, score))
            return filtered_results
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
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

    def get_all_documents(self) -> List[str]:
        """获取所有文档的文本内容（用于BM25索引）"""
        if not self.vector_store:
            return []
        try:
            collection = self.vector_store._collection
            if not collection:
                return []
            results = collection.get()
            if results and 'documents' in results:
                return results['documents']
            else:
                return []
        except Exception as e:
            print(f"❌ 获取所有文档失败: {e}")
            return []


class HybridRetrievalManager:
    """混合检索管理器"""

    def __init__(self,
                 chroma_manager: ChromaDBManager,
                 reranker_model=None,
                 retrieval_mode: str = "hybrid",
                 hybrid_weights: Tuple[float, float] = (0.4, 0.6),
                 reranker_enabled: bool = True,
                 reranker_top_k: int = 4):
        """
        初始化混合检索管理器
        Args:
            chroma_manager: ChromaDB管理器
            reranker_model: 重排模型
            retrieval_mode: 检索模式
            hybrid_weights: 混合权重
            reranker_enabled: 是否启用重排
            reranker_top_k: 重排后返回数量
        """
        self.chroma_manager = chroma_manager
        self.reranker_model = reranker_model

        # 初始化BM25检索器
        self.bm25_retriever = BM25Retriever()

        # 初始化混合检索器
        self.hybrid_retriever = HybridRetriever(
            embedding_retriever=chroma_manager,
            bm25_retriever=self.bm25_retriever,
            reranker_model=reranker_model,
            retrieval_mode=retrieval_mode,
            hybrid_weights=hybrid_weights,
            reranker_enabled=reranker_enabled,
            reranker_top_k=reranker_top_k
        )

        # 加载现有文档到BM25索引
        self._load_documents_to_bm25()

    def _load_documents_to_bm25(self):
        """从向量存储加载文档到BM25索引"""
        try:
            documents = self.chroma_manager.get_all_documents()
            if documents:
                self.bm25_retriever.update_documents(documents)
                print(f"📊 已加载 {len(documents)} 个文档到BM25索引")
            else:
                print("⚠️  向量存储中没有文档，BM25索引为空")
        except Exception as e:
            print(f"❌ 加载文档到BM25索引失败: {e}")

    def add_documents(self, document_texts: List[str]):
        """添加文档到BM25索引"""
        if document_texts:
            self.bm25_retriever.add_documents(document_texts)
            print(f"📊 已添加 {len(document_texts)} 个文档到BM25索引")

    def search(self, query: str, k: int = 10, score_threshold: float = 0.3) -> List[Tuple[str, float]]:
        """执行混合检索"""
        return self.hybrid_retriever.search(query, k, score_threshold)

    def update_retrieval_config(self,
                                retrieval_mode: str = None,
                                hybrid_weights: Tuple[float, float] = None,
                                reranker_enabled: bool = None,
                                reranker_top_k: int = None):
        """更新检索配置"""
        if retrieval_mode:
            self.hybrid_retriever.retrieval_mode = retrieval_mode
        if hybrid_weights:
            self.hybrid_retriever.hybrid_weights = hybrid_weights
        if reranker_enabled is not None:
            self.hybrid_retriever.reranker_enabled = reranker_enabled
        if reranker_top_k:
            self.hybrid_retriever.reranker_top_k = reranker_top_k
        print(f"🔄 已更新检索配置: 模式={self.hybrid_retriever.retrieval_mode}")

    def get_retrieval_info(self) -> Dict[str, Any]:
        """获取检索信息"""
        return self.hybrid_retriever.get_retrieval_info()


class FileVectorizationManager:
    """文件上传和向量化管理器（支持混合检索自动同步）"""

    def __init__(self, data_dir: Path = DATA_DIR, hybrid_manager: Optional[HybridRetrievalManager] = None):
        """
        初始化文件管理器
        Args:
            data_dir: 数据存储目录
            hybrid_manager: 混合检索管理器实例（用于自动更新BM25索引）
        """
        self.data_dir = data_dir
        self.hybrid_manager = hybrid_manager  # 保存实例，供上传方法使用
        self.processed_files = []
        self.data_dir.mkdir(exist_ok=True)
        print(f"📁 数据目录: {self.data_dir}")

    def upload_and_vectorize(self,
                            source_path: str,
                            processor,
                            vector_manager: ChromaDBManager,
                            hybrid_manager: Optional[HybridRetrievalManager] = None) -> Tuple[bool, str, List]:
        """
        上传文件并立即转换为向量，一步到位
        Args:
            source_path: 源文件路径
            processor: 文档处理器
            vector_manager: 向量存储管理器
            hybrid_manager: 混合检索管理器（若传入则优先使用，否则使用self.hybrid_manager）
        Returns:
            (是否成功, 消息, 文档块列表)
        """
        # 确定使用的混合检索管理器
        hm = hybrid_manager or self.hybrid_manager

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

            # 处理文件
            print(f"🔄 正在处理文件并转换为向量...")
            documents = processor.process_file(str(destination))

            if not documents:
                return False, "文件处理失败，无法提取文本内容", []

            # 添加到向量存储
            if vector_manager.vector_store:
                document_texts = vector_manager.add_documents(documents)
            else:
                document_texts = vector_manager.create_from_documents(documents)

            # 如果有混合检索管理器，添加到BM25索引
            if hm and document_texts:
                hm.add_documents(document_texts)

            if document_texts:
                self.processed_files.append(str(destination))
                return True, f"文件 '{filename}' 已成功上传并转换为向量，添加到知识库", documents
            else:
                return False, "添加到向量存储失败", documents

        except Exception as e:
            print(f"❌ 上传和向量化失败: {e}")
            return False, str(e), []

    def upload_multiple_files(self,
                             source_paths: List[str],
                             processor,
                             vector_manager: ChromaDBManager,
                             hybrid_manager: Optional[HybridRetrievalManager] = None) -> Dict[str, Any]:
        """
        批量上传文件并转换为向量
        Args:
            source_paths: 源文件路径列表
            processor: 文档处理器
            vector_manager: 向量存储管理器
            hybrid_manager: 混合检索管理器
        Returns:
            处理结果统计
        """
        results = {
            "success": [],
            "failed": [],
            "total_documents": 0
        }

        for source_path in source_paths:
            success, message, documents = self.upload_and_vectorize(
                source_path, processor, vector_manager, hybrid_manager
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