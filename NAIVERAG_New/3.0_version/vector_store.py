"""
向量存储管理器模块
"""
import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma


class ChromaDBManager:
    """ChromaDB向量存储管理器"""

    def __init__(self, embedding_model, persist_directory: str = None):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory or "chroma_db"
        self.vector_store = None
        self.collection_name = "qwen_rag_collection"

        # 创建持久化目录
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        print(f"📂 ChromaDB持久化目录: {self.persist_directory}")

    def create_from_documents(self, documents: List[Document]):
        """从文档创建向量存储"""
        print("🔄 正在创建ChromaDB向量存储...")

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

            if not any(Path(self.persist_directory).iterdir()):
                print(f"⚠️  持久化目录为空: {self.persist_directory}")
                return None

            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model.embeddings,
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
            results = self.vector_store.similarity_search_with_relevance_scores(
                query,
                k=k
            )

            # 过滤结果
            filtered_results = []
            for doc, score in results:
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