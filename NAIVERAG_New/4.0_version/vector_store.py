"""
向量存储和文件管理模块：整合向量数据库和文件上传功能
"""
import os
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import DATA_DIR


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

    def create_from_documents(self, documents: List[Document]):
        """从文档创建向量存储"""
        print("🔄 正在创建ChromaDB向量存储...")

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,  # 直接使用embedding_model，它有embed_documents方法
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
                embedding_function=self.embedding_model,  # 直接使用embedding_model
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


class FileVectorizationManager:
    """文件上传和向量化管理器"""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.processed_files = []
        self.data_dir.mkdir(exist_ok=True)
        print(f"📁 数据目录: {self.data_dir}")

    def upload_and_vectorize(self, source_path: str, processor, vector_manager) -> Tuple[bool, str, List]:
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

            # 处理文件
            print(f"🔄 正在处理文件并转换为向量...")
            documents = processor.process_file(str(destination))

            if not documents:
                return False, "文件处理失败，无法提取文本内容", []

            # 添加到向量存储
            if vector_manager.vector_store:
                success = vector_manager.add_documents(documents)
            else:
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

    def upload_multiple_files(self, source_paths: List[str], processor, vector_manager) -> Dict[str, Any]:
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