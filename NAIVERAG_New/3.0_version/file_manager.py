"""
文件管理器模块
"""
import os
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from config import DATA_DIR


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