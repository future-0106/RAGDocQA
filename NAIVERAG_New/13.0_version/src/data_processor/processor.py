"""法律数据批量处理器"""
import shutil
from pathlib import Path
from typing import Dict


class LegalDataProcessor:
    """法律数据批量处理器 - 加载目录文档并向量化"""
    
    def __init__(self, vector_manager, hybrid_manager, file_manager=None):
        from data_processor.legal_loader import LegalDocumentLoader
        
        self.vector_manager = vector_manager
        self.hybrid_manager = hybrid_manager
        self.file_manager = file_manager
        self.loader = LegalDocumentLoader()
    
    def _get_existing_files(self) -> set:
        """获取已存在的文件名集合"""
        if self.file_manager:
            existing_files = set(self.file_manager.list_data_files())
        else:
            existing_files = set()
        return existing_files
    
    def _copy_files_to_data_dir(self, processed_files: list) -> list:
        """复制处理成功的文件到 data_dir 目录
        
        Args:
            processed_files: 处理成功的文件路径列表
            
        Returns:
            复制后的文件名列表
        """
        if not self.file_manager:
            return []
        
        copied_files = []
        data_dir = self.file_manager.data_dir
        
        for src_path in processed_files:
            src = Path(src_path)
            dest = data_dir / src.name
            
            # 如果目标文件已存在，添加时间戳
            if dest.exists():
                timestamp = src.stat().st_mtime
                import time
                ts = time.strftime("%Y%m%d_%H%M%S")
                name = src.stem
                ext = src.suffix
                dest = data_dir / f"{name}_{ts}{ext}"
            
            try:
                shutil.copy2(src, dest)
                print(f"📄 已复制: {src.name} -> {dest.name}")
                copied_files.append(dest.name)
            except Exception as e:
                print(f"❌ 复制失败: {src.name} - {e}")
        
        return copied_files
    
    def process_legal_documents(self, source_dir: str) -> Dict:
        """处理法律文档并向量化
        
        Args:
            source_dir: 文档目录路径
            
        Returns:
            处理结果字典
        """
        print(f"\n📂 开始处理: {source_dir}")
        print("=" * 50)
        
        # 1. 获取已存在的文件
        existing_files = self._get_existing_files()
        print(f"📁 已存在文件: {len(existing_files)} 个")
        
        # 2. 加载文档（跳过已存在的）
        documents, file_count, skipped_files, processed_files = self.loader.load_directory(
            source_dir, 
            skip_files=existing_files
        )
        
        if not documents and skipped_files:
            return {
                "success": True,
                "file_count": 0,
                "document_count": 0,
                "skipped_count": len(skipped_files),
                "skipped_files": skipped_files,
                "copied_files": [],
                "message": f"所有文件均已存在，跳过 {len(skipped_files)} 个重复文件"
            }
        
        if not documents:
            return {
                "success": False,
                "file_count": 0,
                "document_count": 0,
                "skipped_count": 0,
                "skipped_files": [],
                "copied_files": [],
                "message": "未找到可处理的新文档"
            }
        
        # 3. 复制文件到 data_dir
        print("\n📦 正在复制文件到数据目录...")
        copied_files = self._copy_files_to_data_dir(processed_files)
        
        # 4. 向量化存储
        print("\n💾 正在向量化存储...")
        
        if self.vector_manager.vector_store:
            doc_texts = self.vector_manager.add_documents(documents)
        else:
            doc_texts = self.vector_manager.create_from_documents(documents)
        
        # 5. 更新 BM25 索引
        if hasattr(self.hybrid_manager, 'bm25_retriever') and doc_texts:
            print("📝 更新 BM25 索引...")
            self.hybrid_manager.bm25_retriever.update_documents(doc_texts)
        
        print(f"\n✅ 处理完成!")
        
        # 生成消息
        msg_parts = [f"成功处理 {file_count} 个新文件"]
        if skipped_files:
            msg_parts.append(f"跳过 {len(skipped_files)} 个重复文件")
        
        return {
            "success": True,
            "file_count": file_count,
            "document_count": len(documents),
            "skipped_count": len(skipped_files),
            "skipped_files": skipped_files,
            "copied_files": copied_files,
            "message": "，".join(msg_parts) + f"，共 {len(documents)} 个文档块"
        }
