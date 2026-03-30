"""法律文档加载器"""
from pathlib import Path
from typing import List, Tuple, Set
from langchain_core.documents import Document


class LegalDocumentLoader:
    """法律文档加载器 - 加载 PDF/DOCX/XLSX/TXT/MD 格式"""
    
    SUPPORTED_FORMATS = {".pdf", ".docx", ".xlsx", ".txt", ".md"}
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        from core.documents import DocumentProcessor
        self.processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_file(self, file_path: str) -> List[Document]:
        """加载单个文件"""
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix not in self.SUPPORTED_FORMATS:
            print(f"⚠️ 不支持的格式: {suffix}")
            return []
        
        try:
            documents = self.processor.process_file(file_path)
            print(f"✅ 已加载: {path.name} ({len(documents)} 个文档块)")
            return documents
        except Exception as e:
            print(f"❌ 加载失败: {path.name} - {e}")
            return []
    
    def load_directory(self, dir_path: str, skip_files: Set[str] = None) -> Tuple[List[Document], int, List[str], List[str]]:
        """批量加载目录下的所有法律文档
        
        Args:
            dir_path: 目录路径
            skip_files: 已存在的文件名集合，用于跳过重复文件
            
        Returns:
            (documents, file_count, skipped_files, processed_files): 文档列表、文件数量、跳过的文件名、处理成功的文件路径
        """
        documents = []
        skipped_files = []
        processed_files = []
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            print(f"❌ 目录不存在: {dir_path}")
            return documents, 0, skipped_files, processed_files
        
        if skip_files is None:
            skip_files = set()
        
        file_count = 0
        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                # 检查是否已存在
                if file_path.name in skip_files:
                    print(f"⏭️  跳过（已存在）: {file_path.name}")
                    skipped_files.append(file_path.name)
                    continue
                
                file_count += 1
                docs = self.load_file(str(file_path))
                if docs:
                    documents.extend(docs)
                    processed_files.append(str(file_path))
        
        print(f"\n📊 总计: {file_count} 个新文件, {len(documents)} 个文档块, 跳过 {len(skipped_files)} 个重复文件")
        return documents, file_count, skipped_files, processed_files
