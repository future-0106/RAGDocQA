"""
文档处理模块：支持PDF、TXT、MD格式的文档加载和处理
"""
import time
import pdfplumber
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


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