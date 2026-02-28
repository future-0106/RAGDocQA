"""
文档处理模块：支持PDF、TXT、MD格式的文档加载和处理
PDF提取采用混合引擎 + 逐级降级策略
文本分割采用递归字符语义分割
"""
import time
from pathlib import Path
from typing import List, Dict, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """文档处理器，支持PDF、TXT、MD格式，PDF采用混合引擎降级提取"""

    # ---------- PDF混合引擎定义（按优先级排序）----------
    ENGINE_PRIORITY = [
        ("PyMuPDF", "_extract_with_pymupdf"),
        ("pdfplumber", "_extract_with_pdfplumber"),
        ("pypdf", "_extract_with_pypdf"),
        ("pdfminer", "_extract_with_pdfminer"),
    ]

    def __init__(self, chunk_size=300, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # ========== 增强递归字符语义分割 ==========
        # 分隔符按语义粒度从大到小排列，优先保持段落/句子完整性
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",        # 多空行分隔的大段落
                "\n\n",          # 段落
                "\n",            # 行
                "。",            # 句子（中文句号）
                "！", "？",      # 句子（中文感叹/疑问）
                "；",            # 分句（中文分号）
                "：",            # 分句（中文冒号）
                "，",            # 短语（中文逗号）
                " ",            # 单词间空格
                ""              # 字符级回退
            ],
            keep_separator=False,   # 不保留分隔符，减少冗余
        )

    # ---------- 各引擎提取函数（惰性导入）----------
    def _extract_with_pymupdf(self, file_path: str) -> Dict[int, str]:
        """使用 PyMuPDF (fitz) 按页提取文本"""
        try:
            import fitz
        except ImportError:
            return {}
        try:
            pages = {}
            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    text = doc[page_num].get_text()
                    if text and text.strip():
                        pages[page_num + 1] = ' '.join(text.split())
            return pages
        except Exception:
            return {}

    def _extract_with_pdfplumber(self, file_path: str) -> Dict[int, str]:
        """使用 pdfplumber 按页提取文本"""
        try:
            import pdfplumber
        except ImportError:
            return {}
        try:
            pages = {}
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages[page_num] = ' '.join(text.split())
            return pages
        except Exception:
            return {}

    def _extract_with_pypdf(self, file_path: str) -> Dict[int, str]:
        """使用 pypdf 按页提取文本"""
        try:
            from pypdf import PdfReader
        except ImportError:
            return {}
        try:
            pages = {}
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    pages[page_num] = ' '.join(text.split())
            return pages
        except Exception:
            return {}

    def _extract_with_pdfminer(self, file_path: str) -> Dict[int, str]:
        """
        使用 pdfminer.six 提取文本（尽力按页，否则全文档作为第1页）
        """
        try:
            from pdfminer.high_level import extract_text_to_fp
            from pdfminer.layout import LAParams
            from io import StringIO
        except ImportError:
            return {}
        try:
            output = StringIO()
            with open(file_path, 'rb') as f:
                extract_text_to_fp(f, output, laparams=LAParams(), output_type='text', codec='utf-8')
            full_text = output.getvalue()
            if full_text and full_text.strip():
                return {1: ' '.join(full_text.split())}
            return {}
        except Exception:
            return {}

    # ---------- PDF 混合引擎降级提取（核心逻辑）----------
    def load_pdf_document(self, file_path: str) -> List[Document]:
        """
        加载PDF文档：混合引擎逐级降级，每个页面独立尝试所有引擎
        """
        documents = []
        print(f"📖 正在读取PDF文件: {Path(file_path).name} (混合引擎降级提取)")

        # 尝试获取总页数（使用第一个可工作的引擎）
        total_pages = 0
        for name, method_name in self.ENGINE_PRIORITY:
            method = getattr(self, method_name)
            pages = method(file_path)
            if pages:
                total_pages = max(pages.keys())
                break
        if total_pages == 0:
            print("⚠️  无法获取PDF页数，文件可能损坏")
            return []

        print(f"   → PDF总页数: {total_pages}")

        # 每个页面独立降级提取
        for page_num in range(1, total_pages + 1):
            page_text = None
            used_engine = None
            for name, method_name in self.ENGINE_PRIORITY:
                method = getattr(self, method_name)
                pages = method(file_path)
                if page_num in pages and pages[page_num].strip():
                    page_text = pages[page_num]
                    used_engine = name
                    break

            if page_text:
                doc = Document(
                    page_content=page_text,
                    metadata={
                        "source": file_path,
                        "type": "pdf",
                        "page": page_num,
                        "total_pages": total_pages,
                        "extract_engine": used_engine
                    }
                )
                documents.append(doc)
                print(f"   → 第 {page_num}/{total_pages} 页 (引擎: {used_engine})")
            else:
                print(f"   ⚠️ 第 {page_num}/{total_pages} 页 所有引擎均提取失败，跳过")

        if not documents:
            print(f"⚠️  PDF文件 {Path(file_path).name} 未提取到任何文本")
        return documents

    # ---------- 其他格式（TXT/MD）保持不变，略作优化----------
    def load_txt_document(self, file_path: str) -> List[Document]:
        """加载TXT文档"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            text = ' '.join(text.split())
            doc = Document(
                page_content=text,
                metadata={"source": file_path, "type": "text"}
            )
            return [doc]
        except Exception as e:
            print(f"❌ 读取TXT文件失败 {file_path}: {e}")
            return []

    def load_md_document(self, file_path: str) -> List[Document]:
        """加载MD文档"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            text = ' '.join(text.split())
            doc = Document(
                page_content=text,
                metadata={"source": file_path, "type": "markdown"}
            )
            return [doc]
        except Exception as e:
            print(f"❌ 读取MD文件失败 {file_path}: {e}")
            return []

    def load_documents(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档（入口）"""
        file_ext = Path(file_path).suffix.lower()
        if file_ext == '.pdf':
            return self.load_pdf_document(file_path)
        elif file_ext == '.txt':
            return self.load_txt_document(file_path)
        elif file_ext in ['.md', '.markdown']:
            return self.load_md_document(file_path)
        else:
            print(f"⚠️  不支持的文件格式: {file_ext}，尝试按文本读取")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                text = ' '.join(text.split())
                doc = Document(
                    page_content=text,
                    metadata={"source": file_path, "type": "unknown"}
                )
                return [doc]
            except Exception as e:
                print(f"❌ 无法读取文件: {file_path}, 错误: {e}")
                return []

    def process_file(self, file_path: str) -> List[Document]:
        """处理单个文件并返回文档块（已分割）"""
        all_chunks = []
        if not Path(file_path).exists():
            print(f"❌ 文件不存在: {file_path}")
            return all_chunks

        try:
            print(f"📄 处理文件: {Path(file_path).name}")
            docs = self.load_documents(file_path)
            if docs:
                chunks = self.text_splitter.split_documents(docs)
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "file_name": Path(file_path).name,
                        "file_type": Path(file_path).suffix.lower()[1:],
                        "processed_time": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                all_chunks.extend(chunks)
                print(f"✅ 文件处理完成，分成 {len(chunks)} 个语义块")
            else:
                print(f"❌ 文件为空或无法读取")
        except Exception as e:
            print(f"❌ 处理失败 {Path(file_path).name}: {e}")

        return all_chunks