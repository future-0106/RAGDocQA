"""测试4：PDF加载与分割（适配langchain 0.2.x，解决中文乱码）"""
import os
import re
from typing import List

# -------------------------- 强制验证依赖（确保安装成功） --------------------------
try:
    import pdfplumber
    # 适配langchain 0.2.x的正确导入路径
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError as e:
    print(f"❌ 依赖安装失败：{e}")
    print("💡 执行以下命令重装：")
    print("   pip uninstall -y langchain && pip install langchain==0.2.10 pdfplumber==0.11.4 --no-cache-dir")
    exit(1)

# -------------------------- 核心配置（必须修改为你的PDF路径） --------------------------
PDF_FILE_PATH = r"D:\projects\fastapi_langchain_env\NAIVERAG\datas\1958年消除就业和职业歧视公约.pdf"  # 替换为实际PDF路径
CHUNK_SIZE = 500  # 每个片段字符数
CHUNK_OVERLAP = 50  # 片段重叠字符数

# -------------------------- 文本清理（解决乱码核心函数） --------------------------
def clean_text(text: str) -> str:
    """过滤乱码，仅保留中文、英文、数字、常用标点"""
    if not text:
        return ""
    # 正则保留有效字符，过滤所有乱码/不可见字符
    valid_chars = re.compile(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s，。！？；：""''（）【】《》、·]')
    text = valid_chars.sub('', text)
    # 去除多余空格/换行
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------- PDF加载（中文友好） --------------------------
def load_pdf(pdf_path: str) -> str:
    """用pdfplumber加载PDF，避免中文乱码"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在：{pdf_path}")

    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        print(f"✅ PDF加载成功！共 {page_count} 页")

        for page in pdf.pages:
            page_text = page.extract_text() or ""
            clean_page_text = clean_text(page_text)
            if clean_page_text:
                full_text += clean_page_text + "\n"

    if not full_text:
        raise ValueError("⚠️ PDF无有效文本！可能是扫描件（图片型PDF），需OCR处理")
    return full_text

# -------------------------- 文本分割（langchain版） --------------------------
def split_text(text: str) -> List[str]:
    """用langchain分割文本，适配中文标点"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 优先按中文标点分割，避免截断句子
        separators=["\n\n", "\n", "。", "！", "？", "；", "：", "，", "、"],
        length_function=len  # 按字符数计算（中文适配）
    )
    chunks = splitter.split_text(text)
    chunks = [c.strip() for c in chunks if c.strip()]
    print(f"✅ PDF分割成功！共生成 {len(chunks)} 个文本片段")
    return chunks

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    try:
        # 1. 加载PDF（解决乱码）
        pdf_text = load_pdf(PDF_FILE_PATH)

        # 2. 分割文本
        text_chunks = split_text(pdf_text)

        # 3. 验证结果（显示第一个片段）
        print("\n📌 第一个文本片段（无乱码）：")
        print(text_chunks[0][:200] + "..." if len(text_chunks[0])>200 else text_chunks[0])

    except FileNotFoundError as e:
        print(f"❌ 错误：{e}")
        print("💡 检查PDF路径是否正确，路径中不要有中文/空格（或加r前缀）")
    except ValueError as e:
        print(f"❌ 错误：{e}")
    except Exception as e:
        print(f"❌ 处理失败：{str(e)}")