"""工具函数文件：通用工具，不依赖业务逻辑"""
import re
import os
import torch
import pdfplumber
import hashlib
import json
from datetime import datetime
from typing import List, Optional
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ========== 核心修复1：提前禁用OneDNN，解决PaddleOCR运行报错 ==========
# 禁用OneDNN加速，避免Filter输入缺失错误
os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# 强制禁用GPU，避免CUDA冲突（和test.py保持一致）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PADDLE_ENABLE_GPU"] = "0"

# ========== 缓存相关配置 ==========
# 缓存目录（项目根目录下）
CACHE_DIR = "pdf_vector_cache"
# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

# 新增OCR相关依赖（按需安装）
try:
    import pytesseract
    from pdf2image import convert_from_path

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR依赖未安装，扫描件/图片PDF将无法处理（可执行：pip install pytesseract pdf2image pillow）")


def fix_cpu_compatibility():
    """修复Qwen3-Embedding兼容性（CPU/GPU通用）"""
    if not hasattr(torch.library, "register_fake"):
        def dummy_register_fake(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        torch.library.register_fake = dummy_register_fake

    if not hasattr(torch._C, "_dispatch_has_kernel_for_dispatch_key"):
        torch._C._dispatch_has_kernel_for_dispatch_key = lambda *args, **kwargs: False
    print("✅ 环境兼容修复完成")


def clean_text(text: str) -> str:
    """过滤PDF乱码，适配中文"""
    if not text:
        return ""
    valid_chars = re.compile(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s，。！？；：""''（）【】《》、·-]')
    text = valid_chars.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_table_to_text(table_data) -> str:
    """
    适配pdfplumber.extract_tables()返回的二维列表
    将pdfplumber提取的表格转为结构化文本
    :param table_data: pdfplumber.page.extract_table()返回的二维列表
    """
    if not table_data:
        return ""

    table_text = "\n【表格开始】\n"
    # 提取表头（第一行作为表头）
    header_row = [str(cell).strip() if cell else "" for cell in table_data[0]]
    table_text += "| " + " | ".join(header_row) + " |\n"
    table_text += "| " + " | ".join(["---"] * len(header_row)) + " |\n"

    # 提取表格内容（从第二行开始）
    for row in table_data[1:]:
        row_cells = [str(cell).strip() if cell else "" for cell in row]
        table_text += "| " + " | ".join(row_cells) + " |\n"
    table_text += "【表格结束】\n"
    return table_text


# ========== 核心修复2：PaddleOCR初始化（复用test.py的成功逻辑） ==========
try:
    from paddleocr import PaddleOCR
    import fitz  # PyMuPDF
    # 和test.py完全一致的初始化参数
    ocr_reader = PaddleOCR(
        use_angle_cls=True,
        lang="ch",
        use_gpu=False,  # 强制CPU，和test.py一致
        show_log=False  # 关闭冗余日志
    )
    PADDLE_OCR_AVAILABLE = True
    print("✅ PaddleOCR初始化成功（和test.py逻辑对齐，禁用OneDNN+CPU模式）")
except ImportError as e:
    PADDLE_OCR_AVAILABLE = False
    print(f"⚠️ PaddleOCR/PyMuPDF未安装：{e}")
    print("💡 安装命令：pip install paddleocr==2.7.3 PyMuPDF==1.20.2")
    # Windows用户额外提示
    if os.name == "nt":
        print("💡 Windows用户需安装：https://aka.ms/vs/17/release/vc_redist.x64.exe")


# ========== 复用test.py的成功逻辑：图片OCR函数 ==========
def ocr_image(image_path: str) -> str:
    """
    复用test.py的成功逻辑：识别单张图片的文字
    :param image_path: 图片路径（PNG/JPG等）
    :return: 清洗后的识别文本
    """
    if not PADDLE_OCR_AVAILABLE:
        raise ImportError("❌ 需安装依赖：pip install paddleocr==2.7.3 PyMuPDF==1.20.2")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ 图片文件不存在：{image_path}")

    # 完全复用test.py的OCR调用逻辑
    result = ocr_reader.ocr(image_path, cls=True)
    ocr_text = ""
    if result and len(result) > 0:
        # 和test.py一致的解析方式
        ocr_text = "\n".join([line[1][0] for line in result[0]])

    # 清洗文本后返回
    return clean_text(ocr_text)


# ========== 替换原有ocr_pdf_page函数（基于test.py的成功逻辑） ==========
def ocr_pdf_page(pdf_path: str, page_idx: int) -> str:
    """
    用PyMuPDF+PaddleOCR识别单页PDF（无需poppler，中文识别更优）
    :param pdf_path: PDF文件路径
    :param page_idx: 页码（从1开始）
    :return: 清洗后的识别文本
    """
    if not PADDLE_OCR_AVAILABLE:
        raise ImportError("❌ 需安装依赖：pip install paddleocr==2.7.3 PyMuPDF==1.20.2")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF文件不存在：{pdf_path}")

    # 1. 用PyMuPDF打开PDF并转高清图片（300DPI提升识别率）
    doc = fitz.open(pdf_path)
    if page_idx < 1 or page_idx > len(doc):
        raise ValueError(f"❌ 页码{page_idx}超出PDF范围（总页数：{len(doc)}）")

    page = doc[page_idx - 1]  # PyMuPDF页码从0开始
    # 生成高清图片（dpi=300，格式=PNG）
    temp_img_path = f"temp_pdf_page_{page_idx}.png"
    pix = page.get_pixmap(dpi=300)
    pix.save(temp_img_path)
    doc.close()

    # 2. 复用test.py的成功逻辑：调用ocr_image识别图片
    try:
        ocr_text = ocr_image(temp_img_path)
    finally:
        # 无论是否成功，都删除临时文件（避免残留）
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

    return ocr_text


def load_pdf(
        pdf_path: str,
        extract_table: bool = True,
        use_ocr_fallback: bool = True
) -> str:
    """
    加载PDF（兼容原有调用）
    - 优先提取原生文本
    - 精准提取表格（转为结构化文本）
    - OCR兜底处理扫描件/图片型PDF
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF文件不存在：{pdf_path}")

    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        print(f"✅ PDF加载成功！共 {page_count} 页")

        for page_idx, page in enumerate(pdf.pages, start=1):
            page_text = ""

            # 1. 提取原生文本
            native_text = page.extract_text() or ""
            if native_text:
                page_text += clean_text(native_text) + "\n\n"

            # 2. 精准提取表格（修复：传参为extract_table()的二维列表）
            if extract_table:
                tables = page.extract_tables()  # 返回二维列表
                if tables:
                    print(f"🔍 第{page_idx}页检测到{len(tables)}个表格，正在提取...")
                    for table in tables:
                        # 跳过空表格
                        if not any(cell for row in table for cell in row):
                            continue
                        table_text = extract_table_to_text(table)  # 修复传参问题
                        page_text += table_text + "\n\n"

            # 3. OCR兜底（无原生文本/表格时触发）
            if use_ocr_fallback and not page_text.strip():
                print(f"⚠️ 第{page_idx}页无原生文本/表格，启用OCR兜底...")
                ocr_text = ocr_pdf_page(pdf_path, page_idx)
                if ocr_text:
                    page_text += ocr_text + "\n\n"

            # 拼接当前页内容（去重空行）
            if page_text.strip():
                full_text += page_text + "\n"

    # 最终校验：无任何文本时抛出异常
    full_text = full_text.strip()
    if not full_text:
        raise ValueError(
            "⚠️ PDF无有效文本！\n"
            "- 若为扫描件：确保已安装OCR依赖并配置语言包\n"
            "- 若为正常PDF：检查是否有可提取的原生文本/表格"
        )
    return full_text


def split_text(
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        pdf_path: str,
        keep_table_complete: bool = True
) -> List[Document]:
    """
    分割文本（兼容原有调用）
    - 补充_type元数据
    - 可选保证表格完整性（避免表格被拆分）
    """
    # 若开启表格完整性保护，调整分隔符（优先按表格边界分割）
    separators = ["\n\n【表格结束】\n\n", "\n\n", "\n", "。", "！", "？", "；", "：", "，", "、"] if keep_table_complete else [
        "\n\n", "\n", "。", "！", "？", "；", "：", "，", "、"
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len
    )
    chunks = splitter.split_text(text)
    chunks = [c.strip() for c in chunks if c.strip()]

    docs = []
    for idx, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": pdf_path,
                "chunk_idx": idx,
                "_type": "Document"
            }
        )
        docs.append(doc)

    print(f"✅ PDF分割成功！共生成 {len(docs)} 个文本片段")
    return docs


def validate_config(api_key: str):
    """验证核心配置（安全获取GPU信息）"""
    if not api_key:
        raise ValueError(
            f"❌ .env文件缺失DASHSCOPE_API_KEY配置\n"
            "💡 .env文件只需添加一行：\n"
            "DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        )
    # 安全打印GPU信息（避免报错）
    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            if device_count > 0:
                print(f"✅ 检测到GPU：{torch.cuda.get_device_name(0)}")
                print(f"✅ GPU显存：{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
            else:
                print("⚠️ CUDA可用但无GPU设备，使用CPU运行")
        except Exception:
            print("⚠️ 无法获取GPU信息，自动使用CPU运行")
    else:
        print("⚠️ 未检测到GPU，自动使用CPU运行")
    print("✅ 阿里百炼API Key配置验证通过！")

# ========== 缓存相关工具函数（仅基于修改时间触发更新） ==========
def get_file_content_hash(pdf_path: str) -> str:
    """
    生成文件内容的MD5哈希（仅和内容相关，和文件名/路径无关）
    大文件分块读取，避免内存溢出
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在：{pdf_path}")

    hash_md5 = hashlib.md5()
    with open(pdf_path, "rb") as f:
        # 分块读取（4KB/块）
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_pdf_unique_id(pdf_path: str) -> tuple[str, str]:
    """
    生成PDF双维度标识：
    返回：(内容哈希ID, 修改时间戳)
    仅修改时间变化时，才重新生成向量
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在：{pdf_path}")

    # 维度1：文件内容哈希（唯一标识内容）
    content_hash = get_file_content_hash(pdf_path)
    # 维度2：文件最后修改时间戳（秒级）
    file_stat = os.stat(pdf_path)
    modify_time = str(file_stat.st_mtime)

    return content_hash, modify_time

def check_cache_valid(pdf_path: str) -> tuple[bool, str]:
    """
    检查缓存是否有效（仅校验修改时间）
    返回：(缓存是否有效, 内容哈希ID)
    """
    content_hash, current_modify_time = get_pdf_unique_id(pdf_path)
    cache_dir = os.path.join(CACHE_DIR, content_hash)
    cache_meta_path = os.path.join(cache_dir, "cache_meta.json")

    # 缓存目录不存在 → 无效
    if not os.path.exists(cache_dir):
        return False, content_hash

    # 缓存元数据不存在 → 无效
    if not os.path.exists(cache_meta_path):
        return False, content_hash

    # 校验：仅对比修改时间（核心逻辑）
    with open(cache_meta_path, "r", encoding="utf-8") as f:
        cache_meta = json.load(f)

    # 只要修改时间一致，就复用缓存（忽略文件名/路径）
    if cache_meta["modify_time"] == current_modify_time:
        return True, content_hash
    else:
        # 修改时间变化 → 缓存失效
        return False, content_hash

def save_cache_meta(pdf_path: str, content_hash: str):
    """保存缓存元数据（记录修改时间、文件名等）"""
    _, modify_time = get_pdf_unique_id(pdf_path)
    file_stat = os.stat(pdf_path)

    cache_meta = {
        "content_hash": content_hash,  # 内容唯一标识
        "pdf_path": pdf_path,          # 记录当前路径（便于查看）
        "file_name": os.path.basename(pdf_path),  # 记录文件名
        "modify_time": modify_time,    # 核心校验字段
        "file_size": file_stat.st_size,
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    cache_dir = os.path.join(CACHE_DIR, content_hash)
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump(cache_meta, f, ensure_ascii=False, indent=2)

def clean_expired_cache(days: int = 7):
    """删除超过指定天数的缓存"""
    now = datetime.now().timestamp()
    for content_hash in os.listdir(CACHE_DIR):
        meta_path = os.path.join(CACHE_DIR, content_hash, "cache_meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        create_time = datetime.strptime(meta["create_time"], "%Y-%m-%d %H:%M:%S").timestamp()
        if (now - create_time) > days * 86400:
            # 删除缓存目录
            import shutil
            shutil.rmtree(os.path.join(CACHE_DIR, content_hash))
            print(f"🗑️ 删除过期缓存：{content_hash[:8]}...")

def list_caches():
    """列出所有缓存信息"""
    print("📋 现有缓存列表：")
    for content_hash in os.listdir(CACHE_DIR):
        meta_path = os.path.join(CACHE_DIR, content_hash, "cache_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(f"- 内容哈希：{content_hash[:8]}... | 文件名：{meta['file_name']} | 修改时间：{meta['modify_time']} | 创建时间：{meta['create_time']}")

# ========== 自测入口：复用test.py的成功逻辑 ==========
if __name__ == "__main__":
    # 测试1：验证OCR图片识别（和test.py一致）
    test_image_path = r"C:\Users\32459\Pictures\Screenshots\屏幕截图 2025-11-15 203315.png"
    if PADDLE_OCR_AVAILABLE and os.path.exists(test_image_path):
        result_text = ocr_image(test_image_path)
        print("✅ 图片OCR测试成功：\n", result_text)
    else:
        print("⚠️ 图片OCR测试失败：请检查PaddleOCR安装或图片路径")

    # 测试2：验证PDF加载+OCR兜底（可选，替换为你的PDF路径）
    test_pdf_path = r"datas/准予就业最低年龄公约.pdf"
    if os.path.exists(test_pdf_path):
        try:
            pdf_text = load_pdf(test_pdf_path)
            print(f"✅ PDF加载测试成功，提取文本长度：{len(pdf_text)}")
        except Exception as e:
            print(f"⚠️ PDF加载测试失败：{e}")