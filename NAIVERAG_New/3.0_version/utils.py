"""
工具函数模块
"""
import warnings
from transformers import logging


def setup_environment():
    """设置环境"""
    # 关闭transformers的特定警告
    logging.set_verbosity_error()
    warnings.filterwarnings("ignore",
                            message=".*generation_config.*default values have been modified.*")


def check_imports():
    """检查必要的导入"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ 使用 langchain_huggingface.HuggingFaceEmbeddings")
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print("⚠️  使用 langchain_community.HuggingFaceEmbeddings")
        except ImportError:
            print("❌ 无法导入 HuggingFaceEmbeddings")
            print("请安装: pip install langchain-huggingface 或 langchain-community")
            return False

    try:
        from langchain_chroma import Chroma
        print("✅ 使用 langchain_chroma.Chroma")
        return True
    except ImportError as e:
        print(f"❌ 无法导入 ChromaDB: {e}")
        print("请安装: pip install chromadb langchain-chroma")
        return False


def display_menu():
    """显示菜单"""
    print("\n" + "=" * 50)
    print("🚀 Qwen3-0.6B RAG 系统")
    print("📚 支持PDF、TXT、MD文档格式，上传后立即转换为向量")
    print("=" * 50)
    print("\n请选择操作:")
    print("  [1] 基于文档内容回答问题")
    print("  [2] 上传单个文件并向量化")
    print("  [3] 批量上传多个文件")
    print("  [4] 查看已上传文件列表")
    print("  [5] 查看系统状态和统计")
    print("  [6] 重新处理所有文件并重建向量存储")
    print("  [7] 清除屏幕")
    print("  [8] 退出系统")
    print("-" * 50)