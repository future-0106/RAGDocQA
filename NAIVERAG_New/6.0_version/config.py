"""
全局配置文件：存放所有可配置参数，统一管理
支持本地多模型和阿里云百炼API
"""
import os
import torch
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent

# 直接指定 .env 文件路径
def load_env_file():
    """加载指定位置的 .env 文件"""
    # 直接指定你的 .env 文件路径
    env_path = Path(r"D:\projects\fastapi_langchain_env\NAIVERAG_New\.env")

    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=True)
            return True
        except ImportError:
            return False
    else:
        return False

# 加载环境变量
USE_DOTENV = load_env_file()

# 数据目录
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
MODELS_DIR = BASE_DIR / "models"

# 创建目录
for dir_path in [DATA_DIR, CHROMA_DB_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# 设备配置（GPU优先）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 使用设备: {DEVICE}")

# 文档处理配置
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# RAG配置
SIMILARITY_TOP_K = 10  # 初始检索更多结果用于重排
SCORE_THRESHOLD = 0.3
MAX_CONTEXT_LENGTH = 1500

# 混合检索配置
RETRIEVAL_MODE = "hybrid"  # 可选: "vector", "bm25", "hybrid"
HYBRID_WEIGHTS = (0.4, 0.6)  # (BM25权重, 向量权重)
RERANKER_ENABLED = True  # 是否启用重排
RERANKER_TOP_K = 4  # 重排后返回的数量

# 环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ===== 统一模型管理配置 =====
ALL_MODELS = {
    # 阿里百炼API模型
    "dashscope-qwen-turbo": {
        "type": "api",
        "provider": "dashscope",
        "class": "DashScopeChatModel",
        "params": {
            "model_name": "qwen-turbo",
            "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
            "temperature": 0.3,
            "max_tokens": 1500,
            "top_p": 0.9,
        },
        "description": "阿里云千问Turbo模型，速度快"
    },
    # 本地模型
    "local-qwen-0.6b": {
        "type": "local",
        "provider": "local",
        "class": "LocalChatModel",
        "params": {
            "model_path": os.getenv("Qwen3_0.6B_PATH",
                                   r"D:\projects\fastapi_langchain_env\NAIVERAG_test\model\Qwen3_0.6B"),
            "temperature": 0.3,
            "max_new_tokens": 200,
            "top_p": 0.9,
            "device": DEVICE,
            "repetition_penalty": 1.5,
            "do_sample": True
        },
        "description": "本地千问0.6B模型"
    },
}

# 嵌入模型配置
ALL_EMBEDDING_MODELS = {
    "qwen3-embedding-0.6b": {
        "type": "local",
        "class": "LocalEmbeddingModel",
        "params": {
            "model_path": os.getenv("Qwen3_Embedding_0.6B_PATH",
                                   r"D:\projects\fastapi_langchain_env\NAIVERAG_test\model\Qwen3_Embedding_0.6B"),
            "device": DEVICE,
            "max_length": 8192,
        },
        "description": "千问3嵌入模型，维度1024"
    },
}

# 重排模型配置
ALL_RERANKER_MODELS = {
    "bge-reranker-base": {
        "type": "local",
        "class": "LocalRerankerModel",
        "params": {
            "model_path": r"D:\projects\fastapi_langchain_env\NAIVERAG_New\model\bge-reranker-base",
            "device": DEVICE,
            "max_length": 512,
            "batch_size": 4,
        },
        "description": "BGE重排模型，用于检索结果重排"
    },
}

# 默认使用的模型
DEFAULT_MODEL = "local-qwen-0.6b"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding-0.6b"
DEFAULT_RERANKER_MODEL = "bge-reranker-base"

# 模型工厂配置 - 从环境变量读取api_base
MODEL_FACTORY_CONFIG = {
    "dashscope": {
        "api_base": os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "timeout": 30,
        "enable_search": False
    }
}