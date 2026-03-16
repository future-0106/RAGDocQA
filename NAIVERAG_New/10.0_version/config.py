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
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# RAG配置
SIMILARITY_TOP_K = 12  # 初始检索更多结果用于重排
SCORE_THRESHOLD = 0.45  # 放宽距离阈值（原0.3，增大以召回更多相关片段）
MAX_CONTEXT_LENGTH = 2000

# 混合检索配置
RETRIEVAL_MODE = "hybrid"  # 可选: "vector", "bm25", "hybrid"
HYBRID_WEIGHTS = (0.2, 0.8)  # (BM25权重, 向量权重) —— 向量权重提高
RERANKER_ENABLED = True  # 是否启用重排
RERANKER_TOP_K = 4  # 重排后返回的数量

# 环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"



# ==================== 查询改写配置====================
# 全局开关
QUERY_REWRITING_ENABLED = False

# 改写专用模型键（若为 None 或空字符串，则复用问答模型）
# 示例：QUERY_REWRITING_MODEL = "dashscope-qwen-turbo"
QUERY_REWRITING_MODEL = "local-Qwen3_0.6B"

# 最大生成查询数（多查询、子问题分解等）
QUERY_REWRITING_MAX_QUERIES = 3

# 是否使用LLM进行改写（若关闭，所有改写策略仅返回原查询）
QUERY_REWRITING_USE_LLM = True

# ---------- 可选：自定义提示模板（留空则使用内置默认模板）----------
QUERY_EXPANSION_PROMPT = ""      # 查询扩展
HYDE_PROMPT = ""                 # 假设文档生成
MULTI_QUERY_PROMPT = ""          # 多查询生成
DECOMPOSE_PROMPT = ""           # 子问题分解




# ===== 统一模型管理配置 =====
ALL_MODELS = {
    # 阿里百炼API模型
    "dashscope-qwen-plus": {
        "type": "api",
        "provider": "dashscope",
        "class": "DashScopeChatModel",
        "params": {
            "model_name": "qwen3.5-plus",
            "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
            "temperature": 0.3,
            "max_tokens": 1500,
            "top_p": 0.9,
        },
        "description": "阿里云千问plus模型，速度快"
    },






    # ModelScope API模型
    "modelscope-glm-5": {
        "type": "api",
        "provider": "modelscope",
        "class": "ModelScopeChatModel",
        "params": {
            "model_name": "ZhipuAI/GLM-5",
            "api_key": os.getenv("MODELSCOPE_API_KEY", ""),
            "temperature": 0.3,
            "max_tokens": 2000,
        },
        "description": "ModelScope GLM-5 推理模型"
    },


    # 本地模型
    "local-Qwen3_0.6B": {
        "type": "local",
        "provider": "local",
        "class": "LocalChatModel",
        "params": {
            "model_path": os.getenv("Qwen3_0.6B_PATH",
                                   r"D:\projects\fastapi_langchain_env\NAIVERAG_New\model\LLM\Qwen3_0.6B"),
            "temperature": 0.3,
            "max_new_tokens": 512,
            "top_p": 0.9,
            "device": DEVICE,
            "repetition_penalty": 1.1,
            "do_sample": True
        },
        "description": "本地千问0.6B模型"
    },

    "local-Qwen2.5-1.5B-Instruct": {
        "type": "local",
        "provider": "local",
        "class": "LocalChatModel",
        "params": {
            "model_path": os.getenv("",
                                    r"D:\projects\fastapi_langchain_env\NAIVERAG_New\model\LLM\Qwen2.5-1.5B-Instruct"),
            "temperature": 0.3,
            "max_new_tokens": 512,
            "top_p": 0.9,
            "device": DEVICE,
            "repetition_penalty": 1.1,
            "do_sample": True
        },
        "description": "本地千问1.5B模型"
    },

    # "local-Qwen3-4B": {
    #     "type": "local",
    #     "provider": "local",
    #     "class": "LocalChatModel",
    #     "params": {
    #         "model_path": os.getenv("",
    #                                 r"D:\projects\fastapi_langchain_env\NAIVERAG_New\model\LLM\Qwen3-4B"),
    #         "temperature": 0.3,
    #         "max_new_tokens": 512,
    #         "top_p": 0.9,
    #         "device": DEVICE,
    #         "repetition_penalty": 1.1,
    #         "do_sample": True
    #     },
    #     "description": "本地千问4B模型"
    # },
}

# 嵌入模型配置
ALL_EMBEDDING_MODELS = {
    "qwen3-embedding-0.6b": {
        "type": "local",
        "class": "LocalEmbeddingModel",
        "params": {
            "model_path": os.getenv("Qwen3_Embedding_0.6B_PATH",
                                   r"D:\projects\fastapi_langchain_env\NAIVERAG_New\model\Embedding\Qwen3_Embedding_0.6B"),
            "device": DEVICE,
            "max_length": 8192,
        },
        "description": "千问3嵌入模型，维度1024"
    },

    # "gte_Qwen2-1.5B-instruct": {
    #     "type": "local",
    #     "class": "LocalEmbeddingModel",
    #     "params": {
    #         "model_path": os.getenv("",
    #                                 r"D:\projects\fastapi_langchain_env\NAIVERAG_New\model\Embedding\gte_Qwen2-1.5B-instruct"),
    #         "device": "cpu",
    #         "max_length": 8192,
    #     },
    #     "description": "千问2嵌入模型，维度1536"
    # },
}

# 重排模型配置
ALL_RERANKER_MODELS = {
    "bge-reranker-base": {
        "type": "local",
        "class": "LocalRerankerModel",
        "params": {
            "model_path": r"D:\projects\fastapi_langchain_env\NAIVERAG_New\model\Reranker\bge-reranker-base",
            "device": DEVICE,
            "max_length": 512,
            "batch_size": 4,
        },
        "description": "BGE重排模型，用于检索结果重排"
    },
}

# 默认使用的模型
DEFAULT_MODEL = "local-Qwen2.5-1.5B-Instruct"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding-0.6b"
DEFAULT_RERANKER_MODEL = "bge-reranker-base"

# 模型工厂配置 - 从环境变量读取api_base
MODEL_FACTORY_CONFIG = {
    "dashscope": {
        "api_base": os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "timeout": 30,
        "enable_search": False
    },
    "modelscope": {
        "api_base": os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1"),
        "timeout": 30
    }
}