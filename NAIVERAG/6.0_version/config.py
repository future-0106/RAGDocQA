"""全局配置文件：存放所有可配置参数，统一管理"""
import os
import torch
import dotenv

# 加载.env文件
dotenv.load_dotenv(override=True)

# 设备配置（GPU优先）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 嵌入模型配置
LOCAL_MODEL_PATH = (r"D:\projects\fastapi_langchain_env\NAIVERAG\model\Qwen3_Embedding_0.6B")
MAX_EMBED_LENGTH = 8192

# PDF配置
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# 检索配置
SEARCH_K = 30
GLOBAL_VECTOR_DB_PATH = r"./global_chroma_db"
UPLOAD_RECORD_FILE = r"./pdf_uploads.json"

# ===== 统一模型管理配置 =====
ALL_MODELS = {
    # 阿里百炼API模型
    "dashscope-qwen-turbo": {
        "type": "api",
        "class": "DashScopeChatModel",
        "params": {
            "model_name": "qwen-turbo",
            "api_key": os.getenv("DASHSCOPE_API_KEY"),
            "temperature": 0.3,
            "max_tokens": 1500
        }
    },
    "dashscope-qwen-plus": {
        "type": "api",
        "class": "DashScopeChatModel",
        "params": {
            "model_name": "qwen-max",
            "api_key": os.getenv("DASHSCOPE_API_KEY"),
            "temperature": 0.3,
            "max_tokens": 2000
        }
    },
    # 可添加其他API模型...

    # 本地模型（示例）
    "local-qwen-0.6b": {
        "type": "local",
        "class": "LocalChatModel",
        "params": {
            "model_path": r"D:\projects\fastapi_langchain_env\NAIVERAG\model\Qwen3_0.6B",
            "temperature": 0,
            "max_tokens": 1500,
            "device": DEVICE
        }
    }
}

# 默认使用的模型
DEFAULT_MODEL = "local-qwen-0.6b"