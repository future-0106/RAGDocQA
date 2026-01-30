"""全局配置文件：存放所有可配置参数，统一管理"""
import os
import torch
import dotenv

# 加载.env文件
dotenv.load_dotenv(override=True)

# 设备配置（GPU优先）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 嵌入模型配置
LOCAL_MODEL_PATH = (r"D:\projects\fastapi_langchain_env\NAIVERAG\Qwen3_Embedding_0.6B")
MAX_EMBED_LENGTH = 8192

# PDF配置
PDF_PATH = r"D:\projects\fastapi_langchain_env\NAIVERAG\datas\劳动合同法问题解答.pdf"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 检索配置
SEARCH_K = 8
PERSIST_DIR = r"./chroma_db"

# 阿里百炼LLM配置
MODEL_NAME = "qwen-turbo"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
TEMPERATURE = 0.3
MAX_TOKENS = 1500