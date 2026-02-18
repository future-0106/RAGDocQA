"""
配置文件 - 常量、路径和配置项
"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent

# 数据目录
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
MODELS_DIR = BASE_DIR / "models"

# 模型路径
QWEN_LLM_PATH = r"D:\projects\fastapi_langchain_env\NAIVERAG_test\model\Qwen3_0.6B"
QWEN_EMBEDDING_PATH = r"D:\projects\fastapi_langchain_env\NAIVERAG_test\model\Qwen3_Embedding_0.6B"

# 创建目录
for dir_path in [DATA_DIR, CHROMA_DB_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# 环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 文档处理配置
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# RAG配置
SIMILARITY_TOP_K = 4
SCORE_THRESHOLD = 0.3
MAX_CONTEXT_LENGTH = 1500

# LLM配置
LLM_CONFIG = {
    "max_new_tokens": 200,
    "temperature": 0.3,
    "top_p": 0.9,
    "repetition_penalty": 1.5,
    "do_sample": True
}