"""主入口文件：统一调用所有模块，执行RAG查询"""
import warnings
import os
import torch
from config import (
    DEVICE, TEMPERATURE, MAX_TOKENS, SEARCH_K, PDF_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, LOCAL_MODEL_PATH, DASHSCOPE_API_KEY
)
from rag_core import NaiveRAG
from embeddings import Qwen3Embeddings
from llm import DashScopeChatModel
from utils import validate_config

warnings.filterwarnings("ignore")

def main():
    # 打印配置信息
    print("📌 核心配置信息：")
    print(f"PyTorch版本：{torch.__version__}")
    print(f"运行设备：{DEVICE}")

    # 修复GPU型号检测逻辑：增加设备有效性校验
    if torch.cuda.is_available():
        try:
            # 检查设备ID是否有效
            if torch.cuda.device_count() > 0:
                print(f"GPU型号：{torch.cuda.get_device_name(0)}")
            else:
                print("GPU型号：CUDA可用但未检测到有效GPU设备")
        except AssertionError:
            print("GPU型号：检测到无效设备ID，自动切换至CPU模式")
    else:
        print("GPU型号：未检测到可用GPU（已屏蔽或无NVIDIA GPU）")

    print(f"生成参数：temperature={TEMPERATURE}，max_tokens={MAX_TOKENS}")
    print(f"检索参数：SEARCH_K={SEARCH_K}")
    print(f"文本分割：chunk_size={CHUNK_SIZE}，chunk_overlap={CHUNK_OVERLAP}")
    print(f"PDF路径：{PDF_PATH}")
    print(f"嵌入模型路径：{LOCAL_MODEL_PATH}\n")

    try:
        # 验证配置
        validate_config(DASHSCOPE_API_KEY)

        # 初始化嵌入模型
        embedding_model = Qwen3Embeddings(model_path=LOCAL_MODEL_PATH)

        # 初始化LLM模型
        llm_model = DashScopeChatModel(
            api_key=DASHSCOPE_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        # 初始化RAG系统
        rag_system = NaiveRAG(embedding_model, llm_model)

        # 加载PDF并构建向量库
        rag_system.load_pdf_and_build_vector_db(
            pdf_path=PDF_PATH,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        # 测试问题
        print(rag_system.query("新疆和内蒙古的部分地区出现“弃风”的原因", search_k=SEARCH_K))

    except FileNotFoundError as e:
        print(f"\n❌ 文件不存在：{str(e)}")
    except ImportError as e:
        print(f"\n❌ 依赖缺失：{str(e)}")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ GPU显存不足：{str(e)}")
            print("💡 解决：减小CHUNK_SIZE或改用CPU")
        else:
            print(f"\n❌ GPU错误：{str(e)}")
    except Exception as e:
        print(f"\n❌ 运行失败：{str(e)}")
        import traceback
        traceback.print_exc()
        print("💡 请确保阿里百炼API Key有效、GPU环境配置正确、PDF路径正确！")

if __name__ == "__main__":
    main()