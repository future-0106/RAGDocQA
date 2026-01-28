"""主入口文件：统一调用所有模块，执行RAG查询"""
import warnings
import os
import torch
from config import (
    DEVICE, TEMPERATURE, MAX_TOKENS, SEARCH_K, PDF_PATH
)
from rag_core import NaiveRAG

warnings.filterwarnings("ignore")

def main():
    # 打印配置信息
    print("📌 核心配置信息：")
    print(f"PyTorch版本：{torch.__version__}")
    print(f"运行设备：{DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU型号：{torch.cuda.get_device_name(0)}")
    else:
        print("GPU型号：未检测到可用GPU（已屏蔽或无NVIDIA GPU）")
    print(f"生成参数：temperature={TEMPERATURE}，max_tokens={MAX_TOKENS}")
    print(f"检索参数：SEARCH_K={SEARCH_K}")
    print(f"PDF路径：{PDF_PATH}\n")

    try:
        rag_system = NaiveRAG()
        # 测试问题
        # rag_system.query("试用期是否包含在劳动合同期限内？")
        rag_system.query("为什么东北地区多沼泽湿地")


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