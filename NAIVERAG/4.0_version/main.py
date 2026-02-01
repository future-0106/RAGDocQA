"""主入口文件：交互模式管理PDF、全局检索回答"""
import warnings
import os
import torch
from config import (
    DEVICE, TEMPERATURE, MAX_TOKENS, SEARCH_K,
    CHUNK_SIZE, CHUNK_OVERLAP, LOCAL_MODEL_PATH, DASHSCOPE_API_KEY,
    UPLOAD_RECORD_FILE
)
from rag_core import NaiveRAG
from embeddings import Qwen3Embeddings
from llm import DashScopeChatModel
from utils import validate_config, list_uploaded_pdfs_detail

warnings.filterwarnings("ignore")

def main():
    # 打印配置信息
    print("📌 核心配置信息：")
    print(f"PyTorch版本：{torch.__version__}")
    print(f"运行设备：{DEVICE}")

    # GPU信息检测
    if torch.cuda.is_available():
        try:
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

        # 初始化RAG系统（全局向量库）
        rag_system = NaiveRAG(embedding_model, llm_model)

        # 交互菜单
        print("="*60)
        print("🎯 多PDF智能检索系统")
        print("="*60)
        while True:
            print("\n请选择操作：")
            print("1. 上传PDF文件（支持单个/多个路径）")
            print("2. 查看已上传PDF列表")
            print("3. 提问（检索所有PDF回答）")
            print("4. 清空所有PDF数据（谨慎）")
            print("0. 退出系统")
            choice = input("\n输入操作序号：").strip()

            if choice == "1":
                # 上传PDF
                pdf_paths_input = input("输入PDF文件路径（多个路径用英文逗号分隔）：").strip()
                pdf_paths = [p.strip() for p in pdf_paths_input.split(",") if p.strip()]
                if not pdf_paths:
                    print("❌ 未输入有效路径")
                    continue
                rag_system.batch_add_pdfs(pdf_paths)

            elif choice == "2":
                # 查看已上传列表
                list_uploaded_pdfs_detail()

            elif choice == "3":
                # 提问检索
                question = input("\n请输入你的问题：").strip()
                if not question:
                    print("❌ 问题不能为空")
                    continue
                answer = rag_system.query_all(question, search_k=SEARCH_K)
                print("\n💡 回答：")
                print("-" * 50)
                print(answer)
                print("-" * 50)

            elif choice == "4":
                # 清空数据
                rag_system.clear_all()

            elif choice == "0":
                # 退出
                print("\n👋 退出系统，感谢使用！")
                break

            else:
                print("❌ 无效序号，请重新输入")

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