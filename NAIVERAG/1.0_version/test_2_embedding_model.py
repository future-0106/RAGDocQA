"""测试2：纯transformers加载本地Qwen3-Embedding-0.6B（最终稳定版）"""
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings("ignore")

# -------------------------- 核心配置（必须修改为你的本地模型绝对路径！） --------------------------
# 示例：LOCAL_MODEL_PATH = "D:/projects/fastapi_langchain_env/NAIVERAG/Qwen3-Embedding-0.6B"
# 要求：路径是本地模型文件夹的绝对路径，无中文、无空格、无特殊字符
LOCAL_MODEL_PATH = r"Qwen3_Embedding_0.6B"
DEVICE = torch.device("cpu")
MAX_LENGTH = 8192


# -------------------------- Qwen3官方核心函数（一字未改） --------------------------
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


# -------------------------- 终极CPU兼容修复 --------------------------
def fix_cpu_compatibility():
    """修复所有CPU环境下的API缺失问题"""
    # 修复torch.library.register_fake
    if not hasattr(torch.library, "register_fake"):
        def dummy_register_fake(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        torch.library.register_fake = dummy_register_fake

    # 修复torch._C相关算子检查
    if not hasattr(torch._C, "_dispatch_has_kernel_for_dispatch_key"):
        torch._C._dispatch_has_kernel_for_dispatch_key = lambda *args, **kwargs: False

    print("✅ CPU环境兼容修复完成")


# -------------------------- 主测试逻辑（纯本地加载，无远程查询） --------------------------
def test_local_qwen3_embedding():
    try:
        # 验证模型路径是否存在
        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(f"本地模型路径不存在：{LOCAL_MODEL_PATH}")

        # 步骤1：CPU兼容修复
        fix_cpu_compatibility()

        # 步骤2：构造测试数据
        print("📌 构造测试数据（带官方指令）")
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        queries = [
            get_detailed_instruct(task, 'What is the capital of China?'),
            get_detailed_instruct(task, 'Explain gravity')
        ]
        documents = [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
        ]
        input_texts = queries + documents

        # 步骤3：加载本地Tokenizer（强制本地加载，无远程）
        print(f"📌 加载本地Tokenizer：{LOCAL_MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_PATH,
            padding_side='left',
            trust_remote_code=True,
            local_files_only=True  # 关键：仅加载本地文件，不走远程
        )

        # 步骤4：加载本地Model（强制本地加载）
        print(f"📌 加载本地Model：{LOCAL_MODEL_PATH}")
        model = AutoModel.from_pretrained(
            LOCAL_MODEL_PATH,
            trust_remote_code=True,
            device_map="cpu",
            local_files_only=True,  # 关键：禁用远程查询
            torch_dtype=torch.float32  # CPU环境用float32，避免精度问题
        )
        model = model.to(DEVICE)
        model.eval()  # 推理模式

        # 步骤5：分词处理
        print("🔍 开始分词...")
        batch_dict = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        batch_dict = {k: v.to(DEVICE) for k, v in batch_dict.items()}

        # 步骤6：模型推理（无梯度，提升速度）
        with torch.no_grad():
            outputs = model(**batch_dict)

        # 步骤7：官方池化+归一化
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 步骤8：计算相似度（对齐官方结果）
        scores = (embeddings[:2] @ embeddings[2:].T)
        print("\n🎉 本地Qwen3-Embedding测试成功！")
        print("📌 官方标准相似度结果：")
        print(scores.tolist())
        # 预期输出：[[0.7645..., 0.1414...], [0.1354..., 0.5999...]]
        print(f"\n📌 嵌入向量维度验证：{embeddings.shape} (预期：torch.Size([4, 1536]))")

    except FileNotFoundError as e:
        print(f"❌ 错误：{e}")
        print("\n💡 解决方法：")
        print(f"   1. 确认模型路径正确：{LOCAL_MODEL_PATH}")
        print("   2. 模型文件夹内必须包含以下文件：")
        print("      - config.json（模型配置）")
        print("      - pytorch_model.bin（模型权重）")
        print("      - tokenizer.json/tokenizer_config.json（分词器）")
        print("      - modeling_qwen3.py（Qwen3自定义代码）")
    except Exception as e:
        print(f"❌ 测试失败：{str(e)}")
        print("\n💡 最终解决方法：")
        print("   1. 执行：pip install --force-reinstall transformers==4.51.0 numpy==1.26.4")
        print("   2. 模型路径用绝对路径，且无中文/空格（如：D:/models/Qwen3-Embedding-0.6B）")
        print("   3. 确保PyTorch版本是2.2.2+cpu：python -c 'import torch; print(torch.__version__)'")


if __name__ == "__main__":
    # 打印核心环境信息
    print("📌 环境验证（最终版）：")
    print(f"PyTorch版本：{torch.__version__} (要求：2.2.2+cpu)")
    import transformers

    print(f"Transformers版本：{transformers.__version__} (要求：4.51.0)")
    import numpy

    print(f"NumPy版本：{numpy.__version__} (要求：1.26.4)\n")

    # 运行测试
    test_local_qwen3_embedding()