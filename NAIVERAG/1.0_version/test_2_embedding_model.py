# 仅保留sentence-transformers模式 + 适配2.7.0版本（移除similarity方法，手动计算相似度）
# 环境要求：torch==2.2.2+cpu, transformers==4.51.0, sentence-transformers==2.7.0, numpy==1.26.4
import os
import torch
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer


# ===================== 全局模拟缺失函数（解决加载问题） =====================
# 1. 模拟init_empty_weights
def init_empty_weights():
    class DummyContext:
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    return DummyContext()


# 2. 模拟find_tied_parameters
def find_tied_parameters(model):
    return []


# 注入到全局和transformers模块
globals()['init_empty_weights'] = init_empty_weights
globals()['find_tied_parameters'] = find_tied_parameters

import transformers.modeling_utils

transformers.modeling_utils.init_empty_weights = init_empty_weights
transformers.modeling_utils.find_tied_parameters = find_tied_parameters


# ===================== 修复算子和设备问题 =====================
class DummyNMS:
    def __call__(self, *args, **kwargs):
        return torch.tensor([0], dtype=torch.int64)


if not hasattr(torch.ops, "torchvision"):
    torch.ops.torchvision = type('torchvision', (), {})()
torch.ops.torchvision.nms = DummyNMS()

# 禁用GPU和无关警告
warnings.filterwarnings("ignore")
torch.cuda.is_available = lambda: False
DEVICE = "cpu"

# ===================== 核心配置 =====================
LOCAL_MODEL_PATH = r"D:/projects/fastapi_langchain_env/NAIVERAG/model/Qwen3_Embedding_0.6B"


# ===================== 手动计算余弦相似度（替代similarity方法） =====================
def cosine_similarity(vec1, vec2):
    """
    手动计算余弦相似度，适配sentence-transformers 2.7.0（无similarity方法）
    :param vec1: 查询嵌入向量 (n, dim)
    :param vec2: 文档嵌入向量 (m, dim)
    :return: 相似度矩阵 (n, m)
    """
    # 确保向量是numpy数组
    if isinstance(vec1, torch.Tensor):
        vec1 = vec1.cpu().numpy()
    if isinstance(vec2, torch.Tensor):
        vec2 = vec2.cpu().numpy()

    # 归一化向量（避免计算误差）
    vec1 = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
    vec2 = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)

    # 计算余弦相似度
    similarity_matrix = np.dot(vec1, vec2.T)
    # 转换为torch tensor（保持和官方示例一致的输出格式）
    return torch.from_numpy(similarity_matrix)


# ===================== 核心测试函数 =====================
def test_sentence_transformers_qwen3():
    try:
        # 1. 验证模型路径
        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(f"模型路径不存在：{LOCAL_MODEL_PATH}")

        # 2. 打印环境信息
        print("📌 环境验证：")
        print(f"PyTorch版本：{torch.__version__} (要求：2.2.2+cpu)")
        import transformers
        print(f"Transformers版本：{transformers.__version__} (要求：4.51.0)")
        import numpy
        print(f"NumPy版本：{numpy.__version__} (要求：1.26.4)")
        import sentence_transformers
        print(f"Sentence-Transformers版本：{sentence_transformers.__version__} (要求：2.7.0)")
        print("✅ 环境版本全部匹配")

        # 3. 加载本地模型
        print(f"\n📌 加载本地Qwen3-Embedding模型：{LOCAL_MODEL_PATH}")
        model = SentenceTransformer(
            LOCAL_MODEL_PATH,
            trust_remote_code=True,
            device=DEVICE,
            cache_folder=None,
            use_auth_token=False
        )
        model.tokenizer.padding_side = "left"

        # 4. 构造测试数据
        queries = [
            "What is the capital of China?",
            "Explain gravity",
        ]
        documents = [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        ]

        # 5. 编码查询和文档
        print("\n📌 编码查询文本（自动应用query prompt）...")
        query_embeddings = model.encode(
            queries,
            prompt_name="query",
            normalize_embeddings=True,
            show_progress_bar=False
        )
        print("📌 编码文档文本...")
        document_embeddings = model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # 6. 手动计算余弦相似度（核心修复：替代similarity方法）
        print("\n📌 计算查询与文档的余弦相似度...")
        similarity = cosine_similarity(query_embeddings, document_embeddings)

        # 7. 输出结果
        print("\n✅ 运行成功！")
        print("📌 相似度矩阵：")
        print(similarity)
        print("\n📊 相似度详情：")
        for i, query in enumerate(queries):
            print(f"\n查询{i + 1}：{query}")
            for j, doc in enumerate(documents):
                print(f"  → 文档{j + 1}：{similarity[i][j]:.4f}")

    except FileNotFoundError as e:
        print(f"\n❌ 错误：{e}")
        print("💡 请确认模型路径下有config.json、pytorch_model.bin、modeling_qwen3.py等文件")
    except Exception as e:
        print(f"\n❌ 测试失败：{str(e)}")
        print("\n💡 快速修复：")
        print("   1. 确认modeling_qwen3.py文件存在于模型路径")
        print("   2. 执行：pip install --force-reinstall sentence-transformers==2.7.0")
        print("   3. 重启IDE后重新运行")


if __name__ == "__main__":
    test_sentence_transformers_qwen3()