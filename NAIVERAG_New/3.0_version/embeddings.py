"""
嵌入模型模块
"""
from typing import List

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from config import QWEN_EMBEDDING_PATH


class QwenEmbeddings:
    """适配Qwen3-Embedding-0.6B的嵌入模型"""

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = QWEN_EMBEDDING_PATH

        print(f"🔧 加载嵌入模型: {model_path}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'trust_remote_code': True,
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 8,
            }
        )

        # 测试模型
        try:
            test_embedding = self.embeddings.embed_query("测试文本")
            print(f"✅ 嵌入模型加载成功，向量维度: {len(test_embedding)}")
        except Exception as e:
            print(f"⚠️  嵌入模型测试时出错: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)