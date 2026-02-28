"""
混合检索和重排模块：实现BM25+Embedding混合检索和重排功能
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from rank_bm25 import BM25Okapi
import jieba
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import DEVICE, HYBRID_WEIGHTS, RERANKER_ENABLED, RERANKER_TOP_K, ALL_RERANKER_MODELS


class BM25Retriever:
    """BM25检索器"""

    def __init__(self, documents: List[str] = None, use_jieba: bool = True):
        """
        初始化BM25检索器

        Args:
            documents: 文档列表
            use_jieba: 是否使用jieba进行中文分词
        """
        self.use_jieba = use_jieba
        self.documents = documents or []
        self.tokenized_docs = []
        self.bm25 = None

        if documents:
            self.add_documents(documents)

    def tokenize(self, text: str) -> List[str]:
        """对文本进行分词"""
        if self.use_jieba:
            return [word for word in jieba.lcut(text) if word.strip()]
        else:
            # 简单的空格分词
            return [word for word in text.split() if word.strip()]

    def add_documents(self, documents: List[str]):
        """添加文档到BM25索引"""
        self.documents.extend(documents)

        # 对文档进行分词
        for doc in documents:
            self.tokenized_docs.append(self.tokenize(doc))

        # 重建BM25索引
        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)

    def update_documents(self, documents: List[str]):
        """更新所有文档"""
        self.documents = documents.copy()
        self.tokenized_docs = []

        for doc in documents:
            self.tokenized_docs.append(self.tokenize(doc))

        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        BM25检索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            列表，元素为(文档索引, 分数)
        """
        if not self.bm25:
            return []

        # 对查询进行分词
        tokenized_query = self.tokenize(query)

        # 获取BM25分数
        scores = self.bm25.get_scores(tokenized_query)

        # 获取top-k结果
        if len(scores) == 0:
            return []

        # 标准化分数到0-1范围
        if np.max(scores) > 0:
            scores = scores / np.max(scores)

        # 获取top-k索引
        top_k_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_indices:
            if scores[idx] > 0:  # 只返回分数大于0的结果
                results.append((idx, float(scores[idx])))

        return results

    def get_document(self, idx: int) -> str:
        """根据索引获取文档"""
        if 0 <= idx < len(self.documents):
            return self.documents[idx]
        return ""


class LocalRerankerModel:
    """本地重排模型"""

    def __init__(self, model_path: str, **params):
        """
        初始化重排模型

        Args:
            model_path: 模型路径
            params: 额外参数
        """
        self.model_path = model_path
        self.device = params.get("device", DEVICE)
        self.max_length = params.get("max_length", 512)
        self.batch_size = params.get("batch_size", 4)

        print(f"🔧 初始化本地重排模型: {model_path}")
        self._init_model()

    def _init_model(self):
        """初始化模型"""
        try:
            # 加载tokenizer和模型
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # 移动到设备
            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"✅ 重排模型加载成功！")

        except Exception as e:
            print(f"❌ 重排模型加载失败: {e}")
            raise

    def rerank_batch(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        """
        批量重排文档

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top_k个结果，None表示返回全部

        Returns:
            列表，元素为(文档索引, 分数)
        """
        if not documents:
            return []

        scores = []

        # 批量处理
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]

            # 准备输入
            pairs = [[query, doc] for doc in batch_docs]

            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                # 移动到设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 前向传播
                outputs = self.model(**inputs)

                # 获取分数（logits）
                batch_scores = outputs.logits[:, -1].cpu().numpy()

                scores.extend(batch_scores)

        # 构建结果列表
        results = [(i, float(score)) for i, score in enumerate(scores)]

        # 按分数降序排序
        results.sort(key=lambda x: x[1], reverse=True)

        # 返回top_k个结果
        if top_k is not None:
            results = results[:top_k]

        return results

    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        """重排文档（兼容性方法）"""
        return self.rerank_batch(query, documents, top_k)


class HybridRetriever:
    """混合检索器：BM25 + Embedding + Reranker"""

    def __init__(self,
                 embedding_retriever,  # 向量检索器（ChromaDBManager）
                 bm25_retriever: BM25Retriever = None,
                 reranker_model: LocalRerankerModel = None,
                 retrieval_mode: str = "hybrid",
                 hybrid_weights: Tuple[float, float] = (0.4, 0.6),
                 reranker_enabled: bool = True,
                 reranker_top_k: int = 4):
        """
        初始化混合检索器

        Args:
            embedding_retriever: 向量检索器实例
            bm25_retriever: BM25检索器实例
            reranker_model: 重排模型实例
            retrieval_mode: 检索模式，可选 "vector", "bm25", "hybrid"
            hybrid_weights: 混合权重，(bm25_weight, vector_weight)
            reranker_enabled: 是否启用重排
            reranker_top_k: 重排后返回的数量
        """
        self.embedding_retriever = embedding_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker_model = reranker_model

        self.retrieval_mode = retrieval_mode
        self.hybrid_weights = hybrid_weights
        self.reranker_enabled = reranker_enabled and reranker_model is not None
        self.reranker_top_k = reranker_top_k

        # 存储所有文档的文本内容，用于BM25和重排
        self.all_documents = []

        print(f"🔍 初始化混合检索器 - 模式: {retrieval_mode}")
        if retrieval_mode == "hybrid":
            print(f"   混合权重: BM25={hybrid_weights[0]}, 向量={hybrid_weights[1]}")
        if self.reranker_enabled:
            print(f"   启用重排 - 返回数量: {reranker_top_k}")

    def update_documents(self, documents: List[str]):
        """更新所有文档（用于BM25索引）"""
        self.all_documents = documents.copy()

        if self.bm25_retriever:
            self.bm25_retriever.update_documents(documents)

    def add_documents(self, documents: List[str]):
        """添加文档（用于BM25索引）"""
        self.all_documents.extend(documents)

        if self.bm25_retriever:
            self.bm25_retriever.add_documents(documents)

    def search(self, query: str, k: int = 10, score_threshold: float = 0.3) -> List[Tuple[str, float]]:
        """
        混合检索

        Args:
            query: 查询文本
            k: 初始检索数量
            score_threshold: 分数阈值

        Returns:
            列表，元素为(文档文本, 分数)
        """
        # 获取初始检索结果
        if self.retrieval_mode == "vector":
            initial_results = self._vector_search(query, k, score_threshold)
        elif self.retrieval_mode == "bm25":
            initial_results = self._bm25_search(query, k, score_threshold)
        elif self.retrieval_mode == "hybrid":
            initial_results = self._hybrid_search(query, k, score_threshold)
        else:
            raise ValueError(f"未知的检索模式: {self.retrieval_mode}")

        # 如果没有结果，直接返回
        if not initial_results:
            return []

        # 重排
        if self.reranker_enabled:
            reranked_results = self._rerank(query, initial_results)
            return reranked_results

        return initial_results

    def _vector_search(self, query: str, k: int, score_threshold: float) -> List[Tuple[str, float]]:
        """向量检索"""
        if not self.embedding_retriever or not self.embedding_retriever.vector_store:
            return []

        try:
            # 使用向量检索器搜索
            results = self.embedding_retriever.search(query, k=k, score_threshold=score_threshold)

            # 转换为(文档文本, 分数)格式
            formatted_results = []
            for doc, score in results:
                formatted_results.append((doc.page_content, float(score)))

            return formatted_results

        except Exception as e:
            print(f"❌ 向量检索失败: {e}")
            return []

    def _bm25_search(self, query: str, k: int, score_threshold: float) -> List[Tuple[str, float]]:
        """BM25检索"""
        if not self.bm25_retriever or not self.all_documents:
            return []

        try:
            # BM25检索
            results = self.bm25_retriever.search(query, k=k)

            # 转换为(文档文本, 分数)格式
            formatted_results = []
            for idx, score in results:
                if score >= score_threshold:
                    doc_text = self.bm25_retriever.get_document(idx)
                    if doc_text:
                        formatted_results.append((doc_text, score))

            return formatted_results

        except Exception as e:
            print(f"❌ BM25检索失败: {e}")
            return []

    def _hybrid_search(self, query: str, k: int, score_threshold: float) -> List[Tuple[str, float]]:
        """混合检索"""
        # 分别进行向量检索和BM25检索
        vector_results = self._vector_search(query, k, score_threshold)
        bm25_results = self._bm25_search(query, k, score_threshold)

        # 如果没有结果，返回空列表
        if not vector_results and not bm25_results:
            return []

        # 使用归一化分数进行融合
        hybrid_scores = defaultdict(float)

        # 处理向量检索结果
        if vector_results:
            # 归一化向量检索分数
            vector_scores = [score for _, score in vector_results]
            if vector_scores:
                max_vector_score = max(vector_scores)
                if max_vector_score > 0:
                    for doc_text, score in vector_results:
                        normalized_score = score / max_vector_score
                        hybrid_scores[doc_text] += normalized_score * self.hybrid_weights[1]

        # 处理BM25检索结果
        if bm25_results:
            # 归一化BM25分数（已在BM25Retriever中归一化）
            for doc_text, score in bm25_results:
                hybrid_scores[doc_text] += score * self.hybrid_weights[0]

        # 按混合分数排序
        sorted_results = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 返回top-k结果
        return sorted_results[:k]

    def _rerank(self, query: str, initial_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """重排检索结果"""
        try:
            # 提取文档文本
            documents = [doc_text for doc_text, _ in initial_results]

            # 重排
            reranked_indices = self.reranker_model.rerank(
                query,
                documents,
                top_k=self.reranker_top_k
            )

            # 构建重排后的结果
            reranked_results = []
            for idx, rerank_score in reranked_indices:
                if idx < len(initial_results):
                    doc_text, original_score = initial_results[idx]
                    # 使用重排分数作为最终分数
                    reranked_results.append((doc_text, float(rerank_score)))

            return reranked_results

        except Exception as e:
            print(f"❌ 重排失败: {e}")
            # 重排失败时返回原始结果
            return initial_results[:self.reranker_top_k]

    def get_retrieval_info(self) -> Dict[str, Any]:
        """获取检索器信息"""
        return {
            "retrieval_mode": self.retrieval_mode,
            "hybrid_weights": self.hybrid_weights,
            "reranker_enabled": self.reranker_enabled,
            "reranker_top_k": self.reranker_top_k,
            "total_documents": len(self.all_documents),
            "has_bm25_index": self.bm25_retriever is not None,
            "has_reranker": self.reranker_model is not None
        }