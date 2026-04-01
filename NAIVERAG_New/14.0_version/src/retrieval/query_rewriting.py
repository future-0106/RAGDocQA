"""
查询改写模块：根据检索模式与权重自动选择改写策略
支持查询扩展、HyDE、多查询生成、子问题分解、组合改写
完全独立，通过依赖注入使用LLM实例
"""
from typing import List, Optional
import re

class QueryRewriter:
    """查询改写器，依赖LLM实例"""

    def __init__(self, llm=None, enabled=False, max_queries=3, use_llm=True):
        """
        Args:
            llm: 用于生成文本的LLM实例（必须实现 _call(prompt) 方法）
            enabled: 是否启用改写
            max_queries: 最大生成查询数
            use_llm: 是否使用LLM（若为False，所有策略仅返回原查询）
        """
        self.llm = llm
        self.enabled = enabled
        self.max_queries = max_queries
        self.use_llm = use_llm

    # ---------- 策略选择入口 ----------
    def rewrite(self, query: str, retrieval_mode: str, hybrid_weights: tuple) -> List[str]:
        """
        根据检索配置选择改写策略，返回改写后的查询列表（始终包含原查询）
        """
        if not self.enabled or not self.use_llm or not self.llm:
            return [query]

        bm25_weight, vector_weight = hybrid_weights
        is_bm25_dominant = (retrieval_mode == "bm25") or \
                           (retrieval_mode == "hybrid" and bm25_weight > 0.6)
        is_vector_dominant = (retrieval_mode == "vector") or \
                             (retrieval_mode == "hybrid" and vector_weight > 0.6)

        if is_bm25_dominant:
            return self.query_expansion(query)
        elif is_vector_dominant:
            return self.vector_dominant_rewrite(query)
        else:
            return self.combined_rewriting(query)

    # ---------- 内置提示模板（可被外部配置覆盖）----------
    _DEFAULT_EXPANSION_PROMPT = "请为以下查询生成{num}个同义改写版本，每行一个，不要序号和额外文字：\n{query}"
    _DEFAULT_HYDE_PROMPT = "请为以下问题生成一个可能的答案片段（约50字）：\n{query}"
    _DEFAULT_MULTI_QUERY_PROMPT = "请将以下问题改写为{num}个不同角度的表述，每行一个，不要序号：\n{query}"
    _DEFAULT_DECOMPOSE_PROMPT = "请将以下复杂问题分解为最多{num}个独立子问题，每行一个，不要序号：\n{query}"

    # ---------- 内部辅助 ----------
    def _clean_queries(self, raw_queries: List[str]) -> List[str]:
        """清理LLM生成的查询，移除特殊标记和思考过程"""
        cleaned = []
        for q in raw_queries:
            q = q.strip()
            if not q:
                continue
            if '<|im_start|>' in q or '<think>' in q:
                continue
            cleaned.append(q)
        return cleaned

    # ---------- 具体改写策略 ----------
    def query_expansion(self, query: str) -> List[str]:
        """查询扩展：生成同义词/相关词变体"""
        prompt = self._DEFAULT_EXPANSION_PROMPT.format(num=self.max_queries, query=query)
        try:
            response = self.llm._call(prompt)
            expansions = [line.strip() for line in response.split('\n') if line.strip()]
            expansions = self._clean_queries(expansions)
            expansions = list(set(expansions))[:self.max_queries]
            return [query] + expansions
        except Exception as e:
            print(f"[WARNING] Query expansion failed: {e}")
            return [query]

    def hyde(self, query: str) -> str:
        """假设文档生成：生成一个虚构的答案片段作为查询"""
        prompt = self._DEFAULT_HYDE_PROMPT.format(query=query)
        try:
            response = self.llm._call(prompt)
            response = response.strip()[:200]
            if '<|im_start|>' in response or '<think>' in response:
                return query
            return response
        except Exception:
            return query

    def multi_query_generation(self, query: str) -> List[str]:
        """多查询生成：生成多个不同角度的问题表述"""
        prompt = self._DEFAULT_MULTI_QUERY_PROMPT.format(num=self.max_queries, query=query)
        try:
            response = self.llm._call(prompt)
            variants = [line.strip() for line in response.split('\n') if line.strip()]
            variants = self._clean_queries(variants)
            variants = list(set(variants))[:self.max_queries]
            return [query] + variants
        except Exception:
            return [query]

    def subquestion_decomposition(self, query: str) -> List[str]:
        """子问题分解：将复杂问题拆分为多个独立子问题"""
        prompt = self._DEFAULT_DECOMPOSE_PROMPT.format(num=self.max_queries, query=query)
        try:
            response = self.llm._call(prompt)
            subqs = [line.strip() for line in response.split('\n') if line.strip()]
            subqs = self._clean_queries(subqs)
            return subqs[:self.max_queries]
        except Exception:
            return [query]

    def vector_dominant_rewrite(self, query: str) -> List[str]:
        """向量主导组合策略：HyDE + 多查询 + 子问题分解，去重后返回"""
        results = [query]
        hyde_q = self.hyde(query)
        if hyde_q != query:
            hyde_clean = self._clean_queries([hyde_q])
            if hyde_clean:
                results.append(hyde_clean[0])
        results.extend(self.multi_query_generation(query)[1:])
        results.extend(self.subquestion_decomposition(query))
        results = self._clean_queries(results)
        seen = {query}
        unique = [query]
        for q in results[1:]:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique[:self.max_queries * 2]

    def combined_rewriting(self, query: str) -> List[str]:
        """均衡型组合改写：查询扩展 + HyDE"""
        results = [query]
        expansions = self.query_expansion(query)[1:]
        expansions = self._clean_queries(expansions)
        results.extend(expansions)
        hyde_q = self.hyde(query)
        if hyde_q != query:
            hyde_clean = self._clean_queries([hyde_q])
            if hyde_clean:
                results.append(hyde_clean[0])
        seen = {query}
        unique = [query]
        for q in results[1:]:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique[:self.max_queries * 2]