"""
RAG流程和工具函数模块：整合RAG流水线和辅助工具
"""
import re
import warnings
from typing import List, Dict, Any, Tuple

from transformers import logging
from config import MAX_CONTEXT_LENGTH, ALL_MODELS, ALL_EMBEDDING_MODELS
from config import RETRIEVAL_MODE, HYBRID_WEIGHTS, RERANKER_ENABLED, RERANKER_TOP_K, SIMILARITY_TOP_K


class QwenRAGPipeline:
    """Qwen RAG流水线 - 支持混合检索"""

    def __init__(self, llm, hybrid_retrieval_manager):
        self.llm = llm
        self.hybrid_manager = hybrid_retrieval_manager

        # 优化的提示模板
        self.prompt_template = """基于以下上下文信息，直接回答问题。不要输出任何思考过程，不要输出选择题格式，直接给出答案。

上下文信息：
{context}

问题：{question}

要求：
1. 只使用上下文中的信息回答问题
2. 如果上下文没有相关信息，直接说"我不知道"
3. 不要输出任何"思考："、"分析："、"首先"、"然后"、"最后"等思考过程
4. 不要输出任何选择题格式（如A. B. C. D.）
5. 直接给出简洁明了的答案

答案："""

    def build_context(self, search_results: List[Tuple[str, float]], max_length: int = MAX_CONTEXT_LENGTH) -> str:
        """构建上下文字符串"""
        context_parts = []
        current_length = 0

        for i, (doc_text, score) in enumerate(search_results):
            if current_length + len(doc_text) > max_length:
                break

            # 添加文档和分数信息
            context_parts.append(f"[文档{i+1}，相关性分数:{score:.3f}]\n{doc_text}")
            current_length += len(doc_text)

        return "\n\n".join(context_parts)

    def query(self, question: str, k: int = None, score_threshold: float = 0.3) -> Dict[str, Any]:
        """执行查询 - 使用混合检索"""
        if k is None:
            k = SIMILARITY_TOP_K

        print(f"\n🔍 混合检索中: '{question}'")
        print(f"   检索模式: {self.hybrid_manager.hybrid_retriever.retrieval_mode}")
        if self.hybrid_manager.hybrid_retriever.retrieval_mode == "hybrid":
            print(f"   混合权重: BM25={self.hybrid_manager.hybrid_retriever.hybrid_weights[0]}, "
                  f"向量={self.hybrid_manager.hybrid_retriever.hybrid_weights[1]}")
        if self.hybrid_manager.hybrid_retriever.reranker_enabled:
            print(f"   启用重排 - 返回数量: {self.hybrid_manager.hybrid_retriever.reranker_top_k}")

        # 1. 混合检索相关文档
        search_results = self.hybrid_manager.search(
            question,
            k=k,
            score_threshold=score_threshold
        )

        if not search_results:
            return {
                "question": question,
                "answer": "没有在文档中找到相关信息，无法回答这个问题。",
                "sources": [],
                "context": "",
                "retrieval_mode": self.hybrid_manager.hybrid_retriever.retrieval_mode,
                "reranker_enabled": self.hybrid_manager.hybrid_retriever.reranker_enabled
            }

        # 2. 构建上下文
        context = self.build_context(search_results)

        # 3. 构建完整提示
        full_prompt = self.prompt_template.format(
            context=context,
            question=question
        )

        # 4. 调用LLM生成回答
        print("🤖 生成回答中...")
        answer = self.llm._call(full_prompt)

        # 5. 进一步清理回答
        answer = self._post_process_answer(answer)

        # 6. 准备返回结果
        result = {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "content": doc_text[:150] + "..." if len(doc_text) > 150 else doc_text,
                    "score": float(score),
                    "rank": i + 1
                }
                for i, (doc_text, score) in enumerate(search_results[:RERANKER_TOP_K])
            ],
            "context_length": len(context),
            "source_count": len(search_results),
            "retrieval_mode": self.hybrid_manager.hybrid_retriever.retrieval_mode,
            "reranker_enabled": self.hybrid_manager.hybrid_retriever.reranker_enabled,
            "hybrid_weights": self.hybrid_manager.hybrid_retriever.hybrid_weights if
                self.hybrid_manager.hybrid_retriever.retrieval_mode == "hybrid" else None
        }

        return result

    def _post_process_answer(self, answer: str) -> str:
        """后处理答案，确保格式正确"""
        # 移除答案开头的"答案："字样
        answer = re.sub(r'^答案[：:]\s*', '', answer)

        # 移除任何剩余的思考标记
        thought_patterns = [
            r'思考[:：].*',
            r'分析[:：].*',
            r'首先，.*',
            r'然后，.*',
            r'最后，.*',
            r'所以，.*',
            r'因此，.*',
            r'综上，.*',
            r'由此可见，.*',
        ]

        for pattern in thought_patterns:
            answer = re.sub(pattern, '', answer)

        # 移除多余的空格和空行
        answer = ' '.join(answer.split())

        # 如果回答以"答案"开头但没内容，重新处理
        if answer.startswith('答案') and len(answer) < 20:
            answer = "根据文档内容，我无法生成有效的回答。"

        return answer.strip()

    def update_retrieval_config(self,
                               retrieval_mode: str = None,
                               hybrid_weights: Tuple[float, float] = None,
                               reranker_enabled: bool = None,
                               reranker_top_k: int = None):
        """更新检索配置"""
        self.hybrid_manager.update_retrieval_config(
            retrieval_mode=retrieval_mode,
            hybrid_weights=hybrid_weights,
            reranker_enabled=reranker_enabled,
            reranker_top_k=reranker_top_k
        )

    def get_retrieval_info(self) -> Dict[str, Any]:
        """获取检索信息"""
        return self.hybrid_manager.get_retrieval_info()


# ==================== 工具函数模块 ====================

def setup_environment():
    """设置环境"""
    # 关闭transformers的特定警告
    logging.set_verbosity_error()
    warnings.filterwarnings("ignore",
                            message=".*generation_config.*default values have been modified.*")


def check_imports():
    """检查必要的导入"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ 使用 langchain_huggingface.HuggingFaceEmbeddings")
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print("⚠️  使用 langchain_community.HuggingFaceEmbeddings")
        except ImportError:
            print("❌ 无法导入 HuggingFaceEmbeddings")
            print("请安装: pip install langchain-huggingface 或 langchain-community")
            return False

    try:
        from langchain_chroma import Chroma
        print("✅ 使用 langchain_chroma.Chroma")
    except ImportError as e:
        print(f"❌ 无法导入 ChromaDB: {e}")
        print("请安装: pip install chromadb langchain-chroma")
        return False

    try:
        import jieba
        print("✅ 使用 jieba 分词")
    except ImportError:
        print("❌ 无法导入 jieba")
        print("请安装: pip install jieba")
        return False

    try:
        from rank_bm25 import BM25Okapi
        print("✅ 使用 BM25 检索")
    except ImportError:
        print("❌ 无法导入 BM25")
        print("请安装: pip install rank-bm25")
        return False

    return True


def display_menu():
    """显示主菜单"""
    print("\n" + "=" * 60)
    print("🚀 统一模型管理 RAG 系统")
    print("📚 支持PDF、TXT、MD文档格式")
    print("🤖 支持多本地模型和云端API模型")
    print("🔍 支持BM25+Embedding混合检索 + 重排")
    print("=" * 60)
    print("\n请选择操作:")
    print("  [1] 基于文档内容回答问题")
    print("  [2] 上传单个文件并向量化")
    print("  [3] 批量上传多个文件")
    print("  [4] 查看已上传文件列表")
    print("  [5] 查看系统状态和统计")
    print("  [6] 重新处理所有文件并重建向量存储")
    print("  [7] 模型管理（切换模型）")
    print("  [8] 检索配置管理")
    print("  [9] 清除屏幕")
    print("  [10] 退出系统")
    print("-" * 60)


def display_retrieval_menu():
    """显示检索配置菜单"""
    print("\n" + "=" * 60)
    print("🔍 检索配置管理系统")
    print("=" * 60)
    print("\n请选择操作:")
    print("  [1] 查看当前检索配置")
    print("  [2] 切换检索模式 (vector/bm25/hybrid)")
    print("  [3] 调整混合检索权重")
    print("  [4] 启用/禁用重排")
    print("  [5] 设置重排返回数量")
    print("  [6] 返回主菜单")
    print("-" * 60)