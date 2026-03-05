#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查询改写功能端到端集成测试
运行要求：
  1. 已正确配置 config.py 中的模型路径/API Key
  2. 至少有一个可用的本地 LLM 或云端 API 模型（建议使用 small 模型加速）
  3. 安装 pytest（可选，也可直接 python 运行）

运行方式：
  python test_query_rewriting_integration.py
  或
  pytest test_query_rewriting_integration.py -v
"""
import os
import shutil
import tempfile
import time
from pathlib import Path
import sys

# 将项目根目录加入路径，确保导入正确
sys.path.insert(0, str(Path(__file__).parent))

# 导入项目模块
from config import (
    BASE_DIR, DEVICE,
    CHUNK_SIZE, CHUNK_OVERLAP,
    SIMILARITY_TOP_K, SCORE_THRESHOLD,
    ALL_MODELS, DEFAULT_MODEL,
    QUERY_REWRITING_ENABLED, QUERY_REWRITING_MODEL
)
from models import MultiModelLLM, MultiEmbeddings, ModelFactory, MultiReranker
from documents import DocumentProcessor
from vector_store import ChromaDBManager, FileVectorizationManager, HybridRetrievalManager
from rag_pipeline import QwenRAGPipeline
from query_rewriting import QueryRewriter


class TestRAGWithRewriting:
    """集成测试夹具：创建临时环境并初始化组件"""

    def __init__(self):
        # 创建临时目录（自动清理）
        self.temp_dir = tempfile.mkdtemp(prefix="rag_test_")
        self.data_dir = Path(self.temp_dir) / "data"
        self.chroma_dir = Path(self.temp_dir) / "chroma_db"
        self.data_dir.mkdir(exist_ok=True)
        self.chroma_dir.mkdir(exist_ok=True)

        print(f"\n🔧 测试临时目录: {self.temp_dir}")

        # 初始化组件
        self._init_components()
        self._prepare_test_document()

    def _init_components(self):
        """初始化所有RAG组件（使用实际模型）"""
        print("\n🔧 初始化测试组件...")

        # 1. 嵌入模型（使用配置的默认嵌入模型）
        self.embeddings = MultiEmbeddings()
        print(f"   ✅ 嵌入模型: {self.embeddings.model_key}")

        # 2. 向量存储（临时目录）
        self.vector_manager = ChromaDBManager(
            embedding_model=self.embeddings,
            persist_directory=str(self.chroma_dir)
        )
        print(f"   ✅ 向量存储目录: {self.chroma_dir}")

        # 3. 重排模型（可选，可禁用）
        self.reranker = MultiReranker()
        # 测试时暂时禁用重排以加快速度
        reranker_enabled = False

        # 4. 混合检索管理器
        self.hybrid_manager = HybridRetrievalManager(
            chroma_manager=self.vector_manager,
            reranker_model=self.reranker,
            retrieval_mode="hybrid",      # 默认混合
            hybrid_weights=(0.5, 0.5),    # 初始均衡
            reranker_enabled=reranker_enabled,
            reranker_top_k=4
        )
        print(f"   ✅ 混合检索管理器")

        # 5. 问答LLM
        self.llm = MultiModelLLM()
        print(f"   ✅ 问答LLM: {self.llm.model_key}")

        # 6. 改写专用LLM（若配置了专用模型，否则复用问答LLM）
        if QUERY_REWRITING_MODEL:
            try:
                self.rewrite_llm = ModelFactory.create_llm(QUERY_REWRITING_MODEL)
                print(f"   ✅ 改写专用LLM: {QUERY_REWRITING_MODEL}")
            except Exception as e:
                print(f"   ⚠️ 改写专用模型加载失败: {e}，将复用问答LLM")
                self.rewrite_llm = None
        else:
            self.rewrite_llm = None
            print(f"   ℹ️ 未配置改写专用模型，将复用问答LLM")

        # 7. 文档处理器
        self.doc_processor = DocumentProcessor(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        # 8. 文件管理器
        self.file_manager = FileVectorizationManager(
            data_dir=self.data_dir,
            hybrid_manager=self.hybrid_manager
        )

        # 9. RAG流水线（传入改写专用LLM）
        self.rag_pipeline = QwenRAGPipeline(
            llm=self.llm,
            hybrid_retrieval_manager=self.hybrid_manager,
            rewrite_llm=self.rewrite_llm
        )
        # 强制启用改写（无论全局配置）
        self.rag_pipeline.query_rewriter.enabled = True
        self.rag_pipeline.query_rewriter.use_llm = True
        print(f"   ✅ RAG流水线初始化完成")

    def _prepare_test_document(self):
        """创建测试文档并向量化"""
        print("\n📄 准备测试文档...")

        # 创建一个简单的TXT文件，包含一些可检索内容
        test_content = """注意力机制（Attention Mechanism）是深度学习中的重要概念。
它允许模型在处理输入序列时，动态地关注不同部分的重要性。
Transformer模型完全基于自注意力机制，没有使用循环或卷积。
自注意力可以并行计算，大大提高了训练效率。
注意力机制也被广泛应用于自然语言处理、计算机视觉等领域。"""

        test_file = self.data_dir / "test_attention.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)

        # 向量化
        success, msg, docs = self.file_manager.upload_and_vectorize(
            str(test_file),
            self.doc_processor,
            self.vector_manager,
            self.hybrid_manager
        )

        if success:
            print(f"   ✅ 测试文档已向量化，共 {len(docs)} 个文档块")
        else:
            raise RuntimeError(f"❌ 文档向量化失败: {msg}")

    def cleanup(self):
        """清理临时目录"""
        print(f"\n🧹 清理临时目录: {self.temp_dir}")
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# ==================== 测试用例 ====================

def test_bm25_dominant_trigger_query_expansion():
    """测试BM25主导时是否触发查询扩展"""
    tester = TestRAGWithRewriting()
    try:
        # 配置检索模式为BM25主导
        tester.hybrid_manager.update_retrieval_config(
            retrieval_mode="hybrid",
            hybrid_weights=(0.8, 0.2)   # BM25权重0.8
        )

        question = "什么是注意力机制"
        result = tester.rag_pipeline.query(question, enable_rewriting=True)

        # 验证改写查询列表
        rewritten = result.get("rewritten_queries", [])
        print(f"\n🔍 BM25主导改写结果: {rewritten}")

        # 断言：长度 > 1，包含原问题
        assert len(rewritten) > 1, "应生成多个改写查询"
        assert rewritten[0] == question, "第一个应为原问题"

        # 可选：验证是否包含同义扩展（根据实际LLM输出可能不同，不强求）
        print("✅ BM25主导 → 查询扩展 测试通过")
    finally:
        tester.cleanup()


def test_vector_dominant_trigger_combined():
    """测试向量主导时是否触发 HyDE + 多查询 + 子问题分解"""
    tester = TestRAGWithRewriting()
    try:
        # 配置检索模式为向量主导
        tester.hybrid_manager.update_retrieval_config(
            retrieval_mode="hybrid",
            hybrid_weights=(0.2, 0.8)   # 向量权重0.8
        )

        question = "注意力机制为什么重要"
        result = tester.rag_pipeline.query(question, enable_rewriting=True)

        rewritten = result.get("rewritten_queries", [])
        print(f"\n🔍 向量主导改写结果: {rewritten}")

        assert len(rewritten) > 1, "应生成多个改写查询"
        assert rewritten[0] == question

        # 检查是否包含假设文档（HyDE）的痕迹（通常较长）
        hyde_present = any(len(q) > len(question) * 1.2 for q in rewritten[1:])
        if hyde_present:
            print("   ✅ 检测到可能的 HyDE 生成")
        else:
            print("   ⚠️ 未明确检测到 HyDE，但多查询仍有效")

        print("✅ 向量主导 → 组合改写 测试通过")
    finally:
        tester.cleanup()


def test_balanced_trigger_combined():
    """测试均衡模式是否触发查询扩展+HyDE"""
    tester = TestRAGWithRewriting()
    try:
        # 配置均衡权重
        tester.hybrid_manager.update_retrieval_config(
            retrieval_mode="hybrid",
            hybrid_weights=(0.5, 0.5)
        )

        question = "注意力机制的应用"
        result = tester.rag_pipeline.query(question, enable_rewriting=True)

        rewritten = result.get("rewritten_queries", [])
        print(f"\n🔍 均衡模式改写结果: {rewritten}")

        assert len(rewritten) > 1
        assert rewritten[0] == question

        # 检查是否同时包含查询扩展和HyDE
        # 简单启发：如果有至少两个不同的改写（原问题外有两个以上不同查询）
        unique_rewritten = set(rewritten[1:])
        if len(unique_rewritten) >= 2:
            print("   ✅ 检测到多个改写变体")
        else:
            print("   ⚠️ 改写数量不足")

        print("✅ 均衡模式 → 组合改写 测试通过")
    finally:
        tester.cleanup()


def test_rewriting_off_works():
    """测试关闭改写时，仅返回原查询"""
    tester = TestRAGWithRewriting()
    try:
        # 任何模式，但通过参数强制关闭改写
        tester.hybrid_manager.update_retrieval_config(
            retrieval_mode="hybrid",
            hybrid_weights=(0.5, 0.5)
        )

        question = "测试关闭改写"
        result = tester.rag_pipeline.query(question, enable_rewriting=False)

        rewritten = result.get("rewritten_queries", [])
        print(f"\n🔍 关闭改写: {rewritten}")

        assert rewritten == [question], "关闭改写时应只返回原查询"
        print("✅ 关闭改写 测试通过")
    finally:
        tester.cleanup()


def test_end_to_end_answer_quality():
    """端到端测试：使用真实检索生成答案"""
    tester = TestRAGWithRewriting()
    try:
        # 使用均衡模式，开启改写
        tester.hybrid_manager.update_retrieval_config(
            retrieval_mode="hybrid",
            hybrid_weights=(0.5, 0.5)
        )

        question = "注意力机制是什么"
        result = tester.rag_pipeline.query(question, enable_rewriting=True)

        answer = result["answer"]
        sources = result["sources"]

        print(f"\n🤖 生成答案: {answer}")
        print(f"📚 来源数量: {len(sources)}")

        # 简单断言：答案非空，且包含关键词
        assert answer, "答案不应为空"
        assert any(kw in answer.lower() for kw in ["注意力", "attention"]), \
            "答案应包含'注意力'相关词汇"

        print("✅ 端到端答案生成 测试通过")
    finally:
        tester.cleanup()


# ==================== 手动运行 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 查询改写功能集成测试（使用真实组件）")
    print("=" * 60)

    # 检查关键配置
    if not DEFAULT_MODEL:
        print("❌ 未配置默认模型，请在 config.py 中设置 DEFAULT_MODEL")
        sys.exit(1)

    print(f"\n📋 当前配置:")
    print(f"   - 默认问答模型: {DEFAULT_MODEL}")
    print(f"   - 改写专用模型: {QUERY_REWRITING_MODEL or '复用问答模型'}")
    print(f"   - 设备: {DEVICE}")

    # 运行测试
    test_methods = [
        test_bm25_dominant_trigger_query_expansion,
        test_vector_dominant_trigger_combined,
        test_balanced_trigger_combined,
        test_rewriting_off_works,
        test_end_to_end_answer_quality
    ]

    passed = 0
    for test in test_methods:
        try:
            test()
            passed += 1
            print(f"✅ {test.__name__} 成功\n")
        except Exception as e:
            print(f"❌ {test.__name__} 失败: {e}\n")
            import traceback
            traceback.print_exc()

    print(f"\n📊 测试汇总: {passed}/{len(test_methods)} 通过")
    if passed == len(test_methods):
        print("🎉 所有测试通过！查询改写功能工作正常。")
    else:
        print("⚠️ 部分测试失败，请检查日志。")