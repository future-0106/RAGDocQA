# -*- coding: utf-8 -*-
"""
检索效果评估脚本
评估查询改写对RAG系统检索效果的影响
"""

import json
import os
import sys
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

# 添加项目路径
PROJECT_DIR = r"D:\projects\fastapi_langchain_env\NAIVERAG_New\10.0_version"
sys.path.insert(0, PROJECT_DIR)

# 配置
TEST_DATA_FILE = os.path.join(PROJECT_DIR, "evaluate", "test_dataset.json")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "evaluate", "results")
RETRIEVAL_K = 10  # 检索Top-K


@dataclass
class EvaluationResult:
    """单次评估结果"""
    query: str
    question_type: str
    
    # 基线检索（无查询改写）
    baseline_recall: float
    baseline_precision: float
    baseline_mrr: float
    baseline_hit: bool
    
    # 改写后检索（有查询改写）
    rewritten_recall: float
    rewritten_precision: float
    rewritten_mrr: float
    rewritten_hit: bool
    
    # 改进情况
    recall_improvement: float
    precision_improvement: float
    mrr_improvement: float
    hit_improved: bool


def calculate_recall(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    """计算召回率"""
    if not relevant_docs:
        return 0.0
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def calculate_precision(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    """计算精确率"""
    if not retrieved_docs:
        return 0.0
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    return len(retrieved_set & relevant_set) / len(retrieved_docs)


def calculate_mrr(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    """计算平均倒数排名"""
    if not relevant_docs or not retrieved_docs:
        return 0.0
    relevant_set = set(relevant_docs)
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_set:
            return 1.0 / i
    return 0.0


def calculate_hit_rate(retrieved_docs: List[str], relevant_docs: List[str]) -> bool:
    """计算命中率"""
    if not relevant_docs or not retrieved_docs:
        return False
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    return len(retrieved_set & relevant_set) > 0


def load_test_data(file_path: str) -> Dict:
    """加载测试数据"""
    print(f"加载测试数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  - 总数: {data['total_count']}条")
    return data


def init_rag_system():
    """初始化RAG系统"""
    print("\n初始化RAG系统...")
    
    try:
        # 导入项目模块
        from config import (
            DEFAULT_EMBEDDING_MODEL, 
            DEFAULT_RERANKER_MODEL,
            SIMILARITY_TOP_K,
            SCORE_THRESHOLD,
            RETRIEVAL_MODE,
            HYBRID_WEIGHTS,
            RERANKER_ENABLED,
            RERANKER_TOP_K
        )
        from models import MultiEmbeddings, MultiReranker
        from vector_store import ChromaDBManager, HybridRetrievalManager
        from query_rewriting import QueryRewriter
        from config import QUERY_REWRITING_ENABLED, QUERY_REWRITING_MODEL
        
        # 初始化组件
        print("  - 初始化嵌入模型...")
        embedding_model = MultiEmbeddings(DEFAULT_EMBEDDING_MODEL)
        
        print("  - 初始化向量存储...")
        chroma_manager = ChromaDBManager(
            embedding_model=embedding_model.embeddings,
            persist_directory=os.path.join(PROJECT_DIR, "chroma_db")
        )
        chroma_manager.load()
        
        print("  - 初始化重排模型...")
        reranker = MultiReranker(DEFAULT_RERANKER_MODEL)
        
        print("  - 初始化混合检索...")
        hybrid_manager = HybridRetrievalManager(
            chroma_manager=chroma_manager,
            reranker_model=reranker.rerank if reranker._model_instance else None,
            retrieval_mode=RETRIEVAL_MODE,
            hybrid_weights=HYBRID_WEIGHTS,
            reranker_enabled=RERANKER_ENABLED,
            reranker_top_k=RERANKER_TOP_K
        )
        
        # 初始化查询改写器（可选）
        query_rewriter = None
        if QUERY_REWRITING_ENABLED:
            from models import MultiModelLLM
            print("  - 初始化查询改写...")
            llm = MultiModelLLM(QUERY_REWRITING_MODEL)
            query_rewriter = QueryRewriter(
                llm=llm,
                enabled=True,
                max_queries=3,
                use_llm=True
            )
        
        print("  - RAG系统初始化完成!\n")
        
        return {
            "hybrid_manager": hybrid_manager,
            "query_rewriter": query_rewriter,
            "config": {
                "retrieval_k": RETRIEVAL_K,
                "score_threshold": SCORE_THRESHOLD
            }
        }
        
    except Exception as e:
        print(f"RAG系统初始化失败: {e}")
        print("将使用模拟检索进行演示...")
        return None


def mock_search(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """模拟检索（当RAG系统未初始化时使用）"""
    # 返回模拟结果
    import random
    results = []
    for i in range(min(k, 5)):
        results.append((f"doc_{i}", random.uniform(0.5, 0.9)))
    return results


def evaluate_single_query(
    rag_system: Dict,
    test_case: Dict,
    use_rewriting: bool = True
) -> Dict:
    """评估单个查询"""
    query = test_case["query"]
    question_type = test_case.get("question_type", "simple")
    
    # 获取相关文档（用于评估）
    # 注意：实际评估时需要预先标注相关文档
    # 这里使用答案中的关键词作为模拟
    relevant_docs = []
    
    # 检索
    if rag_system is None:
        # 使用模拟检索
        results = mock_search(query, RETRIEVAL_K)
    else:
        hybrid_manager = rag_system["hybrid_manager"]
        
        if use_rewriting and rag_system.get("query_rewriter"):
            # 使用查询改写
            rewriter = rag_system["query_rewriter"]
            retrieval_mode = "hybrid"
            hybrid_weights = (0.2, 0.8)
            
            # 执行查询改写
            rewritten_queries = rewriter.rewrite(query, retrieval_mode, hybrid_weights)
            
            # 使用改写后的查询进行检索
            all_results = []
            for q in rewritten_queries:
                results = hybrid_manager.search(
                    q, 
                    k=rag_system["config"]["retrieval_k"],
                    score_threshold=rag_system["config"]["score_threshold"]
                )
                all_results.extend(results)
            
            # 去重并排序
            seen = set()
            final_results = []
            for doc, score in all_results:
                if doc not in seen:
                    seen.add(doc)
                    final_results.append((doc, score))
            final_results.sort(key=lambda x: x[1], reverse=True)
            results = final_results[:RETRIEVAL_K]
        else:
            # 基线检索（不使用查询改写）
            results = hybrid_manager.search(
                query,
                k=rag_system["config"]["retrieval_k"],
                score_threshold=rag_system["config"]["score_threshold"]
            )
    
    # 提取文档ID（这里用文档内容的前50字符作为ID）
    retrieved_docs = [str(i) for i in range(len(results))]
    
    return {
        "retrieved_docs": retrieved_docs,
        "relevant_docs": relevant_docs,
        "question_type": question_type
    }


def evaluate_batch(rag_system: Dict, test_cases: List[Dict], use_rewriting: bool = True) -> List[EvaluationResult]:
    """批量评估"""
    print(f"\n开始评估 ({'使用查询改写' if use_rewriting else '基线检索'})...")
    
    results = []
    for i, test_case in enumerate(test_cases):
        if (i + 1) % 20 == 0:
            print(f"  进度: {i + 1}/{len(test_cases)}")
        
        result = evaluate_single_query(rag_system, test_case, use_rewriting)
        
        # 由于没有预先标注相关文档，这里使用模拟指标
        # 实际使用时需要预先标注每个查询的相关文档
        eval_result = EvaluationResult(
            query=test_case["query"][:50] + "...",
            question_type=result["question_type"],
            baseline_recall=0.0,
            baseline_precision=0.0,
            baseline_mrr=0.0,
            baseline_hit=False,
            rewritten_recall=0.0,
            rewritten_precision=0.0,
            rewritten_mrr=0.0,
            rewritten_hit=False,
            recall_improvement=0.0,
            precision_improvement=0.0,
            mrr_improvement=0.0,
            hit_improved=False
        )
        
        results.append(eval_result)
    
    return results


def generate_report(baseline_results: List[EvaluationResult], 
                   rewritten_results: List[EvaluationResult]) -> str:
    """生成评估报告"""
    report = []
    report.append("# 检索效果评估报告\n")
    
    # 统计
    total = len(baseline_results)
    
    # 按问题类型统计
    types = {}
    for r in baseline_results:
        t = r.question_type
        if t not in types:
            types[t] = {"baseline": 0, "rewritten": 0}
    
    report.append("## 评估摘要\n")
    report.append(f"- 测试用例总数: {total}\n")
    report.append(f"- 检索Top-K: {RETRIEVAL_K}\n")
    report.append("\n")
    
    # 注意：由于测试数据没有预先标注相关文档
    # 实际的召回率等指标需要预先构建相关文档集合
    report.append("## 注意事项\n")
    report.append("> 当前评估使用模拟指标，实际评估需要预先标注每个查询的相关文档。\n")
    report.append("> 请使用以下步骤构建标准评估：\n")
    report.append("> 1. 为每个测试查询预先标注相关文档ID列表\n")
    report.append("> 2. 在evaluate_single_query中使用预标注的相关文档\n")
    report.append("> 3. 重新运行评估\n")
    
    return "".join(report)


def main():
    """主函数"""
    print("="*60)
    print("查询改写效果评估")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载测试数据
    test_data = load_test_data(TEST_DATA_FILE)
    test_cases = test_data["test_cases"]
    
    # 限制测试数量（用于快速测试）
    max_test_count = 50  # 可调整
    test_cases = test_cases[:max_test_count]
    print(f"  - 实际测试数量: {len(test_cases)}\n")
    
    # 初始化RAG系统
    rag_system = init_rag_system()
    
    # 评估1：基线检索（不使用查询改写）
    print("="*60)
    baseline_results = evaluate_batch(rag_system, test_cases, use_rewriting=False)
    
    # 评估2：使用查询改写
    print("="*60)
    rewritten_results = evaluate_batch(rag_system, test_cases, use_rewriting=True)
    
    # 生成报告
    report = generate_report(baseline_results, rewritten_results)
    
    # 保存报告
    report_file = os.path.join(OUTPUT_DIR, "evaluation_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print(f"评估完成!")
    print(f"报告已保存: {report_file}")
    print("="*60)
    
    # 打印报告
    print("\n" + report)


if __name__ == "__main__":
    main()
