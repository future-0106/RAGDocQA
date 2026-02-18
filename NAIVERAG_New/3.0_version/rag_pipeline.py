"""
RAG流水线模块
"""
import re
from typing import List, Dict, Any
from config import MAX_CONTEXT_LENGTH


class QwenRAGPipeline:
    """Qwen RAG流水线"""

    def __init__(self, llm, vector_store_manager):
        self.llm = llm
        self.vector_manager = vector_store_manager

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

    def build_context(self, search_results, max_length: int = MAX_CONTEXT_LENGTH) -> str:
        """构建上下文字符串"""
        context_parts = []
        current_length = 0

        for i, (doc, score) in enumerate(search_results):
            doc_text = doc.page_content

            if current_length + len(doc_text) > max_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n".join(context_parts)

    def query(self, question: str, k: int = 3, score_threshold: float = 0.3) -> Dict[str, Any]:
        """执行查询"""
        print(f"\n🔍 检索中: '{question}'")

        # 1. 检索相关文档
        search_results = self.vector_manager.search(
            question,
            k=k,
            score_threshold=score_threshold
        )

        if not search_results:
            return {
                "question": question,
                "answer": "没有在文档中找到相关信息，无法回答这个问题。",
                "sources": [],
                "context": ""
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
                    "content": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "source_info": f"{doc.metadata.get('file_name', '未知文件')} "
                                   f"(类型: {doc.metadata.get('type', '未知')})"
                }
                for doc, score in search_results
            ],
            "context_length": len(context),
            "source_count": len(search_results)
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