"""RAG核心文件：向量库、检索、Prompt、查询逻辑"""
import os
import shutil
import traceback
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import (
    LOCAL_MODEL_PATH, DEVICE, PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
    SEARCH_K, PERSIST_DIR, DASHSCOPE_API_KEY, TEMPERATURE, MAX_TOKENS
)
from utils import load_pdf, split_text, validate_config
from embeddings import Qwen3Embeddings
from llm import DashScopeChatModel

class NaiveRAG:
    def __init__(self):
        validate_config(DASHSCOPE_API_KEY)
        # 删除旧向量库
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
            print(f"✅ 已删除旧向量库：{PERSIST_DIR}")

        self.embeddings = self._init_embeddings()
        self.pdf_docs = self._load_and_split_pdf()
        self.vector_db = self._build_vector_db()
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": SEARCH_K})
        self.llm = self._init_llm()
        self.rag_chain = self._build_rag_chain()
        print("✅ RAG系统初始化完成！")

    def _init_embeddings(self):
        return Qwen3Embeddings(model_path=LOCAL_MODEL_PATH, device=DEVICE)

    def _load_and_split_pdf(self):
        full_text = load_pdf(PDF_PATH)
        return split_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP, PDF_PATH)

    def _build_vector_db(self):
        os.makedirs(PERSIST_DIR, exist_ok=True)
        vector_db = Chroma.from_documents(
            documents=self.pdf_docs,
            embedding=self.embeddings,
            persist_directory=PERSIST_DIR
        )
        vector_db.persist()
        print(f"✅ 向量库构建成功：{PERSIST_DIR}")
        return vector_db

    def _init_llm(self):
        llm = DashScopeChatModel()
        llm.api_key = DASHSCOPE_API_KEY
        return llm

    def _build_rag_chain(self):
        """构建RAG链，优化Prompt"""
        prompt = ChatPromptTemplate.from_template("""
        请严格按照以下规则处理用户问题，必须遵守所有要求：
        1. 信息完整性：提取上下文所有与问题相关的关键信息，不遗漏任何段落、定义、属性等内容；
        2. 表述方式：不要直接复制原文，用自己的话逻辑连贯地重组信息，可适当分点（如果信息点多）；
        3. 语言风格：中文流畅、正式，符合文档表述逻辑，不添加外部知识；
        4. 无相关信息：若上下文无相关内容，直接回答“无法从文档中找到相关答案”。

        上下文：
        {context}

        用户问题：
        {question}
        """)

        def format_docs(docs):
            context = "\n\n".join([doc.page_content for doc in docs])
            print("\n🔍 检索到的相关上下文：")
            print("=" * 60)
            print(context)
            print("=" * 60)
            return context

        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("✅ RAG链构建成功")
        return rag_chain

    def query(self, question: str) -> str:
        print(f"\n❓ 用户问题：{question}")
        print("🔍 正在检索并生成自然完整的回答...")
        try:
            answer = self.rag_chain.invoke(question)
            print(f"\n💡 最终回答：")
            print("-" * 50)
            print(answer)
            print("-" * 50)
            return answer
        except Exception as e:
            print(f"❌ 查询失败：{str(e)}")
            traceback.print_exc()
            return ""