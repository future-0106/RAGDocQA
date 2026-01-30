"""RAG核心逻辑：向量库构建、检索、Prompt构建、LLM调用"""
import chromadb
from chromadb.config import Settings
from utils import check_cache_valid, save_cache_meta, CACHE_DIR
import os  # 新增导入

class NaiveRAG:
    def __init__(self, embedding_model, llm_model):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vector_db = None  # 向量库实例
        self.collection = None  # 文本片段集合
        self.current_pdf_id = None  # 当前PDF的内容哈希ID

    def load_pdf_and_build_vector_db(self, pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        加载PDF并构建向量库（仅基于修改时间触发更新）
        :param pdf_path: PDF文件路径
        :param chunk_size: 文本分割大小
        :param chunk_overlap: 文本重叠长度
        """
        # 1. 检查缓存是否有效（仅校验修改时间）
        cache_valid, content_hash = check_cache_valid(pdf_path)
        self.current_pdf_id = content_hash
        cache_db_path = os.path.join(CACHE_DIR, content_hash, "chroma_db")

        if cache_valid:
            # 2. 缓存有效 → 加载缓存的向量库（忽略文件名/路径变化）
            print(f"\n✅ 检测到有效缓存（内容未修改），直接加载向量库（内容哈希：{content_hash[:8]}...）")
            print(f"⚠️ 注意：当前文件路径「{pdf_path}」与缓存记录路径可能不同，但内容一致，复用缓存")
            self.vector_db = chromadb.PersistentClient(
                path=cache_db_path,
                settings=Settings(allow_reset=True)
            )
            # 获取已存在的集合
            self.collection = self.vector_db.get_or_create_collection("pdf_chunks")
            print(f"✅ 缓存向量库加载完成，共{self.collection.count()}个文本片段")
            return

        # 3. 缓存无效（修改时间变化）→ 重新处理PDF并构建向量库
        print(f"\n⚠️ 文件修改时间变化，重新生成向量（内容哈希：{content_hash[:8]}...）")
        print(f"🔄 正在处理PDF：{pdf_path}")

        # 提取PDF文本、分割
        from utils import load_pdf, split_text
        pdf_text = load_pdf(pdf_path)
        docs = split_text(
            text=pdf_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            pdf_path=pdf_path
        )

        # 构建向量库并持久化到缓存目录
        self.vector_db = chromadb.PersistentClient(
            path=cache_db_path,
            settings=Settings(allow_reset=True)
        )

        # 修复：删除旧集合（兼容新版chromadb）
        try:
            # 尝试删除已有集合
            self.vector_db.delete_collection("pdf_chunks")
        except Exception as e:
            # 集合不存在时忽略错误
            print(f"ℹ️ 集合不存在，无需删除：{e}")

        # 创建新集合
        self.collection = self.vector_db.get_or_create_collection("pdf_chunks")

        # 生成嵌入向量并添加到集合
        print("🔄 正在生成文本向量...")
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        ids = [f"chunk_{i}" for i in range(len(docs))]
        embeddings = self.embedding_model.embed_documents(texts)

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        # 保存缓存元数据（记录最新修改时间）
        save_cache_meta(pdf_path, content_hash)
        print(f"✅ 向量库重建完成并保存到缓存（内容哈希：{content_hash[:8]}...），共{self.collection.count()}个文本片段")

    def query(self, question: str, search_k: int = 8) -> str:
        """
        基于PDF内容回答问题
        :param question: 用户问题
        :param search_k: 检索返回的相关片段数量
        :return: 生成的回答
        """
        if self.vector_db is None or self.collection is None:
            raise ValueError("❌ 请先调用load_pdf_and_build_vector_db加载PDF！")

        # 生成问题向量
        print(f"\n🔍 正在检索与「{question}」相关的文本片段...")
        query_embedding = self.embedding_model.embed_query(question)

        # 检索相关片段
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_k
        )

        # 构建Prompt
        context = "\n".join(results["documents"][0])
        prompt = f"""基于以下上下文信息回答问题，仅使用上下文里的内容，不要编造信息：

上下文：
{context}

问题：{question}
回答："""

        # 调用LLM生成回答
        print("🤖 正在生成回答...")
        response = self.llm_model.invoke(prompt)
        return response.content.strip()