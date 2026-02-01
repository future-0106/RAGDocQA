"""RAG核心逻辑：多PDF向量库管理、全局检索、自动去重"""
import chromadb
from chromadb.config import Settings
from utils import (
    check_cache_valid, save_cache_meta, CACHE_DIR,
    get_file_content_hash, is_pdf_uploaded, add_upload_record,
    load_pdf, split_text, clean_text
)
import os
from config import GLOBAL_VECTOR_DB_PATH

class NaiveRAG:
    def __init__(self, embedding_model, llm_model):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        # 初始化全局向量库（所有PDF共用）
        self.vector_db = chromadb.PersistentClient(
            path=GLOBAL_VECTOR_DB_PATH,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        # 全局文本片段集合（所有PDF的chunk都存在这里）
        self.collection = self.vector_db.get_or_create_collection("all_pdf_chunks")

    def add_pdf(self, pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        添加PDF到全局向量库（自动去重）
        :param pdf_path: PDF文件路径
        :param chunk_size: 文本分割大小
        :param chunk_overlap: 文本重叠长度
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ PDF文件不存在：{pdf_path}")

        # 1. 计算内容哈希，判断是否已上传
        content_hash = get_file_content_hash(pdf_path)
        if is_pdf_uploaded(content_hash):
            print(f"\n✅ PDF「{os.path.basename(pdf_path)}」已上传（内容未变化），无需重复加载")
            return

        # 2. 加载PDF文本并分割
        print(f"\n🔄 正在处理PDF：{pdf_path}")
        pdf_text = load_pdf(pdf_path)
        docs = split_text(
            text=pdf_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            pdf_path=pdf_path
        )
        if not docs:
            raise ValueError(f"❌ PDF「{os.path.basename(pdf_path)}」无有效文本可处理")

        # 3. 生成向量并添加到全局集合（附带PDF元数据）
        print("🔄 正在生成文本向量并添加到全局库...")
        texts = [doc.page_content for doc in docs]
        # 补充每个chunk的PDF元数据（便于溯源）
        metadatas = []
        for doc in docs:
            meta = doc.metadata.copy()
            meta.update({
                "pdf_content_hash": content_hash,
                "pdf_name": os.path.basename(pdf_path),
                "pdf_path": pdf_path
            })
            metadatas.append(meta)
        ids = [f"{content_hash[:8]}_chunk_{i}" for i in range(len(docs))]  # 全局唯一ID

        # 生成嵌入向量（批量处理避免内存溢出）
        batch_size = 8 if self.embedding_model.device.type == "cpu" else 16
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)

        # 添加到向量库
        self.collection.add(
            embeddings=all_embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        # 4. 记录上传信息（自动去重）
        add_upload_record(pdf_path, content_hash)
        print(f"✅ PDF「{os.path.basename(pdf_path)}」上传成功！新增{len(docs)}个文本片段")
        print(f"   内容哈希：{content_hash[:8]}... | 存储路径：{GLOBAL_VECTOR_DB_PATH}")

    def batch_add_pdfs(self, pdf_paths: list):
        """批量添加多个PDF"""
        for pdf_path in pdf_paths:
            try:
                self.add_pdf(pdf_path)
            except Exception as e:
                print(f"\n❌ 处理PDF「{pdf_path}」失败：{str(e)}")

    def query_all(self, question: str, search_k: int = 8) -> str:
        """
        检索所有已上传PDF回答问题
        :param question: 用户问题
        :param search_k: 检索返回的相关片段数量
        :return: 生成的回答
        """
        if self.collection.count() == 0:
            return "❌ 暂无已上传的PDF数据，请先上传PDF后再提问！"

        # 生成问题向量
        print(f"\n🔍 正在检索所有PDF中与「{question}」相关的文本片段...")
        query_embedding = self.embedding_model.embed_query(question)

        # 全局检索相关片段
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_k,
            include=["documents", "metadatas", "distances"]
        )

        # 过滤低相关度结果（距离<1.0）
        filtered_docs = []
        for doc, distance in zip(results["documents"][0], results["distances"][0]):
            if distance < 1.0:
                filtered_docs.append(doc)
        if not filtered_docs:
            return "❌ 未检索到与问题相关的有效信息"

        # 构建Prompt（附带来源信息）
        context = ""
        for idx, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            context += f"\n【相关片段{idx+1}（来源：{meta['pdf_name']}）】\n{doc}\n"

        prompt = f"""请严格基于以下上下文信息回答问题，仅使用上下文里的内容，不要编造任何信息。
如果上下文没有相关信息，直接回答“未找到相关答案”，不要额外解释。

上下文：
{context}

问题：{question}
回答："""

        # 调用LLM生成回答
        print("🤖 正在生成回答...")
        response = self.llm_model.invoke(prompt)
        return response.content.strip()

    def clear_all(self):
        """清空全局向量库和上传记录（谨慎使用）"""
        from config import UPLOAD_RECORD_FILE
        import json

        confirm = input("\n⚠️ 确认清空所有PDF数据？输入「YES」确认：")
        if confirm == "YES":
            # 清空向量库
            self.vector_db.delete_collection("all_pdf_chunks")
            self.collection = self.vector_db.get_or_create_collection("all_pdf_chunks")
            # 清空上传记录
            with open(UPLOAD_RECORD_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
            print("✅ 已清空所有PDF数据和上传记录")
        else:
            print("🚫 取消清空操作")