"""RAG核心逻辑：多PDF向量库管理、全局检索、自动去重"""
import chromadb
import json
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
        self.collection_name = "all_pdf_chunks"  # 提取集合名称为变量
        self.collection = self.vector_db.get_or_create_collection(self.collection_name)

    # 【原有方法：add_pdf、batch_add_pdfs、query_all 保持不变】
    def add_pdf(self, pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """添加PDF到全局向量库（自动去重）"""
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
        """检索所有已上传PDF回答问题"""
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

    # 【新增：按内容哈希删除指定PDF】
    def delete_pdf_by_hash(self, content_hash: str):
        """
        按PDF内容哈希删除指定PDF的向量数据和上传记录
        :param content_hash: PDF的内容哈希值（get_file_content_hash生成）
        """
        if not content_hash:
            raise ValueError("❌ 必须提供PDF内容哈希值")

        try:
            # 1. 删除向量库中该PDF的所有chunk（按hash筛选）
            self.collection.delete(
                where={"pdf_content_hash": content_hash}  # 精准筛选该PDF的所有数据
            )
            print(f"✅ 已删除哈希为「{content_hash[:8]}...」的PDF向量数据")

            # 2. 删除上传记录中的该PDF信息
            upload_file = "pdf_uploads.json"
            if os.path.exists(upload_file):
                with open(upload_file, "r", encoding="utf-8") as f:
                    upload_records = json.load(f)
                # 过滤掉该hash的记录
                new_records = [r for r in upload_records if r.get("content_hash") != content_hash]
                with open(upload_file, "w", encoding="utf-8") as f:
                    json.dump(new_records, f, ensure_ascii=False, indent=2)
                print(f"✅ 已删除哈希为「{content_hash[:8]}...」的PDF上传记录")

            return f"✅ 成功删除哈希为「{content_hash[:8]}...」的PDF数据"

        except Exception as e:
            print(f"❌ 删除指定PDF失败：{str(e)}")
            raise

    # 【新增：按文件路径删除指定PDF（更易用）】
    def delete_pdf_by_path(self, pdf_path: str):
        """
        按PDF文件路径删除指定PDF（先计算hash，再调用delete_pdf_by_hash）
        :param pdf_path: PDF文件的绝对/相对路径
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ PDF文件不存在：{pdf_path}")

        # 计算该PDF的内容哈希
        content_hash = get_file_content_hash(pdf_path)
        # 检查是否已上传
        if not is_pdf_uploaded(content_hash):
            raise ValueError(f"❌ PDF「{os.path.basename(pdf_path)}」未上传，无需删除")

        # 调用按hash删除的方法
        return self.delete_pdf_by_hash(content_hash)

    # 【原有：全量删除所有PDF】
    def clear_all(self):
        """清空所有PDF向量库和上传记录（纯代码逻辑，无终端交互）"""
        try:
            # 1. 清空向量库：删除集合后重新创建（适配所有Chromadb版本）
            if hasattr(self, "vector_db") and self.vector_db:
                # 先删除现有集合
                self.vector_db.delete_collection(self.collection_name)
                # 重新创建空集合
                self.collection = self.vector_db.get_or_create_collection(self.collection_name)
                print("✅ 向量库已清空（删除并重建集合）")

            # 2. 清空上传记录文件（pdf_uploads.json）
            upload_file = "pdf_uploads.json"
            if os.path.exists(upload_file):
                with open(upload_file, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                print("✅ 上传记录已清空")

            # 3. 清空内存中的变量（兼容原有逻辑）
            if hasattr(self, "uploaded_pdfs"):
                self.uploaded_pdfs = []
            if hasattr(self, "content_hashes"):
                self.content_hashes = set()

            print("✅ 所有PDF数据（向量库+上传记录）已清空")
            return "✅ 所有PDF数据已清空"

        except Exception as e:
            print(f"❌ 清空失败：{str(e)}")
            raise