"""
FastAPI封装：将统一模型管理RAG系统封装为RESTful API
支持混合检索动态配置与查询改写（基于权重自适应）
"""
import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 导入现有模块
from config import *
from models import MultiModelLLM, MultiEmbeddings, ModelFactory
from documents import DocumentProcessor
from vector_store import ChromaDBManager, FileVectorizationManager, HybridRetrievalManager
from rag_pipeline import QwenRAGPipeline, setup_environment, check_imports
from models import MultiReranker
from config import RETRIEVAL_MODE, HYBRID_WEIGHTS, RERANKER_ENABLED, RERANKER_TOP_K
from config import QUERY_REWRITING_MODEL  # 新增导入

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".txt", ".md"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


# ==================== 初始化 ====================

class RAGAPISystem:
    """FastAPI封装的主系统"""

    def __init__(self):
        """初始化系统"""
        setup_environment()
        if not check_imports():
            print("❌ 系统初始化失败，缺少必要的依赖")
            raise RuntimeError("缺少必要的依赖")

        print("=" * 60)
        print("🚀 统一模型管理RAG系统 - FastAPI版本初始化中...")
        print("=" * 60)

        self._create_dirs()
        self._init_components()

        self.query_history = []
        self.max_history = 100

        print("✅ 系统初始化完成！")

    def _create_dirs(self):
        """创建必要的目录"""
        for dir_path in [DATA_DIR, CHROMA_DB_DIR, MODELS_DIR]:
            dir_path.mkdir(exist_ok=True)

        self.upload_dir = BASE_DIR / "uploads"
        self.upload_dir.mkdir(exist_ok=True)
        self.static_dir = BASE_DIR / "static"
        self.static_dir.mkdir(exist_ok=True)
        self.logs_dir = BASE_DIR / "logs"
        self.logs_dir.mkdir(exist_ok=True)

    def _init_components(self):
        """初始化各个组件（包含改写专用模型）"""
        try:
            # 1. 初始化嵌入模型
            print("🔧 初始化嵌入模型...")
            self.embeddings = MultiEmbeddings()

            # 2. 初始化向量存储管理器
            print("🔧 初始化向量存储管理器...")
            self.vector_manager = ChromaDBManager(
                embedding_model=self.embeddings,
                persist_directory=str(CHROMA_DB_DIR)
            )

            # 3. 加载现有的向量存储
            print("🔧 加载向量存储...")
            self.vector_manager.load()

            # 4. 初始化重排模型
            print("🔧 初始化重排模型...")
            self.reranker = MultiReranker()

            # 5. 初始化混合检索管理器
            print("🔧 初始化混合检索管理器...")
            self.hybrid_manager = HybridRetrievalManager(
                chroma_manager=self.vector_manager,
                reranker_model=self.reranker,
                retrieval_mode=RETRIEVAL_MODE,
                hybrid_weights=HYBRID_WEIGHTS,
                reranker_enabled=RERANKER_ENABLED,
                reranker_top_k=RERANKER_TOP_K
            )

            # 6. 初始化大语言模型（问答用）
            print("🔧 初始化大语言模型...")
            self.llm = MultiModelLLM()

            # ========== 新增：初始化改写专用LLM ==========
            if QUERY_REWRITING_MODEL:
                try:
                    print(f"🔧 初始化查询改写专用模型: {QUERY_REWRITING_MODEL}")
                    self.rewrite_llm = ModelFactory.create_llm(QUERY_REWRITING_MODEL)
                except Exception as e:
                    print(f"⚠️ 改写专用模型加载失败，将复用问答模型: {e}")
                    self.rewrite_llm = None
            else:
                self.rewrite_llm = None
            # ===========================================

            # 7. 初始化文档处理器
            print("🔧 初始化文档处理器...")
            self.document_processor = DocumentProcessor(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )

            # 8. 初始化文件管理器
            print("🔧 初始化文件管理器...")
            self.file_manager = FileVectorizationManager(
                data_dir=DATA_DIR,
                hybrid_manager=self.hybrid_manager
            )

            # 9. 初始化RAG流水线（传入改写专用LLM）
            print("🔧 初始化RAG流水线...")
            self.rag_pipeline = QwenRAGPipeline(
                self.llm,
                self.hybrid_manager,
                rewrite_llm=self.rewrite_llm
            )

        except Exception as e:
            print(f"❌ 组件初始化失败: {e}")
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态（保持原有实现，可酌情添加改写状态）"""
        try:
            llm_info = {
                "key": self.llm.model_key,
                "type": self.llm._llm_type,
                "identifying_params": self._serialize_params(self.llm._identifying_params)
            }
            model_config = ALL_MODELS.get(self.llm.model_key, {})
            llm_info.update({
                "provider": model_config.get("provider", "unknown"),
                "description": model_config.get("description", "")
            })

            status = {
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "models": {
                    "llm": self.llm.model_key,
                    "llm_info": llm_info,
                    "embedding": self.embeddings.model_key,
                    "embedding_info": {
                        "key": self.embeddings.model_key,
                        "type": "embedding_model"
                    },
                    "rewrite_llm": QUERY_REWRITING_MODEL if QUERY_REWRITING_MODEL else "复用问答模型",
                    "rewrite_enabled": QUERY_REWRITING_ENABLED
                },
                "device": str(DEVICE),
                "config": {
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "similarity_top_k": SIMILARITY_TOP_K,
                    "score_threshold": SCORE_THRESHOLD,
                    "max_context_length": MAX_CONTEXT_LENGTH
                }
            }

            stats = self.vector_manager.get_collection_stats()
            status["vector_store"] = stats if "error" not in stats else {"status": "not_loaded"}

            files = self.file_manager.list_data_files()
            file_info = self.file_manager.get_file_info()
            status["files"] = {
                "count": len(files),
                "list": files,
                "details": file_info
            }

            status["history"] = {
                "count": len(self.query_history),
                "max": self.max_history
            }

            return status
        except Exception as e:
            print(f"❌ 获取系统状态失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _serialize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """序列化参数（保持不变）"""
        serialized = {}
        for key, value in params.items():
            if isinstance(value, (torch.device, torch.dtype)):
                serialized[key] = str(value)
            elif isinstance(value, Path):
                serialized[key] = str(value)
            elif hasattr(value, '__dict__'):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized


# 创建全局系统实例
rag_system = RAGAPISystem()


# ==================== Pydantic模型定义 ====================

class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str = Field(..., description="要查询的问题")
    k: int = Field(default=SIMILARITY_TOP_K, description="返回的最相关文档数量")
    score_threshold: float = Field(default=SCORE_THRESHOLD, description="相似度阈值")
    include_sources: bool = Field(default=True, description="是否包含来源")
    enable_rewriting: Optional[bool] = Field(
        default=None,
        description="临时开启/关闭查询改写（None表示使用全局配置）"
    )


class ModelSwitchRequest(BaseModel):
    """模型切换请求模型"""
    model_key: str = Field(..., description="要切换到的模型键值")


class UploadResponse(BaseModel):
    """上传响应模型"""
    success: bool
    message: str
    filename: str
    file_size: int
    document_count: int
    processing_time: float


class BatchUploadRequest(BaseModel):
    """批量上传请求模型"""
    file_paths: List[str] = Field(..., description="要上传的文件路径列表")


class RetrievalConfigRequest(BaseModel):
    """检索配置请求模型"""
    retrieval_mode: Optional[str] = Field(default=None, description="检索模式: vector, bm25, hybrid")
    bm25_weight: Optional[float] = Field(default=None, description="BM25权重 (0-1)")
    vector_weight: Optional[float] = Field(default=None, description="向量权重 (0-1)")
    reranker_enabled: Optional[bool] = Field(default=None, description="是否启用重排")
    reranker_top_k: Optional[int] = Field(default=None, description="重排后返回数量")


# ==================== FastAPI应用 ====================

app = FastAPI(
    title="统一模型管理RAG系统API",
    description="基于FastAPI封装的统一模型管理RAG系统（支持混合检索与查询改写）",
    version="1.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:9000", "http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=rag_system.static_dir), name="static")


# ==================== API端点 ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")


@app.get("/api/status")
async def get_status():
    try:
        status = rag_system.get_system_status()
        return JSONResponse(content={"success": True, "data": status})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/query")
async def query_rag(request: QueryRequest):
    import time
    start_time = time.time()

    try:
        rag_system.query_history.append({
            "question": request.question,
            "timestamp": datetime.now().isoformat(),
            "model": rag_system.llm.model_key
        })
        if len(rag_system.query_history) > rag_system.max_history:
            rag_system.query_history = rag_system.query_history[-rag_system.max_history:]

        result = rag_system.rag_pipeline.query(
            request.question,
            k=request.k,
            score_threshold=request.score_threshold,
            enable_rewriting=request.enable_rewriting
        )

        processing_time = time.time() - start_time

        response_data = {
            "success": True,
            "question": result["question"],
            "answer": result["answer"],
            "model_used": rag_system.llm.model_key,
            "processing_time": processing_time,
            "sources": result["sources"] if request.include_sources else [],
            "context_length": result["context_length"],
            "source_count": result["source_count"],
            "retrieval_mode": result["retrieval_mode"],
            "reranker_enabled": result["reranker_enabled"],
            "hybrid_weights": result["hybrid_weights"],
            "rewritten_queries": result.get("rewritten_queries", []),  # 调试信息
            "timestamp": datetime.now().isoformat()
        }
        return JSONResponse(content=response_data)

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
        )


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    import time
    start_time = time.time()

    try:
        # 验证文件扩展名
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"不支持的文件类型: {file_ext}。允许的类型: {', '.join(ALLOWED_EXTENSIONS)}",
                    "filename": file.filename
                }
            )

        # 验证文件大小
        if file.size and file.size > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"文件大小超过限制: {file.size / 1024 / 1024:.1f}MB，最大允许: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB",
                    "filename": file.filename
                }
            )

        temp_file_path = rag_system.upload_dir / file.filename
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        success, message, documents = rag_system.file_manager.upload_and_vectorize(
            str(temp_file_path),
            rag_system.document_processor,
            rag_system.vector_manager,
            rag_system.hybrid_manager
        )

        temp_file_path.unlink()
        processing_time = time.time() - start_time

        if success:
            response = UploadResponse(
                success=True,
                message=message,
                filename=file.filename,
                file_size=file.size,
                document_count=len(documents),
                processing_time=processing_time
            )
            return JSONResponse(content=response.model_dump())
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": message,
                    "filename": file.filename,
                    "processing_time": processing_time
                }
            )
    except Exception as e:
        processing_time = time.time() - start_time
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "filename": file.filename if 'file' in locals() else "unknown",
                "processing_time": processing_time
            }
        )


@app.post("/api/batch-upload")
async def batch_upload(request: BatchUploadRequest):
    import time
    start_time = time.time()

    try:
        results = rag_system.file_manager.upload_multiple_files(
            request.file_paths,
            rag_system.document_processor,
            rag_system.vector_manager,
            rag_system.hybrid_manager
        )

        processing_time = time.time() - start_time

        return JSONResponse(content={
            "success": True,
            "processing_time": processing_time,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        processing_time = time.time() - start_time
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/api/files")
async def get_files():
    try:
        files = rag_system.file_manager.list_data_files()
        file_info = rag_system.file_manager.get_file_info()
        return JSONResponse(content={
            "success": True,
            "count": len(files),
            "files": files,
            "details": file_info,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    try:
        success = rag_system.file_manager.delete_data_file(filename)
        if success:
            return JSONResponse(content={
                "success": True,
                "message": f"文件 '{filename}' 已成功删除",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"删除文件 '{filename}' 失败",
                    "timestamp": datetime.now().isoformat()
                }
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")


@app.get("/api/models")
async def get_models():
    try:
        llm_models = ModelFactory.list_available_models()
        embedding_models = ModelFactory.list_available_embedding_models()

        current_llm = rag_system.llm.model_key
        current_embedding = rag_system.embeddings.model_key

        processed_llm_models = []
        for model in llm_models:
            model_copy = dict(model)
            model_copy["is_current"] = model_copy["key"] == current_llm
            if "params" in model_copy:
                params = model_copy["params"].copy()
                for key, value in params.items():
                    if isinstance(value, (torch.device, torch.dtype, Path)):
                        params[key] = str(value)
                model_copy["params"] = params
            processed_llm_models.append(model_copy)

        processed_embedding_models = []
        for model in embedding_models:
            model_copy = dict(model)
            model_copy["is_current"] = model_copy["key"] == current_embedding
            if "params" in model_copy:
                params = model_copy["params"].copy()
                for key, value in params.items():
                    if isinstance(value, (torch.device, torch.dtype, Path)):
                        params[key] = str(value)
                model_copy["params"] = params
            processed_embedding_models.append(model_copy)

        return JSONResponse(content={
            "success": True,
            "llm_models": processed_llm_models,
            "embedding_models": processed_embedding_models,
            "current_llm": current_llm,
            "current_embedding": current_embedding,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@app.post("/api/switch-model")
async def switch_model(request: ModelSwitchRequest):
    try:
        old_model = rag_system.llm.model_key
        rag_system.llm.switch_model(request.model_key)
        rag_system.rag_pipeline = QwenRAGPipeline(rag_system.llm, rag_system.hybrid_manager, rag_system.rewrite_llm)
        return JSONResponse(content={
            "success": True,
            "message": f"模型已从 '{old_model}' 切换到 '{request.model_key}'",
            "old_model": old_model,
            "new_model": request.model_key,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.post("/api/switch-embedding")
async def switch_embedding(request: ModelSwitchRequest):
    try:
        old_model = rag_system.embeddings.model_key
        rag_system.embeddings.switch_model(request.model_key)

        rag_system.vector_manager = ChromaDBManager(
            embedding_model=rag_system.embeddings,
            persist_directory=str(CHROMA_DB_DIR)
        )
        rag_system.vector_manager.load()

        rag_system.hybrid_manager = HybridRetrievalManager(
            chroma_manager=rag_system.vector_manager,
            reranker_model=rag_system.reranker,
            retrieval_mode=RETRIEVAL_MODE,
            hybrid_weights=HYBRID_WEIGHTS,
            reranker_enabled=RERANKER_ENABLED,
            reranker_top_k=RERANKER_TOP_K
        )

        rag_system.file_manager.hybrid_manager = rag_system.hybrid_manager
        rag_system.rag_pipeline = QwenRAGPipeline(rag_system.llm, rag_system.hybrid_manager, rag_system.rewrite_llm)

        return JSONResponse(content={
            "success": True,
            "message": f"嵌入模型已从 '{old_model}' 切换到 '{request.model_key}'，混合检索已重新初始化",
            "old_model": old_model,
            "new_model": request.model_key,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.post("/api/update-retrieval-config")
async def update_retrieval_config(request: RetrievalConfigRequest):
    try:
        if request.retrieval_mode is not None:
            if request.retrieval_mode not in ["vector", "bm25", "hybrid"]:
                raise HTTPException(status_code=400, detail="检索模式必须是 vector, bm25 或 hybrid")
            rag_system.rag_pipeline.update_retrieval_config(retrieval_mode=request.retrieval_mode)

        if request.bm25_weight is not None and request.vector_weight is not None:
            if not (0 <= request.bm25_weight <= 1 and 0 <= request.vector_weight <= 1):
                raise HTTPException(status_code=400, detail="权重必须在0-1之间")
            if abs(request.bm25_weight + request.vector_weight - 1.0) > 0.01:
                raise HTTPException(status_code=400, detail="权重之和必须为1")
            rag_system.rag_pipeline.update_retrieval_config(
                hybrid_weights=(request.bm25_weight, request.vector_weight)
            )

        if request.reranker_enabled is not None:
            rag_system.rag_pipeline.update_retrieval_config(reranker_enabled=request.reranker_enabled)

        if request.reranker_top_k is not None:
            if request.reranker_top_k <= 0:
                raise HTTPException(status_code=400, detail="重排返回数量必须大于0")
            rag_system.rag_pipeline.update_retrieval_config(reranker_top_k=request.reranker_top_k)

        retrieval_info = rag_system.rag_pipeline.get_retrieval_info()
        return JSONResponse(content={
            "success": True,
            "message": "检索配置已更新",
            "config": retrieval_info,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/api/retrieval-config")
async def get_retrieval_config():
    try:
        retrieval_info = rag_system.rag_pipeline.get_retrieval_info()
        return JSONResponse(content={
            "success": True,
            "config": retrieval_info,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取检索配置失败: {str(e)}")


@app.post("/api/reprocess")
async def reprocess_files():
    try:
        files = rag_system.file_manager.list_data_files()
        if not files:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "数据目录为空，无需重新处理",
                    "timestamp": datetime.now().isoformat()
                }
            )

        rag_system.vector_manager.vector_store = None
        total_documents = 0
        all_document_texts = []

        for i, filename in enumerate(files, 1):
            file_path = DATA_DIR / filename
            print(f"[{i}/{len(files)}] 处理文件: {filename}")
            documents = rag_system.document_processor.process_file(str(file_path))

            if documents:
                if rag_system.vector_manager.vector_store:
                    doc_texts = rag_system.vector_manager.add_documents(documents)
                else:
                    doc_texts = rag_system.vector_manager.create_from_documents(documents)

                all_document_texts.extend(doc_texts)
                total_documents += len(documents)

        rag_system.vector_manager.load()

        if all_document_texts:
            rag_system.hybrid_manager.bm25_retriever.update_documents(all_document_texts)

        return JSONResponse(content={
            "success": True,
            "message": f"所有文件重新处理完成！总共处理了 {len(files)} 个文件，添加了 {total_documents} 个文档块，BM25索引已同步",
            "files_processed": len(files),
            "documents_added": total_documents,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"重新处理文件失败: {str(e)}")


@app.get("/api/history")
async def get_history(limit: int = 20):
    try:
        history = rag_system.query_history[-limit:] if limit > 0 else rag_system.query_history
        return JSONResponse(content={
            "success": True,
            "count": len(history),
            "total": len(rag_system.query_history),
            "max": rag_system.max_history,
            "history": history,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


@app.delete("/api/history")
async def clear_history():
    try:
        count = len(rag_system.query_history)
        rag_system.query_history.clear()
        return JSONResponse(content={
            "success": True,
            "message": f"已清空查询历史，共清除了 {count} 条记录",
            "cleared_count": count,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空历史记录失败: {str(e)}")


@app.get("/api/health")
async def health_check():
    try:
        status = rag_system.get_system_status()
        return JSONResponse(content={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": "统一模型管理RAG系统API",
            "version": "1.1.0",
            "components": {
                "llm": "active" if rag_system.llm else "inactive",
                "embeddings": "active" if rag_system.embeddings else "inactive",
                "vector_store": "active" if rag_system.vector_manager.vector_store else "inactive",
                "rag_pipeline": "active" if rag_system.rag_pipeline else "inactive"
            }
        })
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/frontend", response_class=HTMLResponse)
async def frontend_page():
    return FileResponse("static/index.html")


@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    if full_path.startswith("api/") or full_path.startswith("static/"):
        raise HTTPException(status_code=404, detail="页面未找到")
    return FileResponse("static/index.html")


# ==================== 启动脚本 ====================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("🚀 统一模型管理RAG系统API服务启动中...")
    print("=" * 60)

    print(f"\n📊 API服务器信息:")
    print(f"  地址: http://127.0.0.1:9000")
    print(f"  前端界面: http://127.0.0.1:9000/")
    print(f"  文档: http://127.0.0.1:9000/api/docs")
    print(f"  状态: http://127.0.0.1:9000/api/status")
    print(f"  健康检查: http://127.0.0.1:9000/api/health")

    try:
        status = rag_system.get_system_status()
        print(f"\n📊 系统信息:")
        print(f"  当前LLM模型: {status['models']['llm']}")
        print(f"  当前嵌入模型: {status['models']['embedding']}")
        print(f"  改写专用模型: {status['models'].get('rewrite_llm', '未配置')}")
        print(f"  改写功能开关: {status['models'].get('rewrite_enabled', False)}")
        print(f"  向量存储文档数: {status['vector_store'].get('document_count', '未知')}")
        print(f"  数据文件数: {status['files']['count']}")
    except Exception as e:
        print(f"⚠️ 无法获取完整状态: {e}")

    print("\n" + "=" * 60)
    print("✅ API服务已启动！按 Ctrl+C 停止服务")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=9000)