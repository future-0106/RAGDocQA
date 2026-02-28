"""
FastAPI封装：将统一模型管理RAG系统封装为RESTful API
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
from vector_store import ChromaDBManager, FileVectorizationManager
from rag_pipeline import QwenRAGPipeline, setup_environment, check_imports
# 在导入部分添加
from models import MultiReranker
from vector_store import HybridRetrievalManager
from config import RETRIEVAL_MODE, HYBRID_WEIGHTS, RERANKER_ENABLED, RERANKER_TOP_K
# 在文件顶部的导入部分添加
from typing import Optional  # 如果还没有导入的话
# ==================== 初始化 ====================

class RAGAPISystem:
    """FastAPI封装的主系统"""

    def __init__(self):
        """初始化系统"""
        # 设置环境
        setup_environment()

        # 检查必要的导入
        if not check_imports():
            print("❌ 系统初始化失败，缺少必要的依赖")
            raise RuntimeError("缺少必要的依赖")

        print("=" * 60)
        print("🚀 统一模型管理RAG系统 - FastAPI版本初始化中...")
        print("=" * 60)

        # 创建必要的目录
        self._create_dirs()

        # 初始化组件
        self._init_components()

        # 初始化历史记录
        self.query_history = []
        self.max_history = 100

        print("✅ 系统初始化完成！")

    def _create_dirs(self):
        """创建必要的目录"""
        for dir_path in [DATA_DIR, CHROMA_DB_DIR, MODELS_DIR]:
            dir_path.mkdir(exist_ok=True)

        # 创建上传临时目录
        self.upload_dir = BASE_DIR / "uploads"
        self.upload_dir.mkdir(exist_ok=True)

        # 创建静态文件目录
        self.static_dir = BASE_DIR / "static"
        self.static_dir.mkdir(exist_ok=True)

        # 创建日志目录
        self.logs_dir = BASE_DIR / "logs"
        self.logs_dir.mkdir(exist_ok=True)

    def _init_components(self):
        """初始化各个组件"""
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

            # 6. 初始化LLM
            print("🔧 初始化大语言模型...")
            self.llm = MultiModelLLM()

            # 7. 初始化文档处理器
            print("🔧 初始化文档处理器...")
            self.document_processor = DocumentProcessor(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )

            # 8. 初始化文件管理器
            print("🔧 初始化文件管理器...")
            self.file_manager = FileVectorizationManager()

            # 9. 初始化RAG流水线
            print("🔧 初始化RAG流水线...")
            self.rag_pipeline = QwenRAGPipeline(self.llm, self.hybrid_manager)

        except Exception as e:
            print(f"❌ 组件初始化失败: {e}")
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            # 获取当前实际的模型信息
            llm_info = {
                "key": self.llm.model_key,
                "type": self.llm._llm_type,
                "identifying_params": self.llm._identifying_params
            }

            # 从配置中获取更多模型信息
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
                    }
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

            # 添加向量存储状态
            stats = self.vector_manager.get_collection_stats()
            if "error" not in stats:
                status["vector_store"] = stats
            else:
                status["vector_store"] = {"status": "not_loaded"}

            # 添加文件状态
            files = self.file_manager.list_data_files()
            file_info = self.file_manager.get_file_info()

            status["files"] = {
                "count": len(files),
                "list": files,
                "details": file_info
            }

            # 添加历史记录状态
            status["history"] = {
                "count": len(self.query_history),
                "max": self.max_history
            }

            print(f"📊 系统状态查询 - 当前模型: {self.llm.model_key}")

            return status

        except Exception as e:
            print(f"❌ 获取系统状态失败: {e}")
            import traceback
            traceback.print_exc()
            raise


# 创建全局系统实例
rag_system = RAGAPISystem()


# ==================== Pydantic模型定义 ====================

class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str = Field(..., description="要查询的问题")
    k: int = Field(default=SIMILARITY_TOP_K, description="返回的最相关文档数量")
    score_threshold: float = Field(default=SCORE_THRESHOLD, description="相似度阈值")
    include_sources: bool = Field(default=True, description="是否包含来源")


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


class QueryResponse(BaseModel):
    """查询响应模型"""
    success: bool
    question: str
    answer: str
    model_used: str
    processing_time: float
    sources: List[Dict[str, Any]]
    context_length: int
    source_count: int
    timestamp: str


class BatchUploadRequest(BaseModel):
    """批量上传请求模型"""
    file_paths: List[str] = Field(..., description="要上传的文件路径列表")


class ModelInfo(BaseModel):
    """模型信息模型"""
    key: str
    type: str
    provider: str
    description: str
    params: Dict[str, Any]

# ============ 新增：检索配置相关模型 ============

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
    description="基于FastAPI封装的统一模型管理RAG系统",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=rag_system.static_dir), name="static")


# ==================== API端点 ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """根端点，返回前端页面"""
    return FileResponse("static/index.html")


@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    try:
        status = rag_system.get_system_status()
        return JSONResponse(content={"success": True, "data": status})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


@app.post("/api/query")
async def query_rag(request: QueryRequest):
    """提交查询问题"""
    import time

    start_time = time.time()

    try:
        # 记录查询
        rag_system.query_history.append({
            "question": request.question,
            "timestamp": datetime.now().isoformat(),
            "model": rag_system.llm.model_key  # 使用当前模型
        })

        # 限制历史记录数量
        if len(rag_system.query_history) > rag_system.max_history:
            rag_system.query_history = rag_system.query_history[-rag_system.max_history:]

        # 直接使用当前模型的RAG流水线执行查询
        result = rag_system.rag_pipeline.query(
            request.question,
            k=request.k,
            score_threshold=request.score_threshold
        )

        # 计算处理时间
        processing_time = time.time() - start_time

        # 构建响应
        response_data = {
            "success": True,
            "question": result["question"],
            "answer": result["answer"],
            "model_used": rag_system.llm.model_key,  # 使用当前模型名称
            "processing_time": processing_time,
            "sources": result["sources"] if request.include_sources else [],
            "context_length": result["context_length"],
            "source_count": result["source_count"],
            "timestamp": datetime.now().isoformat()
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ 查询失败: {e}")
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
    """上传单个文件并向量化"""
    import time

    start_time = time.time()

    try:
        # 保存上传的文件到临时位置
        temp_file_path = rag_system.upload_dir / file.filename

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 处理文件
        success, message, documents = rag_system.file_manager.upload_and_vectorize(
            str(temp_file_path),
            rag_system.document_processor,
            rag_system.vector_manager
        )

        # 删除临时文件
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
    """批量上传文件"""
    import time

    start_time = time.time()

    try:
        results = rag_system.file_manager.upload_multiple_files(
            request.file_paths,
            rag_system.document_processor,
            rag_system.vector_manager
        )

        processing_time = time.time() - start_time

        response = {
            "success": True,
            "processing_time": processing_time,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

        return JSONResponse(content=response)

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
    """获取已上传文件列表"""
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
    """删除文件"""
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
    """获取可用模型列表"""
    try:
        llm_models = ModelFactory.list_available_models()
        embedding_models = ModelFactory.list_available_embedding_models()

        # 添加当前使用的模型标记
        current_llm = rag_system.llm.model_key
        current_embedding = rag_system.embeddings.model_key

        # 处理模型数据，确保可序列化
        processed_llm_models = []
        for model in llm_models:
            # 创建新的字典，避免修改原始数据
            model_copy = dict(model)
            model_copy["is_current"] = model_copy["key"] == current_llm

            # 处理params中的不可序列化对象
            if "params" in model_copy:
                params = model_copy["params"].copy()
                # 将device对象转换为字符串
                if "device" in params:
                    params["device"] = str(params["device"])
                # 处理其他可能的不可序列化对象
                for key, value in params.items():
                    if isinstance(value, (torch.device, torch.dtype)):
                        params[key] = str(value)
                model_copy["params"] = params

            processed_llm_models.append(model_copy)

        processed_embedding_models = []
        for model in embedding_models:
            # 创建新的字典，避免修改原始数据
            model_copy = dict(model)
            model_copy["is_current"] = model_copy["key"] == current_embedding

            # 处理params中的不可序列化对象
            if "params" in model_copy:
                params = model_copy["params"].copy()
                # 将device对象转换为字符串
                if "device" in params:
                    params["device"] = str(params["device"])
                # 处理其他可能的不可序列化对象
                for key, value in params.items():
                    if isinstance(value, (torch.device, torch.dtype)):
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
        print(f"❌ 获取模型列表失败，详细错误:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@app.post("/api/switch-model")
async def switch_model(request: ModelSwitchRequest):
    """切换LLM模型"""
    try:
        old_model = rag_system.llm.model_key

        print(f"🔄 接收到模型切换请求: {old_model} -> {request.model_key}")
        print(f"📋 当前模型参数: {rag_system.llm._identifying_params}")

        # 切换模型
        rag_system.llm.switch_model(request.model_key)

        print(f"✅ 模型切换完成")
        print(f"📋 新模型参数: {rag_system.llm._identifying_params}")

        # 重新初始化RAG流水线
        rag_system.rag_pipeline = QwenRAGPipeline(rag_system.llm, rag_system.vector_manager)

        return JSONResponse(content={
            "success": True,
            "message": f"模型已从 '{old_model}' 切换到 '{request.model_key}'",
            "old_model": old_model,
            "new_model": request.model_key,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ 模型切换失败: {e}")
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
    """切换嵌入模型"""
    try:
        old_model = rag_system.embeddings.model_key

        rag_system.embeddings.switch_model(request.model_key)

        # 需要重新初始化向量存储管理器
        rag_system.vector_manager = ChromaDBManager(
            embedding_model=rag_system.embeddings,
            persist_directory=str(CHROMA_DB_DIR)
        )
        rag_system.vector_manager.load()

        # 重新初始化RAG流水线
        rag_system.rag_pipeline = QwenRAGPipeline(rag_system.llm, rag_system.vector_manager)

        return JSONResponse(content={
            "success": True,
            "message": f"嵌入模型已从 '{old_model}' 切换到 '{request.model_key}'",
            "old_model": old_model,
            "new_model": request.model_key,
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


# ============ 新增：检索配置管理端点 ============

@app.post("/api/update-retrieval-config")
async def update_retrieval_config(request: RetrievalConfigRequest):
    """更新检索配置"""
    try:
        # 更新检索模式
        if request.retrieval_mode is not None:
            if request.retrieval_mode not in ["vector", "bm25", "hybrid"]:
                raise HTTPException(status_code=400, detail="检索模式必须是 vector, bm25 或 hybrid")

            rag_system.rag_pipeline.update_retrieval_config(
                retrieval_mode=request.retrieval_mode
            )

        # 更新混合权重
        if request.bm25_weight is not None and request.vector_weight is not None:
            if not (0 <= request.bm25_weight <= 1 and 0 <= request.vector_weight <= 1):
                raise HTTPException(status_code=400, detail="权重必须在0-1之间")

            if abs(request.bm25_weight + request.vector_weight - 1.0) > 0.01:
                raise HTTPException(status_code=400, detail="权重之和必须为1")

            rag_system.rag_pipeline.update_retrieval_config(
                hybrid_weights=(request.bm25_weight, request.vector_weight)
            )

        # 更新重排设置
        if request.reranker_enabled is not None:
            rag_system.rag_pipeline.update_retrieval_config(
                reranker_enabled=request.reranker_enabled
            )

        if request.reranker_top_k is not None:
            if request.reranker_top_k <= 0:
                raise HTTPException(status_code=400, detail="重排返回数量必须大于0")

            rag_system.rag_pipeline.update_retrieval_config(
                reranker_top_k=request.reranker_top_k
            )

        # 获取更新后的配置
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
    """获取当前检索配置"""
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
    """重新处理所有文件并重建向量存储"""
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

        print(f"🔄 开始重新处理所有 {len(files)} 个文件...")

        # 清空现有向量存储
        rag_system.vector_manager.vector_store = None

        total_documents = 0
        for i, filename in enumerate(files, 1):
            file_path = DATA_DIR / filename
            print(f"[{i}/{len(files)}] 处理文件: {filename}")

            documents = rag_system.document_processor.process_file(str(file_path))

            if documents:
                if rag_system.vector_manager.vector_store:
                    rag_system.vector_manager.add_documents(documents)
                else:
                    rag_system.vector_manager.create_from_documents(documents)

                total_documents += len(documents)
                print(f"✅ 处理完成，添加 {len(documents)} 个文档块")
            else:
                print(f"⚠️  文件处理失败或为空")

        # 重新加载向量存储
        rag_system.vector_manager.load()

        return JSONResponse(content={
            "success": True,
            "message": f"所有文件重新处理完成！总共处理了 {len(files)} 个文件，添加了 {total_documents} 个文档块",
            "files_processed": len(files),
            "documents_added": total_documents,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新处理文件失败: {str(e)}")


@app.get("/api/history")
async def get_history(limit: int = 20):
    """获取查询历史"""
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
    """清空查询历史"""
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


# ==================== 健康检查端点 ====================

@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    try:
        # 简单的健康检查
        status = rag_system.get_system_status()

        return JSONResponse(content={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": "统一模型管理RAG系统API",
            "version": "1.0.0",
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


# ==================== 前端页面路由 ====================

@app.get("/frontend", response_class=HTMLResponse)
async def frontend_page():
    """前端界面页面 - 重定向到根路径"""
    return FileResponse("static/index.html")


# 为所有未匹配的路由返回前端页面（支持前端路由）
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    """捕获所有未匹配的路由并返回前端页面"""
    # 检查是否是API路径
    if full_path.startswith("api/") or full_path.startswith("static/"):
        # 返回404错误，因为这些路径应该由其他路由处理
        raise HTTPException(status_code=404, detail="页面未找到")

    # 否则返回前端页面
    return FileResponse("static/index.html")


# ==================== 启动脚本 ====================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("🚀 统一模型管理RAG系统API服务启动中...")
    print("=" * 60)

    # 显示API信息
    print(f"\n📊 API服务器信息:")
    print(f"  地址: http://127.0.0.1:9000")
    print(f"  前端界面: http://127.0.0.1:9000/")
    print(f"  文档: http://127.0.0.1:9000/api/docs")
    print(f"  状态: http://127.0.0.1:9000/api/status")
    print(f"  健康检查: http://127.0.0.1:9000/api/health")

    # 显示系统信息
    status = rag_system.get_system_status()
    print(f"\n📊 系统信息:")
    print(f"  当前LLM模型: {status['models']['llm']}")
    print(f"  当前嵌入模型: {status['models']['embedding']}")
    print(f"  向量存储文档数: {status['vector_store'].get('document_count', '未知')}")
    print(f"  数据文件数: {status['files']['count']}")

    print("\n" + "=" * 60)
    print("✅ API服务已启动！按 Ctrl+C 停止服务")
    print("=" * 60 + "\n")

    # 启动服务器，端口改为9000
    uvicorn.run(app, host="127.0.0.1", port=9000)