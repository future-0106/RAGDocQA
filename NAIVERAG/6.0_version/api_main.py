"""FastAPI版多PDF知识库系统：支持HTTP接口访问"""
import warnings
import os
import json
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# 导入核心模块
from config import (
    DEVICE, SEARCH_K,
    CHUNK_SIZE, CHUNK_OVERLAP, LOCAL_MODEL_PATH,
    ALL_MODELS, DEFAULT_MODEL
)
from rag_core import NaiveRAG
from embeddings import Qwen3Embeddings
from llm import ModelFactory  # 使用模型工厂
from utils import validate_config, get_uploaded_pdfs

warnings.filterwarnings("ignore")


embedding_model = None
llm_model = None
rag_system = None
current_model_name = DEFAULT_MODEL


# ========== 初始化FastAPI应用 ==========
app = FastAPI(
    title="多PDF智能检索API",
    description="支持PDF上传、查看、全局检索回答的API服务",
    version="1.0.0"
)

# ========== 全局变量 ==========
embedding_model = None
llm_model = None
rag_system = None

# ========== 初始化函数 ==========
def init_systems():
    """初始化所有系统组件"""
    global embedding_model, llm_model, rag_system

    # 验证配置（仅针对API模型，本地模型无需api_key）
    default_model_config = ALL_MODELS[DEFAULT_MODEL]
    if default_model_config["type"] == "api" and not default_model_config["params"].get("api_key"):
        raise ValueError("❌ 请在.env文件中配置DASHSCOPE_API_KEY")

    # 初始化嵌入模型
    embedding_model = Qwen3Embeddings(model_path=LOCAL_MODEL_PATH)

    # 初始化默认LLM模型
    llm_model = ModelFactory.create_model(default_model_config)

    # 初始化RAG系统
    rag_system = NaiveRAG(embedding_model, llm_model)

# 启动时初始化
init_systems()

# ========== 简单首页 ==========
@app.get("/", summary="首页")
async def home():
    """首页：返回API使用说明"""
    return JSONResponse({
        "title": "多PDF智能检索API",
        "version": "1.0.0",
        "docs_url": "http://127.0.0.1:8000/docs",
        "available_models": list(ALL_MODELS.keys()),
        "default_model": DEFAULT_MODEL,
        "endpoints": {
            "upload_pdf_by_path": "POST /upload_pdf_by_path - 通过本地路径上传PDF",
            "upload_pdf_by_file": "POST /upload_pdf_by_file - 通过文件上传PDF",
            "list_pdfs": "GET /list_pdfs - 查看已上传PDF列表",
            "query": "POST /query - 提问（检索所有PDF回答）",
            "delete_pdf": "POST /delete_pdf - 按路径删除指定PDF",
            "clear_all": "POST /clear_all - 清空所有PDF数据",
            "switch_model": "POST /switch_model - 切换LLM模型"
        }
    })

# ========== 新增：模型切换接口 ==========
@app.post("/switch_model", summary="切换LLM模型")
async def switch_model(model_name: str = Form(..., description=f"模型名称，可选值：{list(ALL_MODELS.keys())}")):
    global llm_model, rag_system, current_model_name  # 新增current_model_name
    if model_name not in ALL_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"无效的模型名称，可选模型：{list(ALL_MODELS.keys())}"
        )

    try:
        new_llm = ModelFactory.create_model(ALL_MODELS[model_name])
        llm_model = new_llm
        rag_system = NaiveRAG(embedding_model, llm_model)
        current_model_name = model_name  # 更新当前模型名称

        return JSONResponse({
            "code": 200,
            "message": f"成功切换至模型：{model_name}",
            "current_model": model_name
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"切换模型失败：{str(e)}")

# ========== 原有API接口完整实现 ==========
@app.post("/upload_pdf_by_path", summary="通过本地路径上传PDF")
async def upload_pdf_by_path(pdf_paths: str = Form(..., description="PDF路径，多个用英文逗号分隔")):
    """通过本地文件路径上传PDF（适合服务器本地文件）"""
    try:
        pdf_paths_list = [p.strip() for p in pdf_paths.split(",") if p.strip()]
        if not pdf_paths_list:
            raise HTTPException(status_code=400, detail="未输入有效PDF路径")

        results = []
        for pdf_path in pdf_paths_list:
            try:
                rag_system.add_pdf(pdf_path, CHUNK_SIZE, CHUNK_OVERLAP)
                results.append({
                    "pdf_path": pdf_path,
                    "status": "success",
                    "message": f"PDF「{os.path.basename(pdf_path)}」上传成功"
                })
            except Exception as e:
                results.append({
                    "pdf_path": pdf_path,
                    "status": "failed",
                    "message": str(e)
                })

        return JSONResponse({
            "code": 200,
            "message": "批量上传完成",
            "data": results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_pdf_by_file", summary="通过文件上传PDF")
async def upload_pdf_by_file(files: list[UploadFile] = File(..., description="PDF文件")):
    """通过表单上传PDF文件（适合远程上传）"""
    try:
        temp_dir = "./temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)

        results = []
        for file in files:
            if not file.filename.endswith(".pdf"):
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "message": "仅支持PDF格式文件"
                })
                continue

            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            try:
                rag_system.add_pdf(file_path, CHUNK_SIZE, CHUNK_OVERLAP)
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "message": f"PDF「{file.filename}」上传成功"
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "message": str(e)
                })

        return JSONResponse({
            "code": 200,
            "message": "文件上传处理完成",
            "data": results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_pdfs", summary="查看已上传PDF列表")
async def list_pdfs():
    """获取所有已上传PDF的详细信息"""
    try:
        uploaded_list = get_uploaded_pdfs()
        formatted_list = []
        for idx, item in enumerate(uploaded_list, 1):
            formatted_list.append({
                "index": idx,
                "pdf_name": item["pdf_name"],
                "pdf_path": item["pdf_path"],
                "file_size_kb": round(item["file_size"] / 1024, 2),
                "upload_time": item["upload_time"],
                "content_hash": item["content_hash"],
                "content_hash_short": item["content_hash"][:8]
            })

        return JSONResponse({
            "code": 200,
            "message": "获取成功",
            "data": formatted_list
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", summary="提问（检索所有PDF回答）")
async def query(question: str = Form(..., description="用户问题")):
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")

        if rag_system is None:
            raise HTTPException(status_code=500, detail="RAG系统未初始化，请重启服务")

        print(f"🔍 正在检索所有PDF中与「{question}」相关的文本片段...")
        answer = rag_system.query_all(question, search_k=SEARCH_K)

        if not answer or answer.strip() == "":
            answer = "未检索到与该问题相关的有效信息，请尝试调整问题描述或上传更多相关PDF。"

        print(f"🤖 回答生成完成：{answer[:50]}...")
        return JSONResponse({
            "code": 200,
            "message": "查询成功",
            "data": {
                "question": question,
                "answer": answer,
                "used_model": current_model_name  # 使用实际的当前模型名称
            }
        })
    except Exception as e:
        print(f"❌ 生成回答时出错：{str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误：{str(e)}")

@app.post("/delete_pdf", summary="按路径删除指定PDF")
async def delete_pdf(pdf_path: str = Form(..., description="PDF文件的绝对/相对路径")):
    """按路径删除指定PDF的向量数据和上传记录"""
    try:
        result = rag_system.delete_pdf_by_path(pdf_path)
        return JSONResponse({
            "code": 200,
            "message": result
        })
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败：{str(e)}")

@app.post("/clear_all", summary="清空所有PDF数据")
async def clear_all(confirm: str = Form(..., description="确认清空，输入YES")):
    """清空所有已上传PDF的向量库和上传记录"""
    try:
        if confirm != "YES":
            raise HTTPException(status_code=400, detail="必须输入YES确认清空")

        rag_system.clear_all()
        return JSONResponse({
            "code": 200,
            "message": "已清空所有PDF数据和上传记录"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== 启动服务 ==========
if __name__ == "__main__":
    print("="*60)
    print("🎯 多PDF智能检索API服务启动中...")
    print(f"📌 访问地址：http://127.0.0.1:8000")
    print(f"📚 API文档：http://127.0.0.1:8000/docs")
    print(f"🔧 当前使用模型：{DEFAULT_MODEL}")
    print(f"🗂️ 可用模型列表：{list(ALL_MODELS.keys())}")
    print("="*60)

    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # 生产环境建议关闭reload
    )