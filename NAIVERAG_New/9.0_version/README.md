# 统一模型管理 RAG 系统

基于 FastAPI 封装的检索增强生成（RAG）系统，支持混合检索、查询改写、多模型管理和动态配置。

## 功能特性

- **混合检索**：支持向量检索、BM25 检索及其混合模式
- **查询改写**：基于 LLM 的查询扩展、假设文档生成（HyDE）、多查询生成
- **多模型管理**：支持本地模型和阿里云百炼 API 切换
- **动态配置**：运行时调整检索策略、重排参数
- **多格式支持**：PDF、TXT、MD、Word(.docx)、Excel(.xlsx/.xls)
- **OCR 支持**：扫描件 PDF 自动 OCR 识别
- **表格提取**：PDF 表格自动提取

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      FastAPI 前端                        │
│  (RESTful API + Web UI)                                │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                    RAG Pipeline                          │
│  - 查询改写 (Query Rewriting)                            │
│  - 混合检索 (Hybrid Retrieval)                           │
│  - 重排 (Reranking)                                    │
│  - 生成回答 (Answer Generation)                          │
└──────────┬──────────────────┬───────────────────────────┘
           │                  │
┌──────────▼──────────┐  ┌────▼──────────────────────────┐
│   向量存储 (Chroma) │  │    大语言模型 (LLM)            │
│   - 语义检索        │  │    - 本地模型 (Qwen)           │
│   - BM25 索引      │  │    - API 模型 (DashScope)      │
└─────────────────────┘  └───────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│                   文档处理模块                            │
│  - PDF (PyMuPDF + pdfplumber + OCR)                    │
│  - TXT / Markdown                                       │
│  - Word (.docx)                                         │
│  - Excel (.xlsx/.xls)                                   │
└─────────────────────────────────────────────────────────┘
```

## 技术栈

- **后端**：FastAPI + LangChain
- **向量数据库**：ChromaDB
- **嵌入模型**：Qwen3-Embedding-0.6B
- **重排模型**：BGE-Reranker-Base
- **OCR**：PaddleOCR
- **文档处理**：PyMuPDF, pdfplumber, python-docx, openpyxl

## 环境要求

- Python 3.10+
- CUDA (可选，用于 GPU 加速)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

在 `.env` 文件中配置环境变量：

```env
# 阿里云百炼 API
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# 本地模型路径（可选）
Qwen3_0.6B_PATH=path/to/Qwen3_0.6B
Qwen2.5-1.5B-Instruct_PATH=path/to/Qwen2.5-1.5B-Instruct
Qwen3_Embedding_0.6B_PATH=path/to/Qwen3_Embedding_0.6B
```

### 主要配置参数 (config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SIZE` | 500 | 文档分块大小 |
| `CHUNK_OVERLAP` | 80 | 分块重叠长度 |
| `RETRIEVAL_MODE` | hybrid | 检索模式：vector/bm25/hybrid |
| `HYBRID_WEIGHTS` | (0.2, 0.8) | BM25/向量权重 |
| `RERANKER_ENABLED` | True | 是否启用重排 |
| `RERANKER_TOP_K` | 4 | 重排后返回数量 |
| `QUERY_REWRITING_ENABLED` | False | 是否启用查询改写 |

## 启动服务

```bash
python main_api.py
```

服务启动后访问：
- Web 界面：http://127.0.0.1:9000/
- API 文档：http://127.0.0.1:9000/api/docs
- 健康检查：http://127.0.0.1:9000/api/health

## API 接口

### 文件管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/upload | 上传单个文件 |
| POST | /api/batch-upload | 批量上传文件 |
| GET | /api/files | 获取文件列表 |
| DELETE | /api/files/{filename} | 删除文件 |

### 问答

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/query | 问答查询 |

### 模型管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/models | 获取可用模型列表 |
| POST | /api/switch-model | 切换 LLM 模型 |
| POST | /api/switch-embedding | 切换嵌入模型 |

### 检索配置

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/retrieval-config | 获取检索配置 |
| POST | /api/update-retrieval-config | 更新检索配置 |

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/status | 系统状态 |
| GET | /api/health | 健康检查 |
| GET | /api/history | 查询历史 |
| DELETE | /api/history | 清空历史 |
| POST | /api/reprocess | 重新处理文件 |

## 支持的文件格式

- PDF (.pdf) - 支持 OCR 扫描件
- 文本 (.txt)
- Markdown (.md)
- Word (.docx)
- Excel (.xlsx, .xls)

## 项目结构

```
.
├── main_api.py           # FastAPI 主入口
├── config.py             # 配置文件
├── documents.py          # 文档处理模块
├── vector_store.py       # 向量存储管理
├── retrieval.py          # 检索模块
├── rag_pipeline.py       # RAG 流水线
├── models.py             # 模型管理
├── query_rewriting.py    # 查询改写
├── data/                 # 数据目录
├── chroma_db/            # 向量数据库
├── models/               # 本地模型
├── uploads/              # 上传文件
├── static/               # 前端静态文件
└── logs/                 # 日志目录
```

## 使用示例

### 上传文档

通过 Web 界面上传 PDF、Word、Excel 文件，系统自动进行分块、向量化处理。

### 问答

```bash
curl -X POST "http://127.0.0.1:9000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "劳动合同法有哪些规定？",
    "k": 12,
    "score_threshold": 0.45
  }'
```

### 切换模型

```bash
curl -X POST "http://127.0.0.1:9000/api/switch-model" \
  -H "Content-Type: application/json" \
  -d '{"model_key": "dashscope-qwen-turbo"}'
```

### 调整检索配置

```bash
curl -X POST "http://127.0.0.1:9000/api/update-retrieval-config" \
  -H "Content-Type: application/json" \
  -d '{
    "retrieval_mode": "hybrid",
    "bm25_weight": 0.3,
    "vector_weight": 0.7,
    "reranker_enabled": true,
    "reranker_top_k": 5
  }'
```

## 许可证

MIT License
