# Qwen3 RAG 系统
## 注意：在提供的代码文件中要自己配置模型并修改模型导入路径，并且这个一版会存在一些问题后续我持续进行优化。

基于 Qwen3-0.6B 和 Qwen3-Embedding-0.6B 的完整检索增强生成（Retrieval-Augmented Generation）系统，使用 ChromaDB 作为向量数据库。提供两个版本：

- **基础版** (`main.py`)：支持 `.txt`、`.md` 文档，批量处理目录，交互式问答。
- **PDF 增强版** (`main_pdf.py`)：在基础版之上增加 PDF 支持，并提供**文件上传即向量化**功能，支持单个/批量上传 PDF/TXT/MD 文件，即时处理并加入知识库。

---

## 主要特性

- ✅ **轻量级模型**：Qwen3-0.6B（生成）+ Qwen3-Embedding-0.6B（嵌入），可在 CPU 或 GPU 上运行。
- ✅ **本地向量库**：使用 ChromaDB 持久化存储文档向量，无需外部服务。
- ✅ **多格式文档**：
  - 基础版：`.txt`、`.md`
  - PDF 版：`.pdf`、`.txt`、`.md`
- ✅ **文件上传即时向量化**（仅 PDF 版）：上传文件后立即分块、嵌入、存储，无需手动重建索引。
- ✅ **交互式命令行**：支持自定义查询、查看文件列表、向量库统计、重建索引等。
- ✅ **生成控制**：可调整温度、重复惩罚、最大生成长度等参数，支持输出清理（去除思考标签、重复内容等）。

---

## 系统要求

- Python 3.8+
- 操作系统：Windows / Linux / macOS
- 硬件：建议 8GB+ 内存，若使用 GPU 需 CUDA 11.7+

---

## 依赖安装

1. **克隆项目**（或直接将两个 `.py` 文件放入工作目录）：
   ```bash
   git clone <repository-url>
   cd qwen3-rag-chromadb
   ```

2. **创建虚拟环境**（推荐）：
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. **安装依赖**：
   ```bash
   pip install torch transformers langchain-huggingface langchain-chroma chromadb langchain-text-splitters pydantic
   ```
   **PDF 版额外依赖**：
   ```bash
   pip install pdfplumber
   ```

> 若网络环境不便，可使用 `langchain-community` 替代 `langchain-huggingface`，代码已做兼容。

---

## 模型下载与配置

本系统使用 **Qwen3-0.6B** 和 **Qwen3-Embedding-0.6B**，需提前下载至本地。

1. **从 Hugging Face 下载**：
   - [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
   - [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

2. **目录结构建议**（与代码默认路径一致）：
   ```
   your_project/
   ├── main.py
   ├── main_pdf.py
   ├── models/
   │   ├── Qwen3_0.6B/
   │   └── Qwen3_Embedding_0.6B/
   ├── data/           # 文档存放目录（自动创建）
   └── chroma_db/      # 向量数据库目录（自动创建）
   ```

3. **修改模型路径**（可选）：  
   代码中默认使用绝对路径 `D:\projects\...`，**建议修改为你的本地路径**。  
   在 `main()` 中找到 `QwenEmbeddings` 和 `QwenLLM` 的初始化部分，更改 `model_path` 参数即可。

---

## 使用方法

### 1. 基础版 (`main.py`)

适用于纯文本/ Markdown 文档的批量处理与查询。

```bash
python main.py
```

**参数选项**：
- `--test`：仅测试嵌入模型是否加载成功。
- `--data DIR`：指定文档目录（默认 `./data`）。
- `--reset`：启动前删除现有 ChromaDB 存储。

**交互命令**：
- 直接输入问题 → 检索并生成回答。
- `clear`：清屏。
- `stats`：查看向量库统计。
- `delete`：删除整个向量库（需确认）。
- `quit`/`exit`/`q`：退出程序。

### 2. PDF 增强版 (`main_pdf.py`)

支持 PDF，并提供文件上传即时向量化功能。

```bash
python main_pdf.py
```

**参数选项**：
- `--data DIR`：指定文档目录（默认 `./data`）。
- `--reset`：启动前删除现有 ChromaDB 存储。

**交互命令**（包含基础版所有命令，并扩展）：

| 命令            | 说明                                                         |
| --------------- | ------------------------------------------------------------ |
| `upload`        | 上传单个文件（PDF/TXT/MD），立即处理并添加到向量库。         |
| `upload_multi`  | 批量上传多个文件（用逗号或分号分隔路径）。                   |
| `list`          | 列出 `data/` 目录下所有已上传的文件。                        |
| `reload`        | 重新处理 `data/` 目录下所有文件，**重建向量库**（谨慎操作）。 |
| `stats`         | 显示向量库统计 + 数据目录文件列表。                          |
| `clear`         | 清屏。                                                       |
| `quit`/`exit`/`q` | 退出。                                                     |

**上传示例**：
```
💬 请输入命令或问题: upload
📤 上传文件并立即转换为向量
支持格式: PDF, TXT, MD
请输入文件路径 (或输入 'cancel' 取消): /home/user/docs/企业介绍.pdf
✅ 文件 '企业介绍.pdf' 已成功上传并转换为向量，添加到知识库
⏱️  处理时间: 2.35秒
📄 生成 12 个文本块
```

---

## 参数调优

在 `QwenLLM` 类初始化时可调整生成参数：

| 参数               | 说明                           | 推荐值        |
| ------------------ | ------------------------------ | ------------- |
| `max_new_tokens`   | 生成的最大 token 数            | 200~512       |
| `temperature`      | 温度，越低越确定性             | 0.1~0.3       |
| `top_p`            | 核采样阈值                     | 0.85~0.9      |
| `repetition_penalty` | 重复惩罚，越大越不重复       | 1.1~1.5       |
| `do_sample`        | 是否采样（False 则为贪婪解码） | True          |

PDF 版中已预设更严格的重复惩罚（1.5）和更短的生成长度（200），以减少废话和重复。

---

## 注意事项

1. **模型路径**：首次运行请务必修改代码中的硬编码路径，或确保 `models/` 目录下存在对应模型。
2. **PDF 解析**：`pdfplumber` 对扫描版 PDF 支持有限，若无法提取文本请使用 OCR 工具预处理。
3. **向量库冲突**：使用 `--reset` 或 `reload` 命令会删除已有 ChromaDB 目录，请提前备份。
4. **显存占用**：0.6B 模型在 GPU 上约占用 1.5GB 显存，CPU 模式需要 4GB+ 内存。
5. **警告屏蔽**：代码已屏蔽 Transformers 的 generation_config 相关警告，不影响使用。

---

## 常见问题

**Q：导入 HuggingFaceEmbeddings 失败？**  
A：请安装 `langchain-huggingface` 或 `langchain-community`。代码会自动尝试两种导入方式。

**Q：ChromaDB 加载失败，提示目录损坏？**  
A：可能是向量库版本不兼容。使用 `--reset` 参数或 `reload` 命令重建。

**Q：生成回答总是重复或输出思考过程？**  
A：可提高 `repetition_penalty`（如 1.5），并启用 `_clean_response` 后处理。PDF 版已内置过滤逻辑。

**Q：如何添加自己的文档？**  
- 基础版：直接将 `.txt`/`.md` 放入 `data/` 目录，程序启动时会自动处理。  
- PDF 版：使用 `upload` 命令上传，或直接复制文件到 `data/` 目录后使用 `reload` 重建索引。

---

## 许可证

本项目仅供学习和研究使用，模型权重遵循 Qwen 系列原始许可证。

---

