# 查询改写效果评估 - 使用说明

## 当前状态

评估系统已创建完成，但由于以下原因，当前运行结果为**模拟数据**：

1. **未连接真实RAG系统**：当前使用的是模拟检索结果
2. **未导入文档到知识库**：需要先将可用文件导入RAG系统
3. **未标注相关文档**：需要预先为测试用例标注相关文档

---

## 评估系统文件结构

```
D:\projects\fastapi_langchain_env\NAIVERAG_New\10.0_version\
├── evaluate\
│   ├── prepare_test_data.py     # 测试数据准备脚本
│   ├── evaluate_retrieval.py    # 检索效果评估脚本
│   ├── run_evaluation.py        # 批量评估运行脚本
│   ├── test_dataset.json        # 测试数据集 (184条)
│   └── results\
│       └── evaluation_report.md  # 评估报告
```

---

## 完整评估流程

### 步骤1：准备测试数据（已完成 ✅）

```bash
python evaluate/prepare_test_data.py
```

- 从4个文件提取了184条测试用例
- 已保存到 `evaluate/test_dataset.json`

---

### 步骤2：导入文档到RAG系统

需要先将 `C:\Users\32459\Desktop\可用劳动数据集` 中的文档导入到RAG系统的知识库。

可以使用项目提供的API：

```bash
# 启动API服务
python main_api.py

# 或使用批量上传接口
POST /api/batch-upload
```

---

### 步骤3：运行真实评估

修改评估脚本，接入真实的RAG系统：

```python
# 在 run_evaluation.py 中
# 确保已初始化真实的RAG系统
from your_rag_module import HybridRetrievalManager

# 替换模拟检索为真实检索
def real_search(query, k=10):
    return hybrid_manager.search(query, k=k)
```

---

## 评估指标说明

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **Hit Rate** | 命中率 | 至少检索到1个相关文档的查询比例 |
| **Recall@K** | 召回率 | 检索到的相关文档数 / 总相关文档数 |
| **Precision@K** | 精确率 | 真正相关的文档数 / 检索总数 |
| **MRR** | 平均倒数排名 | 第一个相关文档排名的倒数均值 |

---

## 当前测试数据统计

| 来源 | 数量 |
|------|------|
| JSONL问答对 | 100条 |
| Word问答文件 | 50条 |
| Word法规文件 | 14条 |
| Word纠纷文件 | 20条 |
| **总计** | **184条** |

| 问题类型 | 数量 |
|----------|------|
| simple（简单） | 127条 |
| complex（复杂） | 50条 |
| vague（模糊） | 7条 |

---

## 下一步操作建议

1. **启动RAG系统**：将文档导入知识库
2. **接入真实检索**：修改评估代码，接入真实的混合检索系统
3. **标注相关文档**：为测试用例预先标注相关文档ID
4. **运行评估**：对比基线检索和查询改写的效果差异

---

## 快速测试

如果想快速测试评估系统，可以直接运行：

```bash
cd D:\projects\fastapi_langchain_env\NAIVERAG_New\10.0_version
python evaluate/run_evaluation.py
```

这将使用模拟数据生成一个示例报告。

---

*生成时间：2026-03-16*
