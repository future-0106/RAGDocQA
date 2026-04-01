# 法律智能Agent平台 - 完整实施规划

## 📋 项目概述

基于现有 RAG 系统，构建**多智能体法律Agent平台**，实现：
- ✅ RAG 知识库增强
- ✅ 多Agent协作（咨询、审查、文书生成）
- ✅ 模型微调（法律领域专用模型）

---

## 📊 现有资源

### 模型资源
| 类型 | 模型 | 路径 |
|-----|------|------|
| LLM | Qwen3-4B | `D:\projects\fastapi_langchain_env\NAIVERAG_New\model\LLM\Qwen3-4B` |
| LLM | Qwen2.5-1.5B-Instruct | `...\model\LLM\Qwen2.5-1.5B-Instruct` |
| LLM | Qwen3_0.6B | `...\model\LLM\Qwen3_0.6B` |
| Embedding | Qwen3_Embedding_0.6B | `...\model\Embedding\Qwen3_Embedding_0.6B` |
| Embedding | gte_Qwen2-1.5B-instruct | `...\model\Embedding\gte_Qwen2-1.5B-instruct` |
| Reranker | bge-reranker-base | `...\model\Reranker\bge-reranker-base` |

### 数据集资源
| 数据类型 | 文件 | 大小 | 用途 |
|---------|------|------|------|
| 法律文档 | 劳动合同、工伤条例、仲裁法等 PDF/Word | 多个 | RAG 知识库 |
| 案例手册 | 劳动法102个实务案例、2025年案例手册 | 多个 | RAG 知识库 |
| 问答对 | DISC-Law-SFT-Pair-QA | 94MB | 微调训练 |
| 三元组 | DISC-Law-SFT-Triplet-QA | 85MB | 微调训练 |
| 语料库 | 法律领域语料库1/2 | 1.2GB | 微调训练 |
| 文书模板 | 起诉状、答辩状、仲裁申请书 | 多个 | 文书生成参考 |

---

## 🗂️ 文件结构规划

```
12.0_version/
├── main_api.py                    # FastAPI主入口（新增Agent接口）
├── config.py                      # 配置文件
├── models.py                      # 模型管理
├── documents.py                   # 文档处理
├── vector_store.py                # 向量存储
├── retrieval.py                   # 检索模块
├── rag_pipeline.py               # RAG流水线
├── query_rewriting.py             # 查询改写
│
├── agent/                         # 【新增】Agent模块
│   ├── __init__.py
│   ├── tools.py                   # 工具定义
│   ├── agents.py                  # Agent定义（CrewAI）
│   ├── coordinator.py             # 多Agent协调器
│   └── prompts.py                 # 提示词模板
│
├── data_processor/               # 【新增】数据处理模块
│   ├── __init__.py
│   ├── legal_loader.py            # 法律文档加载器
│   ├── dataset_loader.py          # 数据集加载器
│   └── processor.py               # 批量处理
│
├── fine_tune/                    # 【新增】微调模块
│   ├── __init__.py
│   ├── data_preparator.py         # 训练数据准备
│   ├── trainer.py                 # 训练脚本
│   └── config.py                  # 微调配置
│
├── data/                         # 法律文档数据
├── chroma_db/                    # 向量数据库
└── model/                        # 模型存储
```

---

## 📌 第一阶段：数据处理

**预计时间：1-2天**

### 1.1 法律文档加载器

```python
# data_processor/legal_loader.py
from pathlib import Path
from typing import List
from langchain_core.documents import Document

class LegalDocumentLoader:
    """法律文档加载器"""
    
    SUPPORTED_FORMATS = {".pdf", ".docx", ".xlsx", ".txt", ".md"}
    
    def __init__(self, chunk_size=500, chunk_overlap=80):
        from documents import DocumentProcessor
        self.processor = DocumentProcessor(chunk_size, chunk_overlap)
    
    def load_file(self, file_path: str) -> List[Document]:
        """加载单个文件"""
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix not in self.SUPPORTED_FORMATS:
            return []
        
        return self.processor.process_file(file_path)
    
    def load_directory(self, dir_path: str) -> List[Document]:
        """批量加载目录下的所有法律文档"""
        documents = []
        dir_path = Path(dir_path)
        
        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    docs = self.load_file(str(file_path))
                    documents.extend(docs)
                    print(f"✅ 已加载: {file_path.name} ({len(docs)} 个文档块)")
                except Exception as e:
                    print(f"❌ 加载失败: {file_path.name} - {e}")
        
        return documents
    
    def load_dataset(self, dataset_path: str) -> List[dict]:
        """加载问答数据集（DISC-Law）"""
        import json
        
        datasets = []
        path = Path(dataset_path)
        
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    datasets.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                datasets = json.load(f)
        
        return datasets
```

### 1.2 数据集加载器

```python
# data_processor/dataset_loader.py
import json
from pathlib import Path
from typing import List, Dict

class DatasetLoader:
    """数据集加载器 - 用于微调数据准备"""
    
    def __init__(self):
        self.supported_formats = {".jsonl", ".json", ".jsonc"}
    
    def load_disc_law_pair_qa(self, file_path: str) -> List[Dict]:
        """加载 DISC-Law 问答对数据集"""
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                data.append({
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "type": item.get("type", "")
                })
        return data
    
    def load_disc_law_triplet(self, file_path: str) -> List[Dict]:
        """加载 DISC-Law 三元组数据集"""
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                data.append({
                    "query": item.get("query", ""),
                    "positive": item.get("positive", ""),
                    "negative": item.get("negative", "")
                })
        return data
    
    def convert_to_sft_format(self, qa_data: List[Dict]) -> List[Dict]:
        """转换为SFT训练格式"""
        sft_data = []
        for item in qa_data:
            sft_data.append({
                "messages": [
                    {"role": "system", "content": "你是一位专业的法律咨询顾问，擅长解答劳动法相关问题。"},
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]}
                ]
            })
        return sft_data
```

### 1.3 批量处理器

```python
# data_processor/processor.py
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document
from data_processor.legal_loader import LegalDocumentLoader
from data_processor.dataset_loader import DatasetLoader

class LegalDataProcessor:
    """法律数据批量处理器"""
    
    def __init__(self, vector_manager, hybrid_manager):
        self.vector_manager = vector_manager
        self.hybrid_manager = hybrid_manager
        self.loader = LegalDocumentLoader()
        self.dataset_loader = DatasetLoader()
    
    def process_legal_documents(self, source_dir: str) -> Dict:
        """处理法律文档并向量化"""
        print(f"📂 开始处理法律文档: {source_dir}")
        
        # 1. 加载文档
        documents = self.loader.load_directory(source_dir)
        print(f"📄 共加载 {len(documents)} 个文档块")
        
        if not documents:
            return {"success": False, "message": "未找到可处理的文档"}
        
        # 2. 向量化存储
        if self.vector_manager.vector_store:
            doc_texts = self.vector_manager.add_documents(documents)
        else:
            doc_texts = self.vector_manager.create_from_documents(documents)
        
        # 3. 更新BM25索引
        if hasattr(self.hybrid_manager, 'bm25_retriever'):
            self.hybrid_manager.bm25_retriever.update_documents(doc_texts)
        
        return {
            "success": True,
            "document_count": len(documents),
            "message": f"成功处理 {len(documents)} 个文档块"
        }
    
    def prepare_fine_tune_data(self, dataset_paths: List[str], output_dir: str) -> Dict:
        """准备微调数据"""
        print("📊 准备微调数据...")
        
        all_sft_data = []
        
        for dataset_path in dataset_paths:
            path = Path(dataset_path)
            if "Pair-QA" in path.name or "Pair" in path.name:
                qa_data = self.dataset_loader.load_disc_law_pair_qa(dataset_path)
                sft_data = self.dataset_loader.convert_to_sft_format(qa_data)
                all_sft_data.extend(sft_data)
                print(f"  - {path.name}: {len(qa_data)} 条问答")
        
        # 分割训练集/验证集
        import random
        random.shuffle(all_sft_data)
        split_idx = int(len(all_sft_data) * 0.9)
        
        train_data = all_sft_data[:split_idx]
        val_data = all_sft_data[split_idx:]
        
        # 保存
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_path / "train.jsonl", "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        with open(output_path / "val.jsonl", "w", encoding="utf-8") as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        return {
            "success": True,
            "train_count": len(train_data),
            "val_count": len(val_data),
            "output_dir": str(output_path)
        }
```

---

## 🤖 第二阶段：Agent框架

**预计时间：3-5天**

### 2.1 工具定义

```python
# agent/tools.py
from langchain.tools import tool
from typing import List, Dict

class LegalTools:
    """法律领域工具集"""
    
    def __init__(self, hybrid_manager, llm):
        self.hybrid_manager = hybrid_manager
        self.llm = llm
    
    @tool
    def search_law(self, query: str) -> str:
        """搜索相关法律法规
        
        Args:
            query: 法律查询问题
            
        Returns:
            相关的法律条文
        """
        results = self.hybrid_manager.retrieve(query, k=5)
        
        if not results:
            return "未找到相关法律法规"
        
        context = "\n\n".join([
            f"【{r['source']}】\n{r['content'][:500]}"
            for r in results[:5]
        ])
        
        return f"找到 {len(results)} 条相关法规：\n\n{context}"
    
    @tool
    def search_case(self, query: str) -> str:
        """检索类似案例
        
        Args:
            query: 案件相关描述
            
        Returns:
            类似案例及判决结果
        """
        results = self.hybrid_manager.retrieve(query, k=8)
        
        if not results:
            return "未找到类似案例"
        
        context = "\n\n".join([
            f"【案例{i+1}】{r['content'][:400]}..."
            for i, r in enumerate(results[:5])
        ])
        
        return f"找到 {len(results)} 个类似案例：\n\n{context}"
    
    @tool
    def review_contract(self, contract_text: str) -> str:
        """审查合同，识别风险
        
        Args:
            contract_text: 合同文本内容
            
        Returns:
            合同审查结果和风险提示
        """
        # 使用LLM进行合同审查
        prompt = f"""你是一位资深劳动法律师，请审查以下劳动合同，识别风险条款。

合同内容：
{contract_text}

请按以下格式输出：
## 审查结果
### 风险条款（按严重程度排序）
1. [条款内容] - 风险等级：[高/中/低] - 风险说明

### 修改建议
1. [具体修改建议]

### 补充说明
[其他需要注意的事项]
"""
        result = self.llm.invoke(prompt)
        return result.content
    
    @tool
    def generate_document(self, doctype: str, facts: str, legal_basis: str = "") -> str:
        """生成法律文书
        
        Args:
            doctype: 文书类型（起诉状/答辩状/仲裁申请书/劳动合同/协议）
            facts: 事实情况描述
            legal_basis: 相关法律条文（可选）
            
        Returns:
            生成的法律文书
        """
        prompt = f"""你是一位法律文书专家，请根据以下事实生成《{doctype}》。

## 事实情况
{facts}

{f"## 相关法律条文\n{legal_basis}" if legal_basis else ""}

## 要求
- 格式规范，符合法律文书标准
- 语言严谨，表述准确
- 要点完整，逻辑清晰

请生成完整的法律文书：
"""
        result = self.llm.invoke(prompt)
        return result.content
    
    @tool
    def assess_risk(self, case_type: str, facts: str) -> str:
        """评估诉讼风险
        
        Args:
            case_type: 案件类型（如劳动争议、合同纠纷等）
            facts: 案件事实描述
            
        Returns:
            风险评估和建议
        """
        prompt = f"""你是一位资深法律顾问，请评估以下案件的风险。

案件类型：{case_type}

案件事实：
{facts}

请按以下格式输出：
## 风险评估
### 胜诉概率分析
[分析有利因素和不利因素]

### 风险等级
[高/中/低]

### 建议
1. [行动建议]
2. [需要准备的材料]
3. [可能的结果和应对方案]

### 注意事项
[重要提示]
"""
        result = self.llm.invoke(prompt)
        return result.content
    
    @tool
    def get_procedure_guide(self, procedure_type: str) -> str:
        """获取法律流程指引
        
        Args:
            procedure_type: 流程类型（劳动仲裁/诉讼/工伤认定/法律援助等）
            
        Returns:
            流程步骤和注意事项
        """
        guides = {
            "劳动仲裁": """
## 劳动仲裁流程

### 第一步：准备材料
1. 身份证原件及复印件
2. 仲裁申请书（按被申请人数量准备副本）
3. 证据材料清单及证据复印件
4. 用人单位工商登记信息

### 第二步：提交申请
- 管辖范围：用人单位所在地或劳动合同履行地的劳动人事争议仲裁委员会
- 受理时间：收到材料后5个工作日内
- 受理范围：确认劳动关系、报酬、社保、工伤等

### 第三步：受理与开庭
- 受理后5日内将申请书副本送达被申请人
- 被申请人10日内提交答辩书
- 仲裁庭在收到材料后45日内作出裁决

### 注意事项
- 仲裁时效：一年（从知道或应当知道权利被侵害之日起）
- 费用：免费
- 可以委托代理人
            """,
            "工伤认定": """
## 工伤认定流程

### 第一步：申请条件
- 在工作时间和工作场所内
- 因工作原因受到事故伤害
- 患职业病
- 上下班途中非本人主要责任的交通事故
等情形

### 第二步：申请材料
1. 工伤认定申请表
2. 与用人单位存在劳动关系的证明材料
3. 医疗诊断证明或职业病诊断证明
4. 事故现场目击证人证言（如有）

### 第三步：申请程序
- 单位应在事故伤害发生后30日内申请
- 单位不申请的，职工可在1年内申请
- 提交至社会保险行政部门

### 注意事项
- 保留好所有诊疗记录和费用票据
- 及时申报，避免超过时效
            """,
        }
        
        return guides.get(procedure_type, "暂未提供该流程指引，请咨询专业人士。")


def create_tools(hybrid_manager, llm) -> Dict:
    """创建工具实例"""
    tools_instance = LegalTools(hybrid_manager, llm)
    
    return {
        "search_law": tools_instance.search_law,
        "search_case": tools_instance.search_case,
        "review_contract": tools_instance.review_contract,
        "generate_document": tools_instance.generate_document,
        "assess_risk": tools_instance.assess_risk,
        "get_procedure_guide": tools_instance.get_procedure_guide,
    }
```

### 2.2 Agent定义（CrewAI）

```python
# agent/agents.py
from crewai import Agent
from typing import List

class LegalConsultationAgent:
    """法律咨询Agent - 面向普通民众的法律问题解答"""
    
    @staticmethod
    def create(llm, tools: List):
        return Agent(
            role="法律咨询顾问",
            goal="用通俗易懂的语言解答用户的法律问题，帮助他们了解自己的权利和应对方法",
            backstory="""你是资深法律咨询专家，拥有10年以上法律咨询经验。
            你擅长用简单直白的语言解释复杂的法律问题，让普通民众也能听得懂。
            你总是站在用户角度，为他们提供切实可行的建议。""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iterations=5,
        )


class ContractReviewAgent:
    """合同审查Agent - 识别合同风险"""
    
    @staticmethod
    def create(llm, tools: List):
        return Agent(
            role="合同审查专家",
            goal="识别合同中的风险条款并提供具体的修改建议，帮助用户规避法律风险",
            backstory="""你是执业15年以上的资深律师，精通劳动法、合同法、公司法。
            你审阅过上万份劳动合同，擅长识别各种隐藏的风险条款。
            你不仅指出问题，还会提供具体可操作的修改建议。""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iterations=5,
        )


class DocumentGeneratorAgent:
    """文书生成Agent - 生成各类法律文书"""
    
    @staticmethod
    def create(llm, tools: List):
        return Agent(
            role="法律文书起草专家",
            goal="根据用户提供的facts生成规范、完整、有针对性的法律文书",
            backstory="""你是法律文书写作专家，专门从事法律文书起草工作20年。
            你精通各类法律文书的格式规范和写作技巧，能够根据不同案情
            起草准确、完整、有说服力的法律文书。""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iterations=5,
        )


class RiskAssessmentAgent:
    """风险评估Agent - 诉讼风险评估"""
    
    @staticmethod
    def create(llm, tools: List):
        return Agent(
            role="诉讼风险评估专家",
            goal="客观评估用户的诉讼风险，提供专业的应对建议",
            backstory="""你是资深法律顾问，擅长诉讼策略制定和风险评估。
            你曾帮助上千当事人分析案件走向，评估胜诉概率，制定诉讼方案。
            你总是给出客观、理性、实事求是的分析。""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iterations=5,
        )
```

### 2.3 多Agent协调器

```python
# agent/coordinator.py
from crewai import Task, Crew, Process
from typing import Dict, Any
from agent.agents import (
    LegalConsultationAgent,
    ContractReviewAgent, 
    DocumentGeneratorAgent,
    RiskAssessmentAgent
)
from agent.tools import create_tools


class AgentCoordinator:
    """多Agent协调器 - 统一入口"""
    
    def __init__(self, llm, hybrid_manager):
        self.llm = llm
        self.hybrid_manager = hybrid_manager
        
        # 创建工具
        self.tools = create_tools(hybrid_manager, llm)
        
        # 初始化各个Agent
        self.consultation_agent = LegalConsultationAgent.create(
            llm, 
            [self.tools["search_law"], self.tools["search_case"], 
             self.tools["assess_risk"], self.tools["get_procedure_guide"]]
        )
        
        self.contract_agent = ContractReviewAgent.create(
            llm,
            [self.tools["review_contract"], self.tools["search_law"], 
             self.tools["assess_risk"]]
        )
        
        self.document_agent = DocumentGeneratorAgent.create(
            llm,
            [self.tools["generate_document"], self.tools["search_law"], 
             self.tools["search_case"]]
        )
        
        self.risk_agent = RiskAssessmentAgent.create(
            llm,
            [self.tools["assess_risk"], self.tools["search_case"], 
             self.tools["search_law"]]
        )
    
    def process_consultation(self, query: str) -> Dict[str, Any]:
        """处理法律咨询"""
        task = Task(
            description=f"""用户咨询：{query}

请用通俗易懂的语言回答用户的问题。
如果需要，可以检索相关法律条文和案例作为参考。
最后给出具体建议和行动步骤。""",
            agent=self.consultation_agent,
            expected_output="通俗易懂的法律建议，包含具体步骤"
        )
        
        crew = Crew(
            agents=[self.consultation_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        
        return {
            "type": "consultation",
            "result": str(result),
            "agent": "LegalConsultationAgent"
        }
    
    def process_contract_review(self, contract_text: str) -> Dict[str, Any]:
        """处理合同审查"""
        task = Task(
            description=f"""请审查以下劳动合同：

{contract_text}

请识别所有风险条款，并提供具体的修改建议。""",
            agent=self.contract_agent,
            expected_output="合同审查报告，包含风险条款和修改建议"
        )
        
        crew = Crew(
            agents=[self.contract_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        
        return {
            "type": "contract_review",
            "result": str(result),
            "agent": "ContractReviewAgent"
        }
    
    def process_document_generation(self, doctype: str, facts: str, 
                                     legal_basis: str = "") -> Dict[str, Any]:
        """处理文书生成"""
        description = f"""请生成《{doctype}》
        
事实情况：{facts}
"""
        if legal_basis:
            description += f"\n相关法律条文：{legal_basis}"
        
        task = Task(
            description=description,
            agent=self.document_agent,
            expected_output=f"完整的{doctype}"
        )
        
        crew = Crew(
            agents=[self.document_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        
        return {
            "type": "document_generation",
            "doctype": doctype,
            "result": str(result),
            "agent": "DocumentGeneratorAgent"
        }
    
    def process_risk_assessment(self, case_type: str, facts: str) -> Dict[str, Any]:
        """处理风险评估"""
        task = Task(
            description=f"""请评估以下案件的风险：

案件类型：{case_type}
案件事实：{facts}

请给出客观的风险评估和专业的建议。""",
            agent=self.risk_agent,
            expected_output="风险评估报告，包含胜诉概率和建议"
        )
        
        crew = Crew(
            agents=[self.risk_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        
        return {
            "type": "risk_assessment",
            "case_type": case_type,
            "result": str(result),
            "agent": "RiskAssessmentAgent"
        }
    
    def auto_route(self, user_input: str) -> Dict[str, Any]:
        """自动路由 - 判断用户意图并分发的Agent"""
        # 简单的关键词匹配判断意图
        # 实际可以使用LLM来判断
        
        user_input_lower = user_input.lower()
        
        # 合同审查
        if any(kw in user_input_lower for kw in ["审查合同", "合同风险", "帮我看看合同", "合同分析"]):
            # 需要用户提供合同内容
            return {
                "type": "need_input",
                "message": "请提供需要审查的合同文本内容",
                "required_input": "contract_text",
                "next_action": "contract_review"
            }
        
        # 文书生成
        if any(kw in user_input_lower for kw in ["起诉状", "申请书", "写文书", "生成文书", "起草"]):
            return {
                "type": "need_input",
                "message": "请提供以下信息：1）文书类型 2）事实情况 3）相关法律条文（可选）",
                "required_input": "document_info",
                "next_action": "document_generation"
            }
        
        # 风险评估
        if any(kw in user_input_lower for kw in ["风险评估", "胜诉概率", "诉讼建议", "胜诉率"]):
            return {
                "type": "need_input",
                "message": "请提供：1）案件类型 2）案件事实描述",
                "required_input": "case_info",
                "next_action": "risk_assessment"
            }
        
        # 流程指引
        if any(kw in user_input_lower for kw in ["流程", "怎么仲裁", "如何起诉", "怎么办理", "步骤"]):
            guide = self.tools["get_procedure_guide"].invoke(
                {"procedure_type": self._extract_procedure_type(user_input)}
            )
            return {
                "type": "procedure_guide",
                "result": guide,
                "agent": "LegalTools"
            }
        
        # 默认：法律咨询
        return self.process_consultation(user_input)
    
    def _extract_procedure_type(self, query: str) -> str:
        """从查询中提取流程类型"""
        query_lower = query.lower()
        
        if "仲裁" in query:
            return "劳动仲裁"
        elif "工伤" in query:
            return "工伤认定"
        elif "诉讼" in query or "起诉" in query:
            return "民事诉讼"
        elif "法律援助" in query:
            return "法律援助"
        else:
            return "劳动仲裁"  # 默认
```

### 2.4 提示词模板

```python
# agent/prompts.py

CONSULTATION_SYSTEM_PROMPT = """你是一位面向普通民众的法律咨询顾问。

## 角色定位
你是一位有温度的法律咨询专家，用通俗易懂的语言帮助普通民众解决法律问题。

## 回答要求
1. 用简单直白的语言，避免过多法律术语
2. 结构清晰，分点说明
3. 告诉用户应该怎么做
4. 提醒需要准备什么材料
5. 指引去相关部门寻求帮助

## 重要提示
- 这只是参考信息，不构成正式法律意见
- 重要决定前建议咨询专业律师
- 如涉及重大权益，及时寻求法律援助

## 输出格式
```
## 问题分析
[简要分析法律关系]

## 建议步骤
1. [第一步]
2. [第二步]
...

## 需要准备的材料
- [材料1]
- [材料2]

## 相关部门
- [部门名称]：地址、电话

## 参考法律
- 《中华人民共和国劳动法》第X条
- 《中华人民共和国劳动合同法》第X条
```
"""

CONTRACT_REVIEW_SYSTEM_PROMPT = """你是一位资深劳动法律师，负责审查劳动合同。

## 审查要点
1. 合同期限和试用期是否合法（试用期最长6个月，同一单位只能约定一次）
2. 工作内容和工作地点是否明确合理
3. 工资构成和支付时间是否明确（不得拖欠工资）
4. 社会保险和福利待遇是否合规（必须缴纳社保）
5. 工作时间和休息休假是否合法
6. 违约条款是否合理公平
7. 竞业限制和服务期条款是否合理
8. 解除合同条件是否过于宽松

## 输出格式
```
## 审查结果

### 风险条款（按严重程度排序）
| 条款 | 风险等级 | 说明 |
|-----|---------|------|
| ... | ... | ... |

### 修改建议
1. [具体建议]

### 综合评价
[总体风险评估]
```
"""

DOCUMENT_GENERATION_SYSTEM_PROMPT = """你是法律文书起草专家。

## 文书类型
你擅长起草以下类型的法律文书：
- 劳动仲裁申请书
- 民事起诉状
- 答辩状
- 证据清单
- 劳动合同
- 解除劳动合同通知书
- 赔偿协议

## 要求
1. 格式规范，符合法律文书标准格式
2. 语言严谨，使用法言法语
3. 要点完整，不遗漏重要事实
4. 逻辑清晰，论证有力

## 格式模板
```
[文书名称]

申请人/原告：XXX 性别：X 民族：X 出生日期：XXXX年XX月XX日
身份证号码：XXXXXXXXXXXXXXXXXX
住址：XXXXXXXXXXXXXXXXXX
联系电话：XXXXXXXXXXX

被申请人/被告：XXX
统一社会信用代码：XXXXXXXXXXXXXXX
地址：XXXXXXXXXXXXXXXXXX
法定代表人：XXX 职务：XXX

仲裁请求/诉讼请求：
1. ...
2. ...

事实与理由：
...

证据清单：
1. ...
2. ...

此致
XXXX劳动人事争议仲裁委员会/XXXX人民法院

申请人：XXX（签名）
XXXX年XX月XX日
```
"""

RISK_ASSESSMENT_SYSTEM_PROMPT = """你是一位资深诉讼风险评估专家。

## 评估维度
1. 胜诉概率分析（有利因素 vs 不利因素）
2. 时间成本
3. 经济成本
4. 执行风险
5. 替代方案

## 输出格式
```
## 案件概述
[简要描述案件]

## 胜诉概率
[XX%] - [概率分析]

## 有利因素
1. ...

## 不利因素
1. ...

## 风险等级
高/中/低

## 建议
1. [行动建议]
2. [材料准备]
3. [可能结果]

## 替代方案
[和解/调解/仲裁等]
```
"""


def get_system_prompt(agent_type: str) -> str:
    """获取对应类型的系统提示词"""
    prompts = {
        "consultation": CONSULTATION_SYSTEM_PROMPT,
        "contract": CONTRACT_REVIEW_SYSTEM_PROMPT,
        "document": DOCUMENT_GENERATION_SYSTEM_PROMPT,
        "risk": RISK_ASSESSMENT_SYSTEM_PROMPT,
    }
    return prompts.get(agent_type, CONSULTATION_SYSTEM_PROMPT)
```

### 2.5 模块初始化

```python
# agent/__init__.py
from agent.tools import create_tools, LegalTools
from agent.agents import (
    LegalConsultationAgent,
    ContractReviewAgent,
    DocumentGeneratorAgent,
    RiskAssessmentAgent
)
from agent.coordinator import AgentCoordinator

__all__ = [
    "create_tools",
    "LegalTools",
    "LegalConsultationAgent",
    "ContractReviewAgent", 
    "DocumentGeneratorAgent",
    "RiskAssessmentAgent",
    "AgentCoordinator"
]
```

---

## 🔌 第三阶段：FastAPI集成

**预计时间：1-2天**

```python
# main_api.py 新增内容

# 在文件顶部添加导入
from agent import AgentCoordinator

# 在 RAGAPISystem 类的 _init_components 方法中添加：
def _init_components(self):
    # ... 原有代码 ...
    
    # 新增：初始化多Agent系统
    print("🔧 初始化多Agent系统...")
    self.agent_coordinator = AgentCoordinator(
        self.llm,
        self.hybrid_manager
    )

# ========== 新增API端点 ==========

class ConsultationRequest(BaseModel):
    """法律咨询请求"""
    question: str = Field(..., description="咨询问题")


class ContractReviewRequest(BaseModel):
    """合同审查请求"""
    contract_text: str = Field(..., description="合同文本内容")


class DocumentRequest(BaseModel):
    """文书生成请求"""
    doctype: str = Field(..., description="文书类型")
    facts: str = Field(..., description="事实情况")
    legal_basis: Optional[str] = Field(default="", description="相关法律条文")


class RiskAssessmentRequest(BaseModel):
    """风险评估请求"""
    case_type: str = Field(..., description="案件类型")
    facts: str = Field(..., description="案件事实")


@app.post("/api/agent/consultation")
async def legal_consultation(request: ConsultationRequest):
    """法律咨询"""
    try:
        result = rag_system.agent_coordinator.process_consultation(request.question)
        return JSONResponse(content={
            "success": True,
            "type": result["type"],
            "answer": result["result"],
            "agent": result["agent"]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.post("/api/agent/contract-review")
async def contract_review(request: ContractReviewRequest):
    """合同审查"""
    try:
        result = rag_system.agent_coordinator.process_contract_review(
            request.contract_text
        )
        return JSONResponse(content={
            "success": True,
            "type": result["type"],
            "review": result["result"],
            "agent": result["agent"]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.post("/api/agent/document")
async def generate_document(request: DocumentRequest):
    """文书生成"""
    try:
        result = rag_system.agent_coordinator.process_document_generation(
            request.doctype,
            request.facts,
            request.legal_basis
        )
        return JSONResponse(content={
            "success": True,
            "type": result["type"],
            "document": result["result"],
            "agent": result["agent"]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.post("/api/agent/risk-assessment")
async def risk_assessment(request: RiskAssessmentRequest):
    """风险评估"""
    try:
        result = rag_system.agent_coordinator.process_risk_assessment(
            request.case_type,
            request.facts
        )
        return JSONResponse(content={
            "success": True,
            "type": result["type"],
            "assessment": result["result"],
            "agent": result["agent"]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.post("/api/agent/auto")
async def agent_auto(request: QueryRequest):
    """自动路由 - 根据用户输入自动判断意图并处理"""
    try:
        result = rag_system.agent_coordinator.auto_route(request.question)
        return JSONResponse(content={
            "success": True,
            **result
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


# ========== 批量数据处理端点 ==========

class ProcessDocumentsRequest(BaseModel):
    """处理文档请求"""
    source_dir: str = Field(..., description="源目录路径")


@app.post("/api/data/process-documents")
async def process_documents(request: ProcessDocumentsRequest):
    """批量处理法律文档"""
    try:
        from data_processor.processor import LegalDataProcessor
        
        processor = LegalDataProcessor(
            rag_system.vector_manager,
            rag_system.hybrid_manager
        )
        
        result = processor.process_legal_documents(request.source_dir)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


class PrepareFineTuneRequest(BaseModel):
    """准备微调数据请求"""
    dataset_paths: List[str] = Field(..., description="数据集路径列表")
    output_dir: str = Field(..., description="输出目录")


@app.post("/api/data/prepare-fine-tune")
async def prepare_fine_tune(request: PrepareFineTuneRequest):
    """准备微调数据"""
    try:
        from data_processor.processor import LegalDataProcessor
        
        processor = LegalDataProcessor(
            rag_system.vector_manager,
            rag_system.hybrid_manager
        )
        
        result = processor.prepare_fine_tune_data(
            request.dataset_paths,
            request.output_dir
        )
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })
```

---

## 🔧 第四阶段：模型微调

**预计时间：3-5天**

### 4.1 微调配置

```python
# fine_tune/config.py
"""
微调配置文件
支持 LoRA 微调，降低显存需求
"""

FINE_TUNE_CONFIG = {
    # 基础模型配置
    "base_model": {
        "name": "Qwen3-4B",
        "path": r"D:\projects\fastapi_langchain_env\NAIVERAG_New\model\LLM\Qwen3-4B",
    },
    
    # 训练参数
    "training": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "fp16": True,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "eval_strategy": "steps",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
    },
    
    # LoRA 配置
    "lora": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    
    # 数据配置
    "data": {
        "train_file": "fine_tune/output/train.jsonl",
        "val_file": "fine_tune/output/val.jsonl",
        "output_dir": "fine_tune/output/model",
        "max_seq_length": 2048,
    },
    
    # 推理配置（微调后）
    "inference": {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 2048,
    }
}


# 可选：使用 QLoRA（更省显存）
QLORA_CONFIG = {
    **FINE_TUNE_CONFIG,
    "lora": {
        **FINE_TUNE_CONFIG["lora"],
        "r": 16,
        "lora_alpha": 32,
    },
    "training": {
        **FINE_TUNE_CONFIG["training"],
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
    },
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
    }
}
```

### 4.2 训练数据准备

```python
# fine_tune/data_preparator.py
"""
微调数据准备脚本
将法律领域数据转换为训练格式
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split


class FineTuneDataPreparator:
    """微调数据准备器"""
    
    def __init__(self, output_dir: str = "fine_tune/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_disc_law_qa(self, file_path: str) -> List[Dict]:
        """加载 DISC-Law 问答数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data
    
    def convert_to_chat_format(self, question: str, answer: str, 
                                system_prompt: str = None) -> Dict:
        """转换为聊天格式"""
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        else:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        
        return {"messages": messages}
    
    def prepare_law_qa_data(self, qa_data: List[Dict]) -> List[Dict]:
        """准备法律问答训练数据"""
        system_prompt = "你是一位专业的法律咨询顾问，擅长解答劳动法相关问题。请用通俗易懂的语言回答用户的问题。"
        
        processed = []
        for item in qa_data:
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            if question and answer:
                processed.append(
                    self.convert_to_chat_format(question, answer, system_prompt)
                )
        
        return processed
    
    def prepare_contract_review_data(self, data: List[Dict]) -> List[Dict]:
        """准备合同审查训练数据"""
        system_prompt = "你是一位资深劳动法律师，擅长审查劳动合同，识别风险条款并提供修改建议。"
        
        processed = []
        for item in data:
            # 构造审查任务
            question = f"请审查以下劳动合同，识别风险条款并给出修改建议：\n\n{item.get('contract_text', '')}"
            answer = item.get('review_result', '')
            
            if question and answer:
                processed.append(
                    self.convert_to_chat_format(question, answer, system_prompt)
                )
        
        return processed
    
    def split_data(self, data: List[Dict], 
                    train_ratio: float = 0.9) -> Tuple[List[Dict], List[Dict]]:
        """分割训练集和验证集"""
        train_data, val_data = train_test_split(
            data, 
            test_size=1-train_ratio,
            random_state=42
        )
        return train_data, val_data
    
    def save_jsonl(self, data: List[Dict], file_path: str):
        """保存为 JSONL 格式"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def prepare_from_multiple_sources(self, sources: List[str], 
                                       output_name: str = "law_sft") -> Dict:
        """从多个数据源准备训练数据"""
        all_data = []
        
        for source in sources:
            path = Path(source)
            if not path.exists():
                print(f"⚠️ 文件不存在: {source}")
                continue
            
            print(f"📂 加载: {path.name}")
            
            if "Pair-QA" in path.name or "Pair" in path.name:
                raw_data = self.load_disc_law_qa(str(path))
                processed = self.prepare_law_qa_data(raw_data)
                all_data.extend(processed)
                print(f"  - 加载 {len(raw_data)} 条数据 -> {len(processed)} 条训练数据")
        
        print(f"\n📊 总计: {len(all_data)} 条训练数据")
        
        # 分割
        train_data, val_data = self.split_data(all_data)
        print(f"  - 训练集: {len(train_data)} 条")
        print(f"  - 验证集: {len(val_data)} 条")
        
        # 保存
        train_path = self.output_dir / f"{output_name}_train.jsonl"
        val_path = self.output_dir / f"{output_name}_val.jsonl"
        
        self.save_jsonl(train_data, str(train_path))
        self.save_jsonl(val_data, str(val_path))
        
        return {
            "success": True,
            "train_file": str(train_path),
            "val_file": str(val_path),
            "train_count": len(train_data),
            "val_count": len(val_data)
        }


if __name__ == "__main__":
    # 使用示例
    preparator = FineTuneDataPreparator()
    
    sources = [
        r"C:\Users\32459\Desktop\可用劳动数据集\DISC-Law-SFT-Pair-QA-released.jsonl",
        # 可以添加更多数据源
    ]
    
    result = preparator.prepare_from_multiple_sources(sources)
    print(f"\n✅ 数据准备完成: {result}")
```

### 4.3 训练脚本

```python
# fine_tune/trainer.py
"""
模型训练脚本
支持 LoRA / QLoRA 微调
"""

import os
import torch
from pathlib import Path
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset


class LegalModelTrainer:
    """法律模型训练器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        model_path = self.config["base_model"]["path"]
        
        print(f"📥 加载模型: {model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 设置 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        load_in_8bit = self.config.get("quantization", {}).get("load_in_8bit", False)
        load_in_4bit = self.config.get("quantization", {}).get("load_in_4bit", False)
        
        if load_in_4bit or load_in_8bit:
            # 量化加载
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                trust_remote_code=True
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            # 普通加载
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        print(f"✅ 模型加载完成")
        return self.model, self.tokenizer
    
    def setup_lora(self):
        """设置 LoRA"""
        lora_config = self.config["lora"]
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("lora_alpha", 16),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            bias=lora_config.get("bias", "none"),
            inference_mode=False
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def load_dataset(self, train_file: str, val_file: str):
        """加载数据集"""
        def format_data(examples):
            # 将 messages 格式化为文本
            texts = []
            for messages in examples["messages"]:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(text)
            
            return {"text": texts}
        
        # 加载训练集
        train_dataset = load_dataset("json", data_files=train_file, split="train")
        val_dataset = load_dataset("json", data_files=val_file, split="train")
        
        # 格式化
        train_dataset = train_dataset.map(
            format_data,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            format_data,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        return train_dataset, val_dataset
    
    def train(self):
        """执行训练"""
        # 加载模型
        self.load_model_and_tokenizer()
        
        # 设置 LoRA
        self.setup_lora()
        
        # 加载数据
        train_file = self.config["data"]["train_file"]
        val_file = self.config["data"]["val_file"]
        
        train_dataset, val_dataset = self.load_dataset(train_file, val_file)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config["data"]["output_dir"],
            num_train_epochs=self.config["training"]["num_train_epochs"],
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=self.config["training"]["learning_rate"],
            warmup_steps=self.config["training"]["warmup_steps"],
            weight_decay=self.config["training"]["weight_decay"],
            fp16=self.config["training"].get("fp16", True),
            logging_steps=self.config["training"]["logging_steps"],
            save_steps=self.config["training"]["save_steps"],
            eval_steps=self.config["training"]["eval_steps"],
            eval_strategy=self.config["training"]["eval_strategy"],
            save_total_limit=self.config["training"]["save_total_limit"],
            load_best_model_at_end=self.config["training"]["load_best_model_at_end"],
            report_to=["tensorboard"],
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        print("\n🚀 开始训练...")
        trainer.train()
        
        # 保存模型
        output_dir = Path(self.config["data"]["output_dir"])
        final_path = output_dir / "final"
        trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        
        print(f"\n✅ 训练完成！模型已保存到: {final_path}")
        
        return final_path
    
    def merge_and_save(self, output_path: str):
        """合并 LoRA 权重并保存"""
        from peft import PeftModel
        
        # 合并权重
        merged_model = self.model.merge_and_unload()
        
        # 保存
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        print(f"✅ 合并后模型已保存到: {output_path}")


def main():
    from fine_tune.config import FINE_TUNE_CONFIG, QLORA_CONFIG
    
    # 选择配置（推荐 QLoRA，省显存）
    config = QLORA_CONFIG
    
    # 初始化训练器
    trainer = LegalModelTrainer(config)
    
    # 执行训练
    model_path = trainer.train()
    
    print(f"\n🎉 微调完成！")
    print(f"模型路径: {model_path}")


if __name__ == "__main__":
    main()
```

### 4.4 模块初始化

```python
# fine_tune/__init__.py
from fine_tune.data_preparator import FineTuneDataPreparator
from fine_tune.trainer import LegalModelTrainer
from fine_tune.config import FINE_TUNE_CONFIG, QLORA_CONFIG

__all__ = [
    "FineTuneDataPreparator",
    "LegalModelTrainer",
    "FINE_TUNE_CONFIG",
    "QLORA_CONFIG"
]
```

---

## 📅 完整时间规划

| 阶段 | 任务 | 时间 | 交付物 |
|-----|------|-----|--------|
| **第一阶段** | 数据处理 | 1-2天 | `data_processor/` 目录 |
| | - 法律文档加载器 | | `legal_loader.py` |
| | - 数据集加载器 | | `dataset_loader.py` |
| | - 批量处理器 | | `processor.py` |
| **第二阶段** | Agent框架 | 3-5天 | `agent/` 目录 |
| | - 工具定义 | | `tools.py` |
| | - Agent定义 | | `agents.py` |
| | - 协调器 | | `coordinator.py` |
| | - 提示词 | | `prompts.py` |
| **第三阶段** | FastAPI集成 | 1-2天 | 新增 API 端点 |
| **第四阶段** | 模型微调 | 3-5天 | `fine_tune/` 目录 |
| | - 数据准备 | | `data_preparator.py` |
| | - 训练脚本 | | `trainer.py` |
| | - 微调配置 | | `config.py` |
| **测试优化** | 整体测试 | 2-3天 | 完整可用系统 |

**总计：约 10-17 天**

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install crewai langchain langchain-community
pip install peft transformers datasets accelerate
pip install scikit-learn
```

### 2. 处理数据

```python
from data_processor.processor import LegalDataProcessor
from data_processor.dataset_loader import DatasetLoader

# 处理法律文档到向量库
processor = LegalDataProcessor(vector_manager, hybrid_manager)
result = processor.process_legal_documents(
    r"C:\Users\32459\Desktop\可用劳动数据集"
)
print(result)

# 准备微调数据
result = processor.prepare_fine_tune_data(
    dataset_paths=[
        r"C:\Users\32459\Desktop\可用劳动数据集\DISC-Law-SFT-Pair-QA-released.jsonl"
    ],
    output_dir="fine_tune/output"
)
print(result)
```

### 3. 启动Agent服务

```python
from agent import AgentCoordinator

# 初始化Agent系统
agent_coordinator = AgentCoordinator(llm, hybrid_manager)

# 法律咨询
result = agent_coordinator.process_consultation("老板拖欠工资怎么办？")

# 合同审查
result = agent_coordinator.process_contract_review(contract_text)

# 文书生成
result = agent_coordinator.process_document_generation(
    "劳动仲裁申请书",
    "我被公司违法辞退了..."
)

# 自动路由
result = agent_coordinator.auto_route("帮我看看这份合同有没有问题")
```

### 4. 调用API

```bash
# 法律咨询
curl -X POST "http://127.0.0.1:9000/api/agent/consultation" \
  -H "Content-Type: application/json" \
  -d '{"question": "老板拖欠工资怎么办？"}'

# 合同审查
curl -X POST "http://127.0.0.1:9000/api/agent/contract-review" \
  -H "Content-Type: application/json" \
  -d '{"contract_text": "合同内容..."}'

# 文书生成
curl -X POST "http://127.0.0.1:9000/api/agent/document" \
  -H "Content-Type: application/json" \
  -d '{"doctype": "劳动仲裁申请书", "facts": "事实情况..."}'

# 自动路由
curl -X POST "http://127.0.0.1:9000/api/agent/auto" \
  -H "Content-Type: application/json" \
  -d '{"question": "帮我看看这份合同"}'
```

### 5. 微调模型

```python
from fine_tune.data_preparator import FineTuneDataPreparator
from fine_tune.trainer import LegalModelTrainer
from fine_tune.config import QLORA_CONFIG

# 准备数据
preparator = FineTuneDataPreparator()
preparator.prepare_from_multiple_sources([
    r"C:\Users\32459\Desktop\可用劳动数据集\DISC-Law-SFT-Pair-QA-released.jsonl"
])

# 训练
trainer = LegalModelTrainer(QLORA_CONFIG)
trainer.train()
```

---

## ⚠️ 注意事项

1. **模型选择**
   - 建议先用 Qwen3-4B 测试 Agent 效果
   - 效果好再进行微调

2. **显存需求**
   - Qwen3-4B 全参数训练：约 24GB GPU 显存
   - QLoRA 训练：约 8GB GPU 显存
   - 如显存不足，可使用 Qwen2.5-1.5B

3. **数据集选择**
   - 优先使用 DISC-Law-SFT-Pair-QA（94MB，规模适中）
   - 语料库太大，建议选择性使用

4. **CrewAI版本**
   - 注意兼容性问题，建议固定版本
   - `pip install crewai==0.28.0`

5. **法律风险提示**
   - AI 回答需避免误导，建议标注"仅供参考"
   - 重要法律文书需人工复核

---

## 📚 参考资料

- [CrewAI 文档](https://docs.crewai.com)
- [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/)
- [LoRA 微调教程](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen 微调文档](https://qwen.readthedocs.io/zh-cn/training/lora/)

---

*文档更新时间：2026-03-25*
