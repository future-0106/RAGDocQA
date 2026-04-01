"""
Agent协调器 - 多Agent统一入口
"""
from typing import Dict, Any, Optional
from agent.tools import create_tools, LegalTools
from agent.agents import (
    LegalConsultationAgent,
    ContractReviewAgent,
    DocumentGeneratorAgent,
    RiskAssessmentAgent
)
from agent.delilegal_client import create_deli_client


class AgentCoordinator:
    """多Agent协调器 - 统一入口"""
    
    def __init__(self, llm, use_external_api: bool = True):
        self.llm = llm
        self.use_external_api = use_external_api
        
        deli_client = None
        if use_external_api:
            print("🔧 初始化得理API客户端...")
            deli_client = create_deli_client()
        
        print("🔧 初始化法律工具集...")
        self.tools = create_tools(llm, deli_client)
        
        print("🔧 初始化法律咨询Agent...")
        self.consultation_agent = LegalConsultationAgent.create(
            llm, 
            self.tools
        )
        
        print("🔧 初始化合同审查Agent...")
        self.contract_agent = ContractReviewAgent.create(
            llm,
            self.tools
        )
        
        print("🔧 初始化文书生成Agent...")
        self.document_agent = DocumentGeneratorAgent.create(
            llm,
            self.tools
        )
        
        print("🔧 初始化风险评估Agent...")
        self.risk_agent = RiskAssessmentAgent.create(
            llm,
            self.tools
        )
        
        print("✅ Agent系统初始化完成")
    
    def process_consultation(self, query: str) -> Dict[str, Any]:
        """处理法律咨询"""
        try:
            law_context = self.tools["search_law"](query)
            case_context = self.tools["search_case"](query)
            
            context = f"""
## 检索到的法律条文
{law_context}

## 检索到的类似案例
{case_context}
"""
            
            from agent import prompts
            full_prompt = prompts.format_consultation_prompt(query, context)
            
            result = self.llm.invoke(full_prompt)
            answer = result.content if hasattr(result, "content") else str(result)
            
            return {
                "success": True,
                "type": "consultation",
                "answer": answer,
                "context": {
                    "law": law_context,
                    "case": case_context
                },
                "agent": "LegalConsultationAgent"
            }
        except Exception as e:
            return {
                "success": False,
                "type": "consultation",
                "error": str(e),
                "agent": "LegalConsultationAgent"
            }
    
    def process_contract_review(self, contract_text: str) -> Dict[str, Any]:
        """处理合同审查"""
        try:
            law_context = self.tools["search_law"]("劳动合同法律规定 试用期 加班费 社会保险")
            
            from agent import prompts
            full_prompt = prompts.format_contract_review_prompt(contract_text, law_context)
            
            result = self.llm.invoke(full_prompt)
            review = result.content if hasattr(result, "content") else str(result)
            
            return {
                "success": True,
                "type": "contract_review",
                "review": review,
                "agent": "ContractReviewAgent"
            }
        except Exception as e:
            return {
                "success": False,
                "type": "contract_review",
                "error": str(e),
                "agent": "ContractReviewAgent"
            }
    
    def process_document_generation(self, doctype: str, facts: str, 
                                     legal_basis: str = "") -> Dict[str, Any]:
        """处理文书生成"""
        try:
            if not legal_basis:
                legal_basis = self.tools["search_law"](facts[:200])
            
            from agent import prompts
            full_prompt = prompts.format_document_prompt(doctype, facts, legal_basis)
            
            result = self.llm.invoke(full_prompt)
            document = result.content if hasattr(result, "content") else str(result)
            
            return {
                "success": True,
                "type": "document_generation",
                "doctype": doctype,
                "document": document,
                "agent": "DocumentGeneratorAgent"
            }
        except Exception as e:
            return {
                "success": False,
                "type": "document_generation",
                "error": str(e),
                "agent": "DocumentGeneratorAgent"
            }
    
    def process_risk_assessment(self, case_type: str, facts: str) -> Dict[str, Any]:
        """处理风险评估"""
        try:
            case_context = self.tools["search_case"](facts[:200])
            law_context = self.tools["search_law"](f"{case_type}法律规定")
            
            from agent import prompts
            full_prompt = prompts.format_risk_assessment_prompt(case_type, facts, 
                                                               case_context + "\n\n" + law_context)
            
            result = self.llm.invoke(full_prompt)
            assessment = result.content if hasattr(result, "content") else str(result)
            
            return {
                "success": True,
                "type": "risk_assessment",
                "case_type": case_type,
                "assessment": assessment,
                "agent": "RiskAssessmentAgent"
            }
        except Exception as e:
            return {
                "success": False,
                "type": "risk_assessment",
                "error": str(e),
                "agent": "RiskAssessmentAgent"
            }
    
    def auto_route(self, user_input: str) -> Dict[str, Any]:
        """自动路由 - 判断用户意图并分发到对应Agent"""
        user_input_lower = user_input.lower()
        
        if any(kw in user_input_lower for kw in ["审查合同", "合同风险", "帮我看看合同", "合同分析", "合同条款"]):
            return {
                "success": True,
                "type": "need_input",
                "message": "请提供需要审查的合同文本内容",
                "required_input": "contract_text",
                "next_action": "contract_review"
            }
        
        if any(kw in user_input_lower for kw in ["起诉状", "申请书", "写文书", "生成文书", "起草", "仲裁申请"]):
            return {
                "success": True,
                "type": "need_input",
                "message": "请提供以下信息：1）文书类型（如仲裁申请书、起诉状）2）事实情况描述",
                "required_input": "document_info",
                "next_action": "document_generation"
            }
        
        if any(kw in user_input_lower for kw in ["风险评估", "胜诉概率", "诉讼建议", "胜诉率", "案件分析"]):
            return {
                "success": True,
                "type": "need_input",
                "message": "请提供：1）案件类型（如劳动争议、合同纠纷）2）案件事实描述",
                "required_input": "case_info",
                "next_action": "risk_assessment"
            }
        
        if any(kw in user_input_lower for kw in ["流程", "怎么仲裁", "如何起诉", "怎么办理", "步骤", "程序"]):
            procedure_type = self._extract_procedure_type(user_input)
            guide = self.tools["get_procedure_guide"](procedure_type)
            return {
                "success": True,
                "type": "procedure_guide",
                "result": guide,
                "agent": "LegalTools",
                "procedure_type": procedure_type
            }
        
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
            return "劳动仲裁"
    
    def get_tools(self) -> Dict:
        """获取工具字典"""
        return self.tools