"""
Agent协调器 - Agentic RAG版本
使用 LangChain Agent 实现多轮对话和自动工具调用
"""
from typing import Dict, Any, Optional
from langchain.agents import initialize_agent, AgentType
from agent.tools import create_tools, LegalTools
from agent.delilegal_client import create_deli_client
from agent import prompts


class AgentCoordinator:
    """多Agent协调器 - 统一入口（Agentic RAG）"""
    
    def __init__(self, llm, use_external_api: bool = True, verbose: bool = False):
        self.llm = llm
        self.use_external_api = use_external_api
        self.verbose = verbose
        
        deli_client = None
        if use_external_api:
            print("🔧 初始化得理API客户端...")
            deli_client = create_deli_client()
        
        print("🔧 初始化法律工具集...")
        self.tools_instance = LegalTools(llm, deli_client)
        self.tools_dict = create_tools(llm, deli_client)
        self.langchain_tools = self.tools_instance.get_langchain_tools()
        
        print("🔧 初始化法律咨询Agent...")
        self.consultation_agent = initialize_agent(
            self.langchain_tools,
            llm,
            agent=AgentType.CONVERSATIONAL,
            system_message=prompts.CONSULTATION_SYSTEM_PROMPT,
            verbose=verbose,
            max_iterations=5,
            handle_parsing_errors="抱歉，我遇到了一些问题，请重新描述您的问题。"
        )
        
        print("🔧 初始化合同审查Agent...")
        self.contract_agent = initialize_agent(
            self.langchain_tools,
            llm,
            agent=AgentType.CONVERSATIONAL,
            system_message=prompts.CONTRACT_REVIEW_SYSTEM_PROMPT,
            verbose=verbose,
            max_iterations=5,
            handle_parsing_errors="抱歉，我遇到了一些问题，请重新提供合同内容。"
        )
        
        print("🔧 初始化文书生成Agent...")
        self.document_agent = initialize_agent(
            self.langchain_tools,
            llm,
            agent=AgentType.CONVERSATIONAL,
            system_message=prompts.DOCUMENT_GENERATION_SYSTEM_PROMPT,
            verbose=verbose,
            max_iterations=5,
            handle_parsing_errors="抱歉，我遇到了一些问题，请重新提供文书类型和事实情况。"
        )
        
        print("🔧 初始化风险评估Agent...")
        self.risk_agent = initialize_agent(
            self.langchain_tools,
            llm,
            agent=AgentType.CONVERSATIONAL,
            system_message=prompts.RISK_ASSESSMENT_SYSTEM_PROMPT,
            verbose=verbose,
            max_iterations=5,
            handle_parsing_errors="抱歉，我遇到了一些问题，请重新提供案件类型和事实。"
        )
        
        print("✅ Agent系统初始化完成")
    
    def process_consultation(self, query: str) -> Dict[str, Any]:
        """处理法律咨询 - 使用 Agent"""
        try:
            result = self.consultation_agent.run(query)
            
            return {
                "success": True,
                "type": "consultation",
                "answer": result,
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
        """处理合同审查 - 使用 Agent"""
        try:
            result = self.contract_agent.run(
                f"请审查以下劳动合同，识别风险条款并给出修改建议：\n\n{contract_text}"
            )
            
            return {
                "success": True,
                "type": "contract_review",
                "review": result,
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
        """处理文书生成 - 使用 Agent"""
        try:
            input_str = f"{doctype}|{facts}" + (f"|{legal_basis}" if legal_basis else "")
            result = self.document_agent.run(input_str)
            
            return {
                "success": True,
                "type": "document_generation",
                "doctype": doctype,
                "document": result,
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
        """处理风险评估 - 使用 Agent"""
        try:
            input_str = f"{case_type}|{facts}"
            result = self.risk_agent.run(input_str)
            
            return {
                "success": True,
                "type": "risk_assessment",
                "case_type": case_type,
                "assessment": result,
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
        
        if any(kw in user_input_lower for kw in ["起诉状", "申请书", "写文书", "生成文书", "起草", "仲裁申请", "借条", "欠条"]):
            return {
                "success": True,
                "type": "need_input",
                "message": "请提供以下信息：1）文书类型（如仲裁申请书、起诉状、借条）2）事实情况描述",
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
            guide = self.tools_instance.get_procedure_guide(procedure_type)
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
        return self.tools_dict