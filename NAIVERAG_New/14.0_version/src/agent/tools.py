"""
法律Agent工具集 - Agentic RAG版本
基于得理开放平台API + LangChain Agent
"""
from typing import Dict, List, Optional
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from agent.delilegal_client import DeliLegalClient, create_deli_client
from agent import prompts


class LegalTools:
    """法律领域工具集"""
    
    def __init__(self, llm, deli_client: Optional[DeliLegalClient] = None):
        self.llm = llm
        self.deli_client = deli_client or create_deli_client()
        self._langchain_tools = None
    
    def search_law(self, query: str) -> str:
        """
        搜索相关法律法规
        
        Args:
            query: 法律查询问题
            
        Returns:
            相关的法律条文
        """
        result = self.deli_client.search_law(keywords=query, page_size=5)
        
        if not result.get("success"):
            return f"法规检索失败: {result.get('error', '未知错误')}"
        
        laws = result.get("body", {}).get("list", [])
        if not laws:
            return "[外部数据: 未找到] 未能检索到相关法规"
        
        return self.deli_client.format_law_results(result)
    
    def get_law_detail(self, law_id: str) -> str:
        """
        获取法规详情
        
        Args:
            law_id: 法规ID
            
        Returns:
            法规详细内容
        """
        result = self.deli_client.get_law_detail(law_id)
        
        if not result.get("success"):
            return f"获取法规详情失败: {result.get('error', '未知错误')}"
        
        body = result.get("body", {})
        if not body:
            return "未找到该法规详情"
        
        title = body.get("title", "")
        level = body.get("levelName", "")
        publisher = body.get("publisherName", "")
        date = body.get("publishDate", "")
        content = body.get("lawDetailContent", "")
        
        lines = [f"【{title}】"]
        if level:
            lines.append(f"类型: {level}")
        if publisher:
            lines.append(f"发布: {publisher}")
        if date:
            lines.append(f"日期: {date}")
        lines.append(f"\n{content}")
        
        return "\n".join(lines)
    
    def search_case(self, query: str) -> str:
        """
        检索类似案例
        
        Args:
            query: 案件相关描述
            
        Returns:
            类似案例及判决结果
        """
        result = self.deli_client.search_case(keywords=query, page_size=5)
        
        if not result.get("success"):
            return f"案例检索失败: {result.get('error', '未知错误')}"
        
        cases = result.get("body", {}).get("list", [])
        if not cases:
            return "[外部数据: 未找到] 未能检索到相关案例"
        
        return self.deli_client.format_case_results(result)
    
    def review_contract(self, contract_text: str) -> str:
        """
        审查合同，识别风险
        
        Args:
            contract_text: 合同文本内容
            
        Returns:
            合同审查结果和风险提示
        """
        prompt = prompts.format_contract_review_prompt(contract_text)
        
        try:
            result = self.llm.invoke(prompt)
            if hasattr(result, "content"):
                return result.content
            return str(result)
        except Exception as e:
            return f"合同审查失败: {str(e)}"
    
    def generate_document(self, doctype: str, facts: str, legal_basis: str = "") -> str:
        """
        生成法律文书
        
        Args:
            doctype: 文书类型（起诉状/答辩状/仲裁申请书/劳动合同/协议）
            facts: 事实情况描述
            legal_basis: 相关法律条文（可选）
            
        Returns:
            生成的法律文书
        """
        prompt = prompts.format_document_prompt(doctype, facts, legal_basis)
        
        try:
            result = self.llm.invoke(prompt)
            if hasattr(result, "content"):
                return result.content
            return str(result)
        except Exception as e:
            return f"文书生成失败: {str(e)}"
    
    def assess_risk(self, case_type: str, facts: str) -> str:
        """
        评估诉讼风险
        
        Args:
            case_type: 案件类型（如劳动争议、合同纠纷等）
            facts: 案件事实描述
            
        Returns:
            风险评估和建议
        """
        case_context = self.search_case(facts[:200])
        prompt = prompts.format_risk_assessment_prompt(case_type, facts, case_context)
        
        try:
            result = self.llm.invoke(prompt)
            if hasattr(result, "content"):
                return result.content
            return str(result)
        except Exception as e:
            return f"风险评估失败: {str(e)}"
    
    def get_procedure_guide(self, procedure_type: str) -> str:
        """
        获取法律流程指引
        
        Args:
            procedure_type: 流程类型（劳动仲裁/诉讼/工伤认定/法律援助等）
            
        Returns:
            流程步骤和注意事项
        """
        return prompts.get_procedure_guide(procedure_type)
    
    def get_langchain_tools(self) -> List[Tool]:
        """获取 LangChain Tool 列表"""
        if self._langchain_tools is not None:
            return self._langchain_tools
        
        self._langchain_tools = [
            Tool(
                name="search_law",
                func=self.search_law,
                description="""搜索相关法律法规。当你需要回答法律问题、查找法律条文时使用此工具。
                输入：法律问题或关键词，如"劳动合同试用期"、"工伤认定"等。
                输出：相关法律条文，如果返回"未找到"则表示外部API无数据。"""
            ),
            Tool(
                name="search_case",
                func=self.search_case,
                description="""检索类似案例。当你需要查找相关案例、了解类似案件判决结果时使用此工具。
                输入：案件描述或关键词，如"工伤赔偿"、"劳动纠纷"等。
                输出：类似案例列表，如果返回"未找到"则表示外部API无数据。"""
            ),
            Tool(
                name="get_procedure_guide",
                func=self.get_procedure_guide,
                description="""获取法律流程指引。当用户询问仲裁流程、诉讼流程、工伤认定流程时使用。
                输入：流程类型，如"劳动仲裁"、"工伤认定"、"民事诉讼"、"法律援助"等。
                输出：详细的流程步骤和注意事项。"""
            ),
            Tool(
                name="review_contract",
                func=self.review_contract,
                description="""审查劳动合同，识别风险条款。
                输入：完整的合同文本内容。
                输出：合同审查报告，包括风险条款和修改建议。"""
            ),
            Tool(
                name="generate_document",
                func=lambda x: self._parse_document_request(x),
                description="""生成法律文书。根据用户提供的事实情况生成法律文书。
                输入格式：文书类型|事实情况|[法律依据]
                例如：劳动仲裁申请书|申请人2024年1月入职公司，未签订劳动合同|劳动合同法
                支持的文书类型：劳动仲裁申请书、民事起诉状、答辩状、借条、欠条、劳动合同等。"""
            ),
            Tool(
                name="assess_risk",
                func=lambda x: self._parse_risk_request(x),
                description="""评估诉讼风险。当用户询问案件胜诉概率、诉讼风险时使用。
                输入格式：案件类型|案件事实
                例如：劳动争议|公司拖欠工资3个月，未签订劳动合同
                输出：风险评估报告，包括胜诉概率、有利因素、不利因素等。"""
            ),
        ]
        
        return self._langchain_tools
    
    def _parse_document_request(self, input_str: str) -> str:
        """解析文书生成请求"""
        parts = input_str.split("|")
        if len(parts) >= 2:
            doctype = parts[0].strip()
            facts = parts[1].strip()
            legal_basis = parts[2].strip() if len(parts) > 2 else ""
            return self.generate_document(doctype, facts, legal_basis)
        return "请提供文书类型和事实情况，格式：文书类型|事实情况|[法律依据]"
    
    def _parse_risk_request(self, input_str: str) -> str:
        """解析风险评估请求"""
        parts = input_str.split("|")
        if len(parts) >= 2:
            case_type = parts[0].strip()
            facts = parts[1].strip()
            return self.assess_risk(case_type, facts)
        return "请提供案件类型和事实情况，格式：案件类型|案件事实"


def create_tools(llm, deli_client: Optional[DeliLegalClient] = None) -> Dict:
    """
    创建工具实例
    
    Args:
        llm: 语言模型实例
        deli_client: 得理API客户端（可选）
        
    Returns:
        工具字典
    """
    tools_instance = LegalTools(llm, deli_client)
    
    return {
        "search_law": tools_instance.search_law,
        "get_law_detail": tools_instance.get_law_detail,
        "search_case": tools_instance.search_case,
        "review_contract": tools_instance.review_contract,
        "generate_document": tools_instance.generate_document,
        "assess_risk": tools_instance.assess_risk,
        "get_procedure_guide": tools_instance.get_procedure_guide,
        "get_langchain_tools": tools_instance.get_langchain_tools,
    }


def create_langchain_agent(llm, tools: List[Tool], system_prompt: str, verbose: bool = False):
    """
    创建 LangChain Agent
    
    Args:
        llm: 语言模型
        tools: LangChain Tool 列表
        system_prompt: 系统提示词
        verbose: 是否显示详细日志
        
    Returns:
        配置好的 Agent
    """
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL,
        system_message=system_prompt,
        verbose=verbose,
        max_iterations=5,
        handle_parsing_errors=True
    )
    return agent