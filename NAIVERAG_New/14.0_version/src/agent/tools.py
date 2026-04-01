"""
法律Agent工具集
基于得理开放平台API实现检索功能
"""
from typing import Dict, List, Optional
from agent.delilegal_client import DeliLegalClient, create_deli_client
from agent import prompts


class LegalTools:
    """法律领域工具集"""
    
    def __init__(self, llm, deli_client: Optional[DeliLegalClient] = None):
        self.llm = llm
        self.deli_client = deli_client or create_deli_client()
    
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
        case_context = self.search_case(facts)
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
    }