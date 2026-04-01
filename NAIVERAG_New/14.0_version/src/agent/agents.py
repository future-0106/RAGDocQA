"""
法律Agent定义
"""
from typing import Dict, Any


class LegalAgentBase:
    """法律Agent基类"""
    
    @staticmethod
    def create_system_prompt(role: str, goal: str, backstory: str) -> str:
        """创建系统提示词"""
        return f"""你是{role}。

## 目标
{goal}

## 背景
{backstory}

## 重要提示
- 必须基于事实回答，不能编造
- 使用工具获取最新信息
- 回答要专业、准确
"""


class LegalConsultationAgent:
    """法律咨询Agent - 面向普通民众的法律问题解答"""
    
    @staticmethod
    def create(llm, tools: Dict):
        role = "法律咨询顾问"
        goal = "用通俗易懂的语言解答用户的法律问题，帮助他们了解自己的权利和应对方法"
        backstory = """你是资深法律咨询专家，拥有10年以上法律咨询经验。
        你擅长用简单直白的语言解释复杂的法律问题，让普通民众也能听得懂。
        你总是站在用户角度，为他们提供切实可行的建议。"""
        
        system_prompt = LegalAgentBase.create_system_prompt(role, goal, backstory)
        
        return {
            "role": role,
            "goal": goal,
            "backstory": backstory,
            "system_prompt": system_prompt,
            "tools": tools,
            "llm": llm,
            "agent_type": "consultation"
        }


class ContractReviewAgent:
    """合同审查Agent - 识别合同风险"""
    
    @staticmethod
    def create(llm, tools: Dict):
        role = "合同审查专家"
        goal = "识别合同中的风险条款并提供具体的修改建议，帮助用户规避法律风险"
        backstory = """你是执业15年以上的资深律师，精通劳动法、合同法、公司法。
        你审阅过上万份劳动合同，擅长识别各种隐藏的风险条款。
        你不仅指出问题，还会提供具体可操作的修改建议。"""
        
        system_prompt = LegalAgentBase.create_system_prompt(role, goal, backstory)
        
        return {
            "role": role,
            "goal": goal,
            "backstory": backstory,
            "system_prompt": system_prompt,
            "tools": tools,
            "llm": llm,
            "agent_type": "contract_review"
        }


class DocumentGeneratorAgent:
    """文书生成Agent - 生成各类法律文书"""
    
    @staticmethod
    def create(llm, tools: Dict):
        role = "法律文书起草专家"
        goal = "根据用户提供的facts生成规范、完整、有针对性的法律文书"
        backstory = """你是法律文书写作专家，专门从事法律文书起草工作20年。
        你精通各类法律文书的格式规范和写作技巧，能够根据不同案情
        起草准确、完整、有说服力的法律文书。"""
        
        system_prompt = LegalAgentBase.create_system_prompt(role, goal, backstory)
        
        return {
            "role": role,
            "goal": goal,
            "backstory": backstory,
            "system_prompt": system_prompt,
            "tools": tools,
            "llm": llm,
            "agent_type": "document_generation"
        }


class RiskAssessmentAgent:
    """风险评估Agent - 诉讼风险评估"""
    
    @staticmethod
    def create(llm, tools: Dict):
        role = "诉讼风险评估专家"
        goal = "客观评估用户的诉讼风险，提供专业的应对建议"
        backstory = """你是资深法律顾问，擅长诉讼策略制定和风险评估。
        你曾帮助上千当事人分析案件走向，评估胜诉概率，制定诉讼方案。
        你总是给出客观、理性、实事求是的分析。"""
        
        system_prompt = LegalAgentBase.create_system_prompt(role, goal, backstory)
        
        return {
            "role": role,
            "goal": goal,
            "backstory": backstory,
            "system_prompt": system_prompt,
            "tools": tools,
            "llm": llm,
            "agent_type": "risk_assessment"
        }


def create_agent(agent_type: str, llm, tools: Dict) -> Dict:
    """创建指定类型的Agent"""
    agent_creators = {
        "consultation": LegalConsultationAgent,
        "contract_review": ContractReviewAgent,
        "document_generation": DocumentGeneratorAgent,
        "risk_assessment": RiskAssessmentAgent,
    }
    
    creator = agent_creators.get(agent_type)
    if not creator:
        raise ValueError(f"未知的Agent类型: {agent_type}")
    
    return creator.create(llm, tools)