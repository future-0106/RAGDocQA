"""
Agent模块 - 法律多Agent系统
"""
from agent.delilegal_client import DeliLegalClient, create_deli_client
from agent.tools import LegalTools, create_tools
from agent.agents import (
    LegalConsultationAgent,
    ContractReviewAgent,
    DocumentGeneratorAgent,
    RiskAssessmentAgent,
    create_agent
)
from agent.coordinator import AgentCoordinator

__all__ = [
    "DeliLegalClient",
    "create_deli_client",
    "LegalTools",
    "create_tools",
    "LegalConsultationAgent",
    "ContractReviewAgent",
    "DocumentGeneratorAgent",
    "RiskAssessmentAgent",
    "create_agent",
    "AgentCoordinator"
]