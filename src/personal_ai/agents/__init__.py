"""Specialist Agents package.
"""

from personal_ai.agents.base import BaseSpecialistAgent, AgentMetadata
from personal_ai.agents.specialists import (
    DeepLearningAgent,
    ComputerVisionUAVAgent,
    MathematicsEngineeringAgent,
    SoftwareEngineeringAgent,
    AgentRegistry,
)

__all__ = [
    "BaseSpecialistAgent",
    "AgentMetadata",
    "DeepLearningAgent",
    "ComputerVisionUAVAgent",
    "MathematicsEngineeringAgent",
    "SoftwareEngineeringAgent",
    "AgentRegistry",
]
