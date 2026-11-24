"""Concrete specialist agent implementations and registry (Phase 3).
"""

from pathlib import Path
from typing import Dict, List, Optional

from personal_ai.agents.base import BaseSpecialistAgent


class DeepLearningAgent(BaseSpecialistAgent):
    """Specialist agent for neural architectures, deep learning theory, and PyTorch."""
    def __init__(self, config_path: str | Path = "configs/agents/deep_learning.yaml"):
        super().__init__(config_path)


class ComputerVisionUAVAgent(BaseSpecialistAgent):
    """Specialist agent for real-time vision (YOLO/SAHI), tracking, and UAV flight systems."""
    def __init__(self, config_path: str | Path = "configs/agents/computer_vision_uav.yaml"):
        super().__init__(config_path)


class MathematicsEngineeringAgent(BaseSpecialistAgent):
    """Specialist agent for linear algebra, calculus, ODEs, Kalman filters, and control theory."""
    def __init__(self, config_path: str | Path = "configs/agents/mathematics_engineering.yaml"):
        super().__init__(config_path)


class SoftwareEngineeringAgent(BaseSpecialistAgent):
    """Specialist agent for C++, Python, algorithms, architecture, and testing."""
    def __init__(self, config_path: str | Path = "configs/agents/software_engineering.yaml"):
        super().__init__(config_path)


class AgentRegistry:
    """Central repository managing loaded specialist agents."""

    def __init__(self, agents_dir: str | Path = "configs/agents"):
        self.agents_dir = Path(agents_dir)
        self._agents: Dict[str, BaseSpecialistAgent] = {}
        self._load_registered_agents()

    def _load_registered_agents(self):
        """Discovers and registers all specialist agents."""
        registry_map = {
            "deep_learning": DeepLearningAgent,
            "computer_vision_uav": ComputerVisionUAVAgent,
            "mathematics_engineering": MathematicsEngineeringAgent,
            "software_engineering": SoftwareEngineeringAgent,
        }

        for agent_id, agent_cls in registry_map.items():
            cfg_path = self.agents_dir / f"{agent_id}.yaml"
            if cfg_path.exists():
                self._agents[agent_id] = agent_cls(cfg_path)

    def get_agent(self, agent_id: str) -> Optional[BaseSpecialistAgent]:
        """Retrieves an agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[str]:
        """Returns list of registered agent IDs."""
        return list(self._agents.keys())

    @property
    def all_agents(self) -> Dict[str, BaseSpecialistAgent]:
        """Returns all loaded agents."""
        return self._agents
