"""Base interface for domain-specialist agents (Phase 3).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
import yaml

from personal_ai.runtime.ollama_client import OllamaClient


@dataclass
class AgentMetadata:
    """Stores declarative configuration for a specialist agent."""
    agent_id: str
    name: str
    domain: str
    system_prompt: str
    rag_collection: str
    allowed_tools: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


class BaseSpecialistAgent(ABC):
    """Abstract base class for all engineering specialist agents."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.metadata = self._load_agent_config()

    def _load_agent_config(self) -> AgentMetadata:
        """Loads agent YAML configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Agent config file not found: {self.config_path}")

        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        agent_data = data.get("agent", {})

        return AgentMetadata(
            agent_id=agent_data.get("id", "general"),
            name=agent_data.get("name", "General Assistant"),
            domain=agent_data.get("domain", "general"),
            system_prompt=agent_data.get("system_prompt", "").strip(),
            rag_collection=agent_data.get("rag_collection", "engineering_vault"),
            allowed_tools=agent_data.get("allowed_tools", []),
            keywords=agent_data.get("keywords", []),
        )

    @property
    def agent_id(self) -> str:
        return self.metadata.agent_id

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def domain(self) -> str:
        return self.metadata.domain

    @property
    def system_prompt(self) -> str:
        return self.metadata.system_prompt

    @property
    def rag_collection(self) -> str:
        return self.metadata.rag_collection

    @property
    def allowed_tools(self) -> List[str]:
        return self.metadata.allowed_tools

    @property
    def keywords(self) -> List[str]:
        return self.metadata.keywords

    def execute_stream(
        self,
        messages: List[Dict[str, str]],
        client: OllamaClient,
    ) -> Generator[str, None, None]:
        """Streams model response using the specialist's system prompt."""
        return client.chat_stream(
            messages=messages,
            system_prompt=self.system_prompt,
        )

    def execute(
        self,
        messages: List[Dict[str, str]],
        client: OllamaClient,
    ) -> str:
        """Executes full non-streaming model response."""
        return client.chat(
            messages=messages,
            system_prompt=self.system_prompt,
        )
