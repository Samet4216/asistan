"""High-level Orchestrator coordinating Router, Specialist Agents, RAG, and LLM (Phase 3).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from personal_ai.agents.base import BaseSpecialistAgent
from personal_ai.agents.specialists import AgentRegistry
from personal_ai.rag.retriever import DocumentRetriever, RetrievalResult
from personal_ai.routing.router import RoutingDecision, TaskRouter
from personal_ai.runtime.ollama_client import OllamaClient


@dataclass
class OrchestrationEvent:
    """Carries execution state, streaming tokens, or final metadata from the orchestrator."""
    event_type: str  # "clarification", "routing", "context", "token", "done"
    content: str
    metadata: Dict[str, Any]


class AgentOrchestrator:
    """Manages the lifecycle of user requests across specialist agents and retrieval."""

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        router: Optional[TaskRouter] = None,
        retriever: Optional[DocumentRetriever] = None,
        client: Optional[OllamaClient] = None,
    ):
        self.registry = registry or AgentRegistry()
        self.router = router or TaskRouter(registry=self.registry)
        self.retriever = retriever or DocumentRetriever()
        self.client = client or OllamaClient()

    def handle_query_stream(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        force_agent_id: Optional[str] = None,
        enable_rag: bool = True,
    ) -> Generator[OrchestrationEvent, None, None]:
        """Dispatches query, resolves RAG, and streams response tokens from the selected specialist."""
        # 1. Routing phase
        selected_agent: Optional[BaseSpecialistAgent] = None

        if force_agent_id:
            selected_agent = self.registry.get_agent(force_agent_id)
            if not selected_agent:
                yield OrchestrationEvent(
                    event_type="error",
                    content=f"İstenen uzman ajan '{force_agent_id}' bulunamadı. Mevcut ajanlar: {', '.join(self.registry.list_agents())}",
                    metadata={},
                )
                return
            yield OrchestrationEvent(
                event_type="routing",
                content=f"Kullanıcı tarafından manuel olarak seçildi: {selected_agent.name}",
                metadata={"agent_id": selected_agent.agent_id, "confidence": 1.0, "manual": True},
            )
        else:
            decision = self.router.route(query)

            if decision.needs_clarification or not decision.selected_agent:
                yield OrchestrationEvent(
                    event_type="clarification",
                    content=decision.clarification_question or "Soru belirsiz, lütfen netleştiriniz.",
                    metadata={"reasoning": decision.reasoning, "scores": decision.scores},
                )
                return

            selected_agent = self.registry.get_agent(decision.selected_agent)
            if not selected_agent:
                selected_agent = next(iter(self.registry.all_agents.values()))

            yield OrchestrationEvent(
                event_type="routing",
                content=f"Seçilen Uzman: {selected_agent.name} (Güven: %{decision.confidence * 100:.0f})",
                metadata={
                    "agent_id": selected_agent.agent_id,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning,
                },
            )

        # 2. RAG Retrieval phase
        augmented_query = query
        sources_summary: List[str] = []

        if enable_rag and self.retriever:
            retrieval = self.retriever.retrieve(
                query=query,
                collection_name=selected_agent.rag_collection,
            )
            # If agent's specific collection is empty, fall back to default vault
            if not retrieval.has_context:
                retrieval = self.retriever.retrieve(
                    query=query,
                    collection_name=self.retriever.default_collection,
                )

            if retrieval.has_context:
                sources_summary = retrieval.sources_summary
                augmented_query = self.retriever.build_rag_prompt(query, retrieval)
                yield OrchestrationEvent(
                    event_type="context",
                    content=f"Belgelerden {len(retrieval.chunks)} ilgili bölüm bağlama eklendi.",
                    metadata={"sources": sources_summary},
                )
            else:
                yield OrchestrationEvent(
                    event_type="context",
                    content="Kişisel dokümanlarda doğrudan eşleşme bulunamadı; uzman temel bilgisiyle yanıtlıyor.",
                    metadata={"sources": []},
                )

        # 3. Model Inference phase
        # Formulate messages with the augmented prompt
        messages = list(conversation_history) + [{"role": "user", "content": augmented_query}]

        full_response_parts = []
        for token in selected_agent.execute_stream(messages, self.client):
            full_response_parts.append(token)
            yield OrchestrationEvent(
                event_type="token",
                content=token,
                metadata={},
            )

        yield OrchestrationEvent(
            event_type="done",
            content="".join(full_response_parts),
            metadata={
                "agent_id": selected_agent.agent_id,
                "agent_name": selected_agent.name,
                "sources": sources_summary,
            },
        )
