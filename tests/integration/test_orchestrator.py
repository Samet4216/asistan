"""Integration tests for AgentOrchestrator (Phase 3).
"""

from pathlib import Path
from unittest.mock import MagicMock

from personal_ai.agents.specialists import AgentRegistry
from personal_ai.routing.orchestrator import AgentOrchestrator
from personal_ai.routing.router import TaskRouter
from personal_ai.runtime.ollama_client import OllamaClient


def test_orchestrator_routes_and_streams_response(tmp_path: Path):
    """Verify end-to-end orchestration flow with specialist agent and streaming."""
    registry = AgentRegistry(agents_dir="configs/agents")
    router = TaskRouter(registry=registry, log_path=tmp_path / "routing.jsonl")

    # Mock OllamaClient
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.chat_stream.return_value = ["Euler ", "açıları ", "türetimi."]

    # Mock Retriever
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value.has_context = False

    orchestrator = AgentOrchestrator(
        registry=registry,
        router=router,
        retriever=mock_retriever,
        client=mock_client,
    )

    query = "İHA 6-DoF uçuş dinamiği ve Euler açıları"
    events = list(orchestrator.handle_query_stream(query=query, conversation_history=[]))

    event_types = [e.event_type for e in events]
    assert "routing" in event_types
    assert "context" in event_types
    assert "token" in event_types
    assert "done" in event_types

    # Selected agent should be computer_vision_uav
    routing_event = next(e for e in events if e.event_type == "routing")
    assert routing_event.metadata["agent_id"] == "computer_vision_uav"

    # Verify tokens yielded
    token_contents = [e.content for e in events if e.event_type == "token"]
    assert "".join(token_contents) == "Euler açıları türetimi."


def test_orchestrator_handles_clarification(tmp_path: Path):
    """Verify orchestrator yields clarification event on ambiguous query without querying LLM."""
    registry = AgentRegistry(agents_dir="configs/agents")
    router = TaskRouter(registry=registry, log_path=tmp_path / "routing.jsonl")

    mock_client = MagicMock(spec=OllamaClient)
    orchestrator = AgentOrchestrator(
        registry=registry,
        router=router,
        client=mock_client,
    )

    query = "Hava bugün nasıl?"
    events = list(orchestrator.handle_query_stream(query=query, conversation_history=[]))

    assert len(events) == 1
    assert events[0].event_type == "clarification"
    # LLM should NOT be called on ambiguous query!
    mock_client.chat_stream.assert_not_called()
