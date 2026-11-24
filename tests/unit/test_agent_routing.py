"""Unit tests for Agent Task Routing System (Phase 3).
"""

import json
from pathlib import Path
import pytest

from personal_ai.agents.specialists import AgentRegistry
from personal_ai.routing.router import TaskRouter


@pytest.fixture
def test_router(tmp_path: Path) -> TaskRouter:
    """Provides a TaskRouter instance with a temporary log file."""
    log_file = tmp_path / "test_routing.jsonl"
    registry = AgentRegistry(agents_dir="configs/agents")
    return TaskRouter(registry=registry, log_path=log_file)


def test_agent_registry_loads_all_four_specialists():
    """Verify all 4 required agents are loaded with correct properties."""
    registry = AgentRegistry(agents_dir="configs/agents")
    loaded = registry.list_agents()

    assert "deep_learning" in loaded
    assert "computer_vision_uav" in loaded
    assert "mathematics_engineering" in loaded
    assert "software_engineering" in loaded

    # Verify tool permission separation
    cv_agent = registry.get_agent("computer_vision_uav")
    sw_agent = registry.get_agent("software_engineering")
    assert "trajectory_calculator" in cv_agent.allowed_tools
    assert "code_linter" in sw_agent.allowed_tools
    assert "code_linter" not in cv_agent.allowed_tools


def test_route_deep_learning_query(test_router: TaskRouter):
    """Verify deep learning query routes accurately to deep_learning agent."""
    query = "Transformer mimarisinde multi-head attention ve backpropagation gradyan akışı nasıl hesaplanır?"
    decision = test_router.route(query)

    assert decision.selected_agent == "deep_learning"
    assert decision.confidence >= 0.50
    assert decision.needs_clarification is False
    assert "attention" in decision.matched_keywords["deep_learning"] or "backpropagation" in decision.matched_keywords["deep_learning"]


def test_route_computer_vision_uav_query(test_router: TaskRouter):
    """Verify UAV and vision query routes accurately to computer_vision_uav agent."""
    query = "İHA üzerinde YOLOv8 ve ByteTrack ile gerçek zamanlı nesne tespiti ve yörünge takibi nasıl optimize edilir?"
    decision = test_router.route(query)

    assert decision.selected_agent == "computer_vision_uav"
    assert decision.confidence >= 0.50
    assert decision.needs_clarification is False


def test_route_mathematics_engineering_query(test_router: TaskRouter):
    """Verify math and control query routes accurately to mathematics_engineering agent."""
    query = "Durum uzayı modelinde Kalman filtresinin kovaryans matrisi güncellemesi ve özdeğer kararlılığı nasıldır?"
    decision = test_router.route(query)

    assert decision.selected_agent == "mathematics_engineering"
    assert decision.confidence >= 0.50
    assert decision.needs_clarification is False


def test_route_software_engineering_query(test_router: TaskRouter):
    """Verify software architecture and debugging query routes to software_engineering agent."""
    query = "Modern C++ kodunda smart pointers ile bellek sızıntısı (memory leak) ve thread deadlock nasıl önlenir?"
    decision = test_router.route(query)

    assert decision.selected_agent == "software_engineering"
    assert decision.confidence >= 0.50
    assert decision.needs_clarification is False


def test_route_unrelated_query_requires_clarification(test_router: TaskRouter):
    """Verify out-of-domain query does not pick a random agent, but requests clarification."""
    query = "Akdeniz ikliminde zeytin ağaçları hangi ayda budanmalıdır?"
    decision = test_router.route(query)

    assert decision.selected_agent is None
    assert decision.confidence == 0.0
    assert decision.needs_clarification is True
    assert decision.clarification_question is not None
    assert "Derin Öğrenme" in decision.clarification_question


def test_routing_decision_logged_to_file(test_router: TaskRouter):
    """Verify routing decisions are logged as valid JSONL entries."""
    query = "PyTorch ile AdamW optimizer ve learning rate scheduling."
    test_router.route(query)

    assert test_router.log_path.exists()
    lines = test_router.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1

    last_record = json.loads(lines[-1])
    assert last_record["query"] == query
    assert last_record["selected_agent"] == "deep_learning"
    assert "timestamp" in last_record


def test_route_option_selection(test_router: TaskRouter):
    """Verify router cleanly parses option selection numbers and choice phrases."""
    d1 = test_router.route("1.bakış açısı")
    assert d1.selected_agent == "deep_learning"
    assert d1.needs_clarification is False

    d2 = test_router.route("2")
    assert d2.selected_agent == "computer_vision_uav"
    assert d2.needs_clarification is False

    d3 = test_router.route("3. seçenek")
    assert d3.selected_agent == "mathematics_engineering"
    assert d3.needs_clarification is False

    d4 = test_router.route("4")
    assert d4.selected_agent == "software_engineering"
    assert d4.needs_clarification is False


def test_route_greeting_does_not_require_clarification(test_router: TaskRouter):
    """Verify general polite greetings are routed without annoying clarification quiz."""
    d = test_router.route("merhaba")
    assert d.selected_agent == "deep_learning"
    assert d.needs_clarification is False
