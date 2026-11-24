"""Rule-based and semantic Task Router for multi-agent dispatch (Phase 3).
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from personal_ai.agents.specialists import AgentRegistry


@dataclass
class RoutingDecision:
    """Encapsulates the router's analysis, selected specialist, and confidence."""
    query: str
    selected_agent: Optional[str]
    confidence: float
    reasoning: str
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    matched_keywords: Dict[str, List[str]] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)


class TaskRouter:
    """Routes user queries to the most suitable domain specialist agent."""

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        log_path: str | Path = "logs/routing.jsonl",
        min_confidence_threshold: float = 0.35,
        ambiguity_margin: float = 0.20,
    ):
        self.registry = registry or AgentRegistry()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_confidence_threshold = min_confidence_threshold
        self.ambiguity_margin = ambiguity_margin

    def route(self, query: str) -> RoutingDecision:
        """Analyzes the user query and selects the best specialist agent."""
        norm_query = query.lower().strip()

        # 0. Check for explicit option / clarification response selection
        stripped = norm_query.rstrip(".").strip()
        option_match = re.match(r"^([1-4])(?:\.|\.bakış|\s*bakış|\s*seçenek|\))?.*$", stripped)
        if option_match:
            choice = option_match.group(1)
            agent_map = {
                "1": "deep_learning",
                "2": "computer_vision_uav",
                "3": "mathematics_engineering",
                "4": "software_engineering",
            }
            chosen = agent_map[choice]
            decision = RoutingDecision(
                query=query,
                selected_agent=chosen,
                confidence=1.0,
                reasoning=f"Kullanıcı netleştirme seçimi olarak {choice}. seçeneği ({chosen}) seçti.",
                needs_clarification=False,
                scores={chosen: 1.0},
            )
            self._log_decision(decision)
            return decision

        # 0b. Check for general conversational greetings
        greetings = {"merhaba", "selam", "selamlar", "günaydın", "iyi günler", "hey", "nasılsın"}
        if stripped in greetings or any(stripped.startswith(g) for g in greetings if len(stripped) <= len(g) + 3):
            decision = RoutingDecision(
                query=query,
                selected_agent="deep_learning",
                confidence=0.9,
                reasoning="Genel selamlama/sohbet girişi; birincil mühendislik asistanı yanıtlıyor.",
                needs_clarification=False,
                scores={"deep_learning": 1.0},
            )
            self._log_decision(decision)
            return decision

        scores: Dict[str, float] = {}
        matched_keywords: Dict[str, List[str]] = {}

        # 1. Score each agent based on keyword occurrences
        for agent_id, agent in self.registry.all_agents.items():
            agent_score = 0.0
            agent_matches: List[str] = []

            for kw in agent.keywords:
                kw_lower = kw.lower()
                # Check exact phrase or word boundary match
                pattern = r"(?:\b|_)" + re.escape(kw_lower) + r"(?:\b|_)"
                matches = re.findall(pattern, norm_query)
                if matches:
                    # Longer phrases receive higher weight
                    weight = 2.0 if len(kw_lower.split()) > 1 or len(kw_lower) > 8 else 1.0
                    agent_score += len(matches) * weight
                    agent_matches.append(kw_lower)

            scores[agent_id] = agent_score
            matched_keywords[agent_id] = agent_matches

        total_score = sum(scores.values())

        # Case A: No keywords matched -> Ambiguous / Out of domain
        if total_score == 0:
            decision = RoutingDecision(
                query=query,
                selected_agent=None,
                confidence=0.0,
                reasoning="Sorguda 4 uzmanlık alanına (DL, CV/İHA, Matematik, Yazılım) ait doğrudan bir anahtar kavram tespit edilemedi.",
                needs_clarification=True,
                clarification_question=(
                    "Bu soruyu hangi uzmanlık bakış açısıyla ele almamı istersiniz?\n"
                    "1) Derin Öğrenme / Sinir Ağları Araştırması\n"
                    "2) Bilgisayarlı Görü & İHA Uçuş Sistemleri\n"
                    "3) Uygulamalı Matematik & Kontrol Teorisi\n"
                    "4) Yazılım Mimarisi & Kod Analizi"
                ),
                matched_keywords=matched_keywords,
                scores=scores,
            )
            self._log_decision(decision)
            return decision

        # Sort agents by score descending
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_agent, top_score = sorted_scores[0]
        second_agent, second_score = sorted_scores[1] if len(sorted_scores) > 1 else (None, 0.0)

        confidence = top_score / total_score

        # Case B: Ambiguity / Tie between two top agents
        if second_score > 0 and (top_score - second_score) / top_score < self.ambiguity_margin:
            agent_top_name = self.registry.get_agent(top_agent).name if self.registry.get_agent(top_agent) else top_agent
            agent_sec_name = self.registry.get_agent(second_agent).name if self.registry.get_agent(second_agent) else second_agent

            decision = RoutingDecision(
                query=query,
                selected_agent=None,
                confidence=confidence,
                reasoning=f"Soru hem '{top_agent}' ({top_score:.1f}) hem de '{second_agent}' ({second_score:.1f}) alanlarıyla güçlü kesişim gösteriyor.",
                needs_clarification=True,
                clarification_question=(
                    f"Bu soruyu ağırlıklı olarak '{agent_top_name}' kapsamında mı yoksa "
                    f"'{agent_sec_name}' kapsamında mı detaylandırmamı tercih edersiniz?"
                ),
                matched_keywords=matched_keywords,
                scores=scores,
            )
            self._log_decision(decision)
            return decision

        # Case C: Clear, dominant specialist agent
        decision = RoutingDecision(
            query=query,
            selected_agent=top_agent,
            confidence=confidence,
            reasoning=f"'{top_agent}' uzmanlığı en yüksek puanı aldı (skor: {top_score:.1f}, güven: %{confidence*100:.0f}). Eşleşen kavramlar: {', '.join(matched_keywords[top_agent])}",
            needs_clarification=False,
            clarification_question=None,
            matched_keywords=matched_keywords,
            scores=scores,
        )
        self._log_decision(decision)
        return decision

    def _log_decision(self, decision: RoutingDecision):
        """Appends routing decisions to a JSONL log file for tracking and measuring accuracy."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": decision.query,
                "selected_agent": decision.selected_agent,
                "confidence": round(decision.confidence, 4),
                "needs_clarification": decision.needs_clarification,
                "reasoning": decision.reasoning,
                "scores": decision.scores,
                "matched_keywords": {k: v for k, v in decision.matched_keywords.items() if v},
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            # Logging failures should never crash application execution
            pass
