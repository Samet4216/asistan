"""
Challenge Mode / Devil's Advocate Module (Faz 12).
Evaluates architectural hypotheses and experimental claims critically.
Searches paper limitations, detects memory-bandwidth bottlenecks (split/permute),
and proposes rigorous ablation tests to disprove invalid ideas before training runs.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import llm
import budget
from retrieval.search import searcher
from answer.verify import verify_quotes
from memory.errors import error_manager


CHALLENGE_SYSTEM_PROMPT = """You are a rigorous, skeptical Principal Deep Learning Researcher and critical reviewer.
Your objective is to stress-test the user's hypothesis or architectural idea.
DO NOT be agreeable or sycophantic. Scrutinize assumptions, find hidden trade-offs, and surface failure modes.

MANDATORY RULES:
1. GROUNDED CRITIQUE: Ground your skepticism in retrieved empirical evidence or mathematical complexity.
2. HARDWARE REALITY: Evaluate GPU bottlenecks specifically on the user's RTX 4050 6GB Laptop GPU (e.g. memory bandwidth overheads, split/cat/permute kernel launch overhead, VRAM limits).
3. FALSIFIABLE TEST: Always specify a minimal ablation experiment that would disprove the hypothesis with minimal compute.
4. STRICT ENGLISH: Output 100% technical English.
"""

CHALLENGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["REJECT", "HIGH_RISK", "PLAUSIBLE_WITH_CAVEATS"]},
        "summary": {"type": "string"},
        "criticisms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "vulnerability": {"type": "string"},
                    "evidence_or_rationale": {"type": "string"}
                },
                "required": ["vulnerability", "evidence_or_rationale"]
            }
        },
        "hardware_and_vram_risk": {"type": "string"},
        "minimal_disproving_experiment": {"type": "string"}
    },
    "required": [
        "verdict",
        "summary",
        "criticisms",
        "hardware_and_vram_risk",
        "minimal_disproving_experiment"
    ]
}


class ChallengeEngine:
    def __init__(self):
        self.searcher = searcher

    def challenge_hypothesis(self, hypothesis: str) -> Dict[str, Any]:
        """
        Executes critical peer-review evaluation on the given hypothesis.
        """
        # Step 1: Search limitations and experiments specifically
        search_res = self.searcher.search(
            query=f"limitations computational cost memory trade-off {hypothesis}",
            top_k=4
        )
        evidence = search_res.get("packed_context", "")

        # Step 2: Search past relevant errors
        relevant_errors = error_manager.get_relevant_errors(hypothesis, max_tokens=250)

        # Step 3: Build user prompt
        user_msg = (
            f"### [USER HYPOTHESIS / PROPOSAL]:\n{hypothesis}\n\n"
            f"### [RETRIEVED PAPER EVIDENCE & LIMITATIONS]:\n{evidence}\n\n"
            f"### [HISTORICAL FAILURE PATTERNS ON THIS HARDWARE]:\n{relevant_errors}\n\n"
            "Evaluate this hypothesis with extreme academic skepticism. Fill the required JSON schema."
        )

        messages = [
            {"role": "system", "content": CHALLENGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]

        raw_output = llm.generate(
            messages=messages,
            schema=CHALLENGE_SCHEMA,
            max_tokens=config.TOKEN_BUDGET.get("max_answer", 1000)
        )

        try:
            parsed = json.loads(raw_output)
        except Exception:
            parsed = {
                "verdict": "HIGH_RISK",
                "summary": raw_output[:300],
                "criticisms": [{"vulnerability": "Uncertainty", "evidence_or_rationale": raw_output}],
                "hardware_and_vram_risk": "Requires careful VRAM profiling on RTX 4050.",
                "minimal_disproving_experiment": "Run 5-epoch baseline comparison against standard architecture."
            }

        return parsed


challenger = ChallengeEngine()


if __name__ == "__main__":
    print("[*] Testing answer/challenge.py...")
    test_hypothesis = "We should replace all 3x3 Conv2D layers in MobileNet with Coordinate Attention blocks to achieve 100% spatial awareness and boost accuracy."
    result = challenger.challenge_hypothesis(test_hypothesis)

    print("\n" + "="*50)
    print(f"VERDICT: {result.get('verdict')}")
    print(f"SUMMARY: {result.get('summary')}\n")
    print("CRITICISMS & VULNERABILITIES:")
    for c in result.get("criticisms", []):
        print(f" * Vulnerability: {c.get('vulnerability')}")
        print(f"   Rationale: {c.get('evidence_or_rationale')}\n")
    print(f"HARDWARE & VRAM RISK:\n{result.get('hardware_and_vram_risk')}\n")
    print(f"DISPROVING EXPERIMENT:\n{result.get('minimal_disproving_experiment')}")
    print("="*50)
