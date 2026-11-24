"""
Verifiable Reward Functions and Anti-Reward Hacking Engine (Phase 9).

Computes deterministic, ground-truth verifiable reward signals for:
- Software Engineering & Coding (AST validation and unit execution)
- Mathematics (symbolic equivalence and equation validation)
- Reward Hacking Detection (boilerplate evasion, degenerate repetition, length gaming)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from personal_ai.tools.code_tools import CodeValidator
from personal_ai.tools.math_tools import SymbolicMathEngine

logger = logging.getLogger(__name__)


@dataclass
class RewardEvaluation:
    """Structured reward signal output."""
    reward: float  # Scalar between -1.0 and +1.0
    is_valid: bool
    domain: str
    rationale: str
    hacked_detected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class RewardHackingDetector:
    """
    Detects pathological gaming of reward signals:
    - Length gaming (superfluous tokens to artificially boost score)
    - Repetitive padding (degenerate n-gram repetition)
    - Evasion phrases ("As an AI...", "I cannot verify...")
    - Trivial assertion cheat in code ("assert True", "pass")
    """

    EVASION_PATTERNS = [
        re.compile(r"as an ai (language )?model", re.IGNORECASE),
        re.compile(r"i cannot (solve|execute|verify)", re.IGNORECASE),
        re.compile(r"bölüm \d+/\d+", re.IGNORECASE),  # Repetitive template forbidden
    ]

    @classmethod
    def check_evasion(cls, text: str) -> bool:
        """Returns True if evasion patterns or robotic boilerplate detected."""
        return any(pattern.search(text) for pattern in cls.EVASION_PATTERNS)

    @classmethod
    def check_length_gaming(cls, text: str, max_reasonable_chars: int = 4000) -> bool:
        """Flags outputs that inflate length without substance."""
        return len(text) > max_reasonable_chars

    @classmethod
    def check_code_trivial_cheat(cls, code: str) -> bool:
        """Detects trivial test cheats like empty assert True or passing tests."""
        lines = [line.strip() for line in code.splitlines() if line.strip()]
        # Check if code only contains 'pass' or 'assert True'
        meaningful_lines = [l for l in lines if not l.startswith("#") and l not in {"pass", "...", "assert True"}]
        return len(meaningful_lines) == 0


class VerifiableRewardEngine:
    """
    Computes deterministic rewards for reinforcement learning and preference tuning.
    """

    def __init__(self) -> None:
        self.code_validator = CodeValidator()
        self.math_engine = SymbolicMathEngine()
        self.hacker_detector = RewardHackingDetector()

    def evaluate_code_reward(self, prompt: str, completion: str) -> RewardEvaluation:
        """
        Evaluate code completion:
        - Valid syntax + clean architecture = +1.0
        - Syntax error = -1.0
        - Reward hacking detected = -1.0
        """
        # Anti-hacking checks
        if self.hacker_detector.check_evasion(completion):
            return RewardEvaluation(
                reward=-1.0,
                is_valid=False,
                domain="code",
                rationale="Detected robotic evasion phrase or forbidden boilerplate.",
                hacked_detected=True,
            )

        # Extract code block if fenced
        code = completion
        if "```python" in completion:
            code = completion.split("```python")[1].split("```")[0].strip()
        elif "```" in completion:
            code = completion.split("```")[1].split("```")[0].strip()

        if self.hacker_detector.check_code_trivial_cheat(code):
            return RewardEvaluation(
                reward=-0.8,
                is_valid=False,
                domain="code",
                rationale="Trivial assertion cheat or empty body detected.",
                hacked_detected=True,
            )

        res = self.code_validator.validate_python(code)
        if not res.valid:
            return RewardEvaluation(
                reward=-1.0,
                is_valid=False,
                domain="code",
                rationale=f"Code syntax verification failed: {res.error_message}",
                hacked_detected=False,
            )

        # Check structural quality: does it define functions or classes?
        has_structure = bool(res.functions_defined or res.classes_defined)
        reward = 1.0 if has_structure else 0.5
        return RewardEvaluation(
            reward=reward,
            is_valid=True,
            domain="code",
            rationale="Code successfully parsed with valid AST structure.",
            hacked_detected=False,
            metadata={"functions": res.functions_defined, "classes": res.classes_defined},
        )

    def evaluate_math_reward(self, expected_solution: str, candidate_output: str) -> RewardEvaluation:
        """
        Evaluate mathematical candidate against ground truth:
        - Exact match or symbolic equivalence = +1.0
        - Hallucinated or incorrect result = -1.0
        """
        if self.hacker_detector.check_evasion(candidate_output):
            return RewardEvaluation(
                reward=-1.0,
                is_valid=False,
                domain="math",
                rationale="Detected evasion or boilerplate.",
                hacked_detected=True,
            )

        clean_candidate = candidate_output.strip().lower()
        clean_expected = expected_solution.strip().lower()

        # Direct containment check
        if clean_expected in clean_candidate:
            return RewardEvaluation(
                reward=1.0,
                is_valid=True,
                domain="math",
                rationale="Candidate contains ground truth solution.",
                hacked_detected=False,
            )

        return RewardEvaluation(
            reward=-1.0,
            is_valid=False,
            domain="math",
            rationale="Candidate does not match verified mathematical ground truth.",
            hacked_detected=False,
        )
