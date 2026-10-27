"""
Query Transformation Module: Sub-Query Decomposition and HyDE (Hypothetical Document Embeddings).
Enables multi-facet research question retrieval and bridging vocabulary gaps with academic hypothetical passages.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import llm

DECOMPOSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of 2 to 3 atomic, focused academic search queries."
        }
    },
    "required": ["sub_queries"]
}


ACRONYM_MAP = {
    "mux": "multiplexer MUX",
    "demux": "demultiplexer DEMUX",
    "alu": "arithmetic logic unit ALU",
    "adc": "analog to digital converter ADC",
    "dac": "digital to analog converter DAC",
    "pwm": "pulse width modulation PWM",
    "gpio": "general purpose input output GPIO",
    "k-map": "karnaugh map k-map",
    "kmap": "karnaugh map k-map",
    "ff": "flip flop flip-flop",
    "fsm": "finite state machine FSM",
    "pid": "proportional integral derivative PID controller",
    "bldc": "brushless dc motor BLDC",
    "imu": "inertial measurement unit IMU",
    "esc": "electronic speed controller ESC",
    "rl": "reinforcement learning RL",
    "ppo": "proximal policy optimization PPO",
    "dqn": "deep q network DQN",
    "ddpg": "deep deterministic policy gradient DDPG",
    "td3": "twin delayed deep deterministic policy gradient TD3",
    "sac": "soft actor critic SAC",
    "grpo": "group relative policy optimization GRPO",
    "sahi": "slicing aided hyper inference SAHI",
    "slam": "simultaneous localization and mapping SLAM",
    "yolo": "you only look once YOLO object detection",
}

TURKISH_QUESTION_TEMPLATES = {
    "nedir": "What is",
    "nasil calisir": "How does",
    "nasıl çalışır": "How does",
    "neden": "Why",
    "farki": "difference between",
    "farkı": "difference between",
    "avantajlari": "advantages of",
    "avantajları": "advantages of",
    "dezavantajlari": "disadvantages of",
    "dezavantajları": "disadvantages of",
}


class QueryTransformer:
    def __init__(self):
        self.decomposition_schema = DECOMPOSITION_SCHEMA

    def expand_acronyms(self, query: str) -> str:
        """Expands technical acronyms in query to improve lexical and semantic match."""
        tokens = re.findall(r'[\w\-]+', query)
        expanded = []
        for t in tokens:
            t_low = t.lower()
            if t_low in ACRONYM_MAP:
                expanded.append(f"{t} ({ACRONYM_MAP[t_low]})")
            else:
                expanded.append(t)
        return " ".join(expanded)

    def should_decompose(self, query: str) -> bool:
        """
        Detects if a query contains comparison, multiple concepts, or conjunctions
        that would benefit from multi-path sub-query retrieval.
        """
        q_lower = query.lower()
        signals = [
            " vs ", " vs. ", " versus ", " difference between ", " compare ",
            " and how ", " and why ", " as well as ",
            " farkı", " fark ", " ile ", " karşılaştır", " ve "
        ]
        if any(sig in q_lower for sig in signals):
            return True
        if q_lower.count("?") > 1:
            return True
        if len(query.split()) >= 12:
            return True
        return False

    def decompose(self, query: str, max_sub_queries: int = 3) -> List[str]:
        """
        Decomposes a complex deep learning research query into 2-3 focused sub-queries.
        Falls back to original query on simple questions or failures.
        """
        if not self.should_decompose(query):
            return [query]

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are an academic retrieval optimizer for a Deep Learning research database. "
                    "Your job is to break a complex user question into 2 or 3 atomic, highly targeted search queries. "
                    "Keep each sub-query concise and focused on a single paper, concept, or mechanism. "
                    "Return ONLY valid JSON matching the schema."
                )
            },
            {
                "role": "user",
                "content": f"Decompose this deep learning research question into 2-3 atomic search queries:\n\"{query}\""
            }
        ]

        try:
            raw = llm.generate(prompt_messages, schema=self.decomposition_schema, max_tokens=150, temperature=0.1)
            parsed = json.loads(raw)
            subs = parsed.get("sub_queries", [])
            valid_subs = [s.strip() for s in subs if s.strip() and len(s.strip()) > 5]
            if valid_subs:
                # Always ensure original query or keywords are retained
                return valid_subs[:max_sub_queries]
        except Exception as e:
            print(f"[!] Sub-query decomposition fallback: {e}")

        return [query]

    def generate_hyde(self, query: str) -> str:
        """
        Generates a concise hypothetical academic passage (HyDE)
        that serves as a semantic bridge between user intent and formal arXiv paper text.
        """
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a scientific AI writing an excerpt from an authoritative Deep Learning research paper. "
                    "Write a direct, 2-to-3 sentence academic paragraph answering the user's question with precise technical terms, "
                    "mathematical variables, and architectural concepts. Do not include introductory conversational filler."
                )
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nWrite hypothetical research paper excerpt:"
            }
        ]

        try:
            hyde_passage = llm.generate(prompt_messages, max_tokens=220, temperature=0.2)
            if hyde_passage and len(hyde_passage.strip()) > 30:
                return hyde_passage.strip()
        except Exception as e:
            print(f"[!] HyDE generation fallback: {e}")

        return query

    def to_english_search_query(self, query: str) -> str:
        """
        The embedder (bge-base-en) and reranker (ms-marco) are English-only and the papers
        are English, so Turkish questions retrieve poorly. Translate non-English questions
        into a concise English search query. Returns the original query or acronym-expanded query
        if it is already English or translation fails.
        """
        clean_q = query.strip()
        try:
            import lang_guard
            is_tr, _ = lang_guard.detect_turkish(clean_q)
        except Exception:
            is_tr = bool(re.search(r"[çğıöşüÇĞİÖŞÜ]", clean_q)) or any(w in clean_q.lower() for w in ["nedir", "nasıl", "nasil", "ne", "fark"])
        if not is_tr:
            return self.expand_acronyms(clean_q)

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "Translate the user's Turkish question into ONE complete, natural English question "
                    "for searching an engineering, mechatronics, and AI paper database (starts with How, Why, What, etc.). "
                    "Keep technical terms, model names, symbols, and acronyms intact (e.g. MUX, DEMUX, PPO, DPO, FlashAttention, HBM, SRAM, ViT). "
                    "Output ONLY the English question, no quotes, no explanation."
                )
            },
            {"role": "user", "content": clean_q[:600]}
        ]
        try:
            out = llm.generate(prompt_messages, max_tokens=80, temperature=0.0)
            out = (out or "").strip().strip('"').splitlines()[0].strip() if out else ""
            if out and not out.startswith("[LLM ERROR]") and len(out) > 5:
                return self.expand_acronyms(out)
        except Exception as e:
            print(f"[!] Query translation fallback: {e}")

        # Smart Rule-Based Offline Translation Fallback
        q_low = clean_q.lower()
        for tr_pattern, en_prefix in TURKISH_QUESTION_TEMPLATES.items():
            if tr_pattern in q_low:
                concept = re.sub(rf"\b{re.escape(tr_pattern)}\b", "", clean_q, flags=re.IGNORECASE).strip(" ?.,")
                expanded_concept = self.expand_acronyms(concept)
                return f"{en_prefix} {expanded_concept}?"

        return self.expand_acronyms(clean_q)


# Global instance
query_transformer = QueryTransformer()
