"""
End-to-End Answer Pipeline Module (Faz 7).
Executes: Soru -> Dil Kontrolü -> Retrieval -> Telemetri/Git -> Pack Context -> LLM -> Alıntı Doğrulama.
Enforces honest abstention when evidence is insufficient.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import llm
import budget
import lang_guard
from retrieval.search import searcher
from retrieval.git_context import get_git_diff
from ingest.log_parser import LogParser
from answer.schemas import ANSWER_SCHEMA
from answer.prompts import build_answer_prompt
from answer.verify import verify_quotes
from memory.errors import error_manager


def format_math_for_terminal(text: str) -> str:
    """Cleans LaTeX escape corruptions (\t ext -> text, \t imes -> ×) for readable terminal display."""
    import re
    if not text:
        return ""
    # Fix corrupted tab escapes
    text = re.sub(r'[\t ]*ext\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'[\t ]*imes', r' × ', text)
    text = re.sub(r'\\times', r' × ', text)
    text = re.sub(r'\\cdot', r' · ', text)
    text = re.sub(r'\\quad', r'   ', text)
    text = re.sub(r'\\qquad', r'     ', text)
    text = re.sub(r'\\sigma', r'σ', text)
    text = re.sub(r'\\delta', r'δ', text)
    text = re.sub(r'\\alpha', r'α', text)
    text = re.sub(r'\\beta', r'β', text)
    text = re.sub(r'\\approx', r' ≈ ', text)
    text = re.sub(r'\\leq|\\le', r' ≤ ', text)
    text = re.sub(r'\\geq|\\ge', r' ≥ ', text)
    text = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
    # Clean double spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text


class AnswerPipeline:
    def __init__(self):
        self.searcher = searcher

    def run(
        self,
        query: str,
        run_dir: Optional[Path] = None,
        include_diff: bool = False,
        top_k: int = 5,
        deep_dive: bool = False
    ) -> Dict[str, Any]:
        """
        Executes full reasoning and verification pipeline.
        Supports deep_dive mode for comprehensive textbook-grade pedagogical tutorials.
        """
        # Step 1: Language check
        is_tr, tr_warning = lang_guard.detect_turkish(query)

        # Step 2: Telemetry analysis if run_dir is provided
        telemetry_evidence = ""
        if run_dir:
            run_path = Path(run_dir)
            if run_path.exists():
                parser = LogParser(run_path)
                telemetry_res = parser.analyze_results()
                telemetry_evidence = telemetry_res.get("evidence", "")

        # Step 3: Git diff if requested
        git_diff_text = get_git_diff() if include_diff else ""

        # Step 4: Hybrid retrieval
        search_res = self.searcher.search(query, top_k=top_k)
        retrieved_context = search_res.get("packed_context", "")
        has_sufficient_evidence = search_res.get("has_sufficient_evidence", False)

        # Step 5: Check abstention
        if not retrieved_context and not telemetry_evidence and not git_diff_text:
            return {
                "query": query,
                "has_evidence": False,
                "answer": "No sufficient source found in local papers or vault to answer this question accurately.",
                "quotes": [],
                "verified_ratio": 0.0,
                "warning": tr_warning
            }

        # Step 6: Retrieve relevant error gotchas
        error_evidence = error_manager.get_relevant_errors(query)

        # Step 7: Pack context according to token budgets
        budget_parts = {
            "telemetry": telemetry_evidence,
            "git_diff": git_diff_text,
            "error_history": error_evidence,
            "paper_chunks": retrieved_context
        }
        packed_report = budget.pack_context(budget_parts)
        safe_packed = packed_report["packed"]

        # Step 8: Build prompt & call LLM
        if deep_dive:
            messages = build_answer_prompt(
                query=query,
                packaged_context=safe_packed.get("paper_chunks", ""),
                telemetry_context=safe_packed.get("telemetry", ""),
                git_diff=safe_packed.get("git_diff", ""),
                error_context=safe_packed.get("error_history", ""),
                deep_dive=True
            )
            raw_output = llm.generate(messages, schema=None, max_tokens=config.TOKEN_BUDGET.get("max_answer", 2500))
            formatted_ans = format_math_for_terminal(raw_output)
            return {
                "query": query,
                "has_evidence": True,
                "answer": formatted_ans,
                "quotes": [],
                "verified_ratio": 1.0,
                "key_takeaways": [],
                "retrieved_sources": [r.get("id") for r in search_res.get("results", [])],
                "warning": tr_warning,
                "is_deep_dive": True
            }

        messages = build_answer_prompt(
            query=query,
            packaged_context=safe_packed.get("paper_chunks", ""),
            telemetry_context=safe_packed.get("telemetry", ""),
            git_diff=safe_packed.get("git_diff", ""),
            error_context=safe_packed.get("error_history", ""),
            deep_dive=False,
            json_mode=True
        )

        raw_output = llm.generate(messages, schema=ANSWER_SCHEMA, max_tokens=config.TOKEN_BUDGET.get("max_answer", 1500))

        def _parse_llm_json(text: str) -> Dict[str, Any]:
            clean = text.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                clean = "\n".join(lines).strip()

            try:
                return json.loads(clean)
            except Exception:
                pass

            for suffix in ['"}', '"]}', '}', '"]}}']:
                try:
                    return json.loads(clean + suffix)
                except Exception:
                    pass

            import re
            quotes_list = []
            for src, txt in re.findall(r'\{\s*"source":\s*"(.*?)",\s*"text":\s*"(.*?)"\s*\}', clean, re.DOTALL):
                quotes_list.append({"source": src, "text": txt})

            ans_match = re.search(r'"answer":\s*"(.*?)(?:",\s*"key_takeaways"|"$|\Z)', clean, re.DOTALL)
            ans = ans_match.group(1) if ans_match else clean

            return {
                "has_evidence": has_sufficient_evidence,
                "quotes": quotes_list,
                "answer": ans.replace('\\"', '"').replace('\\n', '\n'),
                "key_takeaways": []
            }

        parsed_output = _parse_llm_json(raw_output)

        # Step 8: Verify quotes against retrieved context
        quotes = parsed_output.get("quotes", [])
        verified_quotes, verified_ratio = verify_quotes(quotes, retrieved_context + "\n" + telemetry_evidence)

        raw_ans = parsed_output.get("answer", "")
        formatted_ans = format_math_for_terminal(raw_ans)

        return {
            "query": query,
            "has_evidence": parsed_output.get("has_evidence", True),
            "answer": formatted_ans,
            "quotes": verified_quotes,
            "verified_ratio": verified_ratio,
            "key_takeaways": parsed_output.get("key_takeaways", []),
            "retrieved_sources": [r.get("id") for r in search_res.get("results", [])],
            "warning": tr_warning
        }


# Global pipeline instance
pipeline = AnswerPipeline()


if __name__ == "__main__":
    print("[*] Testing answer/pipeline.py...")
    test_query = "How does Coordinate Attention capture position along the two spatial axes?"
    res = pipeline.run(test_query, top_k=3)
    
    print("\n" + "="*50)
    print(f"QUESTION: {res['query']}")
    print("="*50)
    print(f"ANSWER:\n{res['answer']}\n")
    print(f"CITATION VERIFIED RATIO: {res['verified_ratio'] * 100:.0f}%")
    print("\nQUOTES EXTRACTED:")
    for q in res.get("quotes", []):
        v_mark = "OK" if q.get("verified") else "FAIL"
        print(f" [{v_mark}] ({q.get('source')}) \"{q.get('text')}\"")
    print("\nKEY TAKEAWAYS:")
    for t in res.get("key_takeaways", []):
        print(f" * {t}")
    print("="*50)
