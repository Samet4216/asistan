"""
Citation & Quote Verification Module (Faz 7).
Verifies that model-extracted quotes are actually present in the retrieved context chunks.
Flags or discards ungrounded/hallucinated quotes deterministically.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def normalize_text(text: str) -> str:
    """Normalizes whitespace and removes punctuation for robust substring verification."""
    t = text.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def verify_quotes(model_quotes: List[Dict[str, str]], retrieved_context: str) -> Tuple[List[Dict[str, Any]], float]:
    """
    Verifies each quote in model_quotes against the retrieved context text.
    Returns:
        (verified_quote_list, verified_ratio)
    """
    if not model_quotes:
        return [], 0.0

    normalized_context = normalize_text(retrieved_context)
    verified_list = []
    verified_count = 0

    for q in model_quotes:
        source = q.get("source", "Unknown")
        quote_text = q.get("text", "").strip()
        norm_q = normalize_text(quote_text)

        is_verified = False
        if norm_q and (norm_q in normalized_context):
            is_verified = True
            verified_count += 1
        elif len(norm_q.split()) >= 4:
            # Check 80% token overlap for slightly paraphrase-tolerant matching
            q_words = norm_q.split()
            n_match = sum(1 for w in q_words if w in normalized_context)
            if (n_match / len(q_words)) >= 0.85:
                is_verified = True
                verified_count += 1

        verified_list.append({
            "source": source,
            "text": quote_text,
            "verified": is_verified
        })

    verified_ratio = (verified_count / len(model_quotes)) if model_quotes else 0.0
    return verified_list, round(verified_ratio, 2)


if __name__ == "__main__":
    context = "Coordinate Attention factorizes 2D spatial pooling into horizontal and vertical 1D pooling kernels."
    quotes = [
        {"source": "Sec 2", "text": "Coordinate Attention factorizes 2D spatial pooling into horizontal and vertical 1D pooling"},
        {"source": "Sec 3", "text": "This model consumes 500GB of VRAM and runs on Mars"}
    ]
    res, ratio = verify_quotes(quotes, context)
    print(f"Verified Ratio: {ratio * 100:.0f}%")
    for r in res:
        print(f" • [{r['source']}] (Verified: {r['verified']}) -> \"{r['text']}\"")
