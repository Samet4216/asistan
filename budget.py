"""
Budget Module for Context Window Management.
Calculates token allocations, prevents silent Ollama context truncation,
and packs multi-source evidence safely within defined token limits.
"""

from typing import Dict, Tuple, Any
import config


def estimate_tokens(text: str) -> int:
    """
    Fast, reliable heuristic token counter for English technical and code text.
    In technical text, 1 token is roughly 3.6 - 3.8 characters or 0.75 words.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.7))


def truncate_to_tokens(text: str, max_tokens: int) -> Tuple[str, bool]:
    """
    Truncates text to stay within max_tokens budget, splitting gracefully at newlines or spaces.
    Returns (truncated_text, was_truncated).
    """
    current_tokens = estimate_tokens(text)
    if current_tokens <= max_tokens:
        return text, False

    # Target character limit
    char_limit = int(max_tokens * 3.7)
    truncated = text[:char_limit]

    # Try to break cleanly at last newline or period
    break_idx = truncated.rfind("\n")
    if break_idx == -1:
        break_idx = truncated.rfind(". ")
    if break_idx != -1 and break_idx > int(char_limit * 0.7):
        truncated = truncated[:break_idx]

    notice = f"\n\n[WARNING: Content truncated to fit token budget ({max_tokens} tokens)]"
    return truncated + notice, True


def pack_context(parts: Dict[str, str], limits: Dict[str, int] = None) -> Dict[str, Any]:
    """
    Packs multi-source evidence (telemetry, git diff, cards, chunks) according to budgets.
    Returns:
        {
            "packed": {part_name: text},
            "stats": {part_name: {"tokens": int, "truncated": bool}},
            "total_tokens": int
        }
    """
    budget_map = limits or config.TOKEN_BUDGET
    packed = {}
    stats = {}
    total_tokens = 0

    for name, content in parts.items():
        if not content:
            packed[name] = ""
            stats[name] = {"tokens": 0, "truncated": False}
            continue

        max_allowed = budget_map.get(name, 500)
        safe_text, was_truncated = truncate_to_tokens(content, max_allowed)
        tokens_used = estimate_tokens(safe_text)

        packed[name] = safe_text
        stats[name] = {
            "tokens": tokens_used,
            "truncated": was_truncated,
            "budget": max_allowed
        }
        total_tokens += tokens_used

    return {
        "packed": packed,
        "stats": stats,
        "total_tokens": total_tokens
    }


if __name__ == "__main__":
    print("[*] Testing budget.py with deliberate overflow...")
    huge_diff = "diff --git a/model.py b/model.py\n" + ("+ torch.nn.Linear(512, 512)\n" * 300)
    
    test_parts = {
        "telemetry": "val_loss rose from 0.45 to 0.62 in epochs 12-16",
        "git_diff": huge_diff,
        "paper_cards": "Attention Is All You Need (Vaswani et al., 2017)"
    }
    
    result = pack_context(test_parts)
    print("\n[Token Usage Summary]:")
    for name, s in result["stats"].items():
        print(f" • {name}: {s['tokens']} / {s['budget']} tokens (Truncated: {s['truncated']})")
    print(f"\nTotal Context Tokens: {result['total_tokens']}")
