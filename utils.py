"""
Shared Utility Module for Local Deep Learning Lab Assistant.
Centralizes GPU telemetry, LaTeX formatting, query contextualization,
and prompt intent classification.
"""

import re
import subprocess
from typing import Dict, Any, List


def get_gpu_status() -> Dict[str, Any]:
    """Queries NVIDIA SMI for real-time RTX 4050 metrics."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            p = [x.strip() for x in res.stdout.strip().split(",")]
            if len(p) >= 6:
                return {
                    "available": True,
                    "name": p[0],
                    "total": int(p[1]),
                    "used": int(p[2]),
                    "free": int(p[3]),
                    "temp": int(p[4]),
                    "util": int(p[5]),
                    # Compatibility aliases for app.py
                    "total_mb": int(p[1]),
                    "used_mb": int(p[2]),
                    "free_mb": int(p[3]),
                    "temp_c": int(p[4]),
                    "util_pct": int(p[5])
                }
    except Exception:
        pass
    return {"available": False}


def normalize_latex(text: str) -> str:
    """
    Cleans LaTeX delimiters so KaTeX in Streamlit renders formulas crisply:
    - Protects code blocks (```...```) so Python code parentheses are NEVER corrupted.
    - Converts \\[ ... \\] -> $$ ... $$
    - Converts \\( ... \\) -> $ ... $
    - Converts standalone bracketed lines [ \\text{...} ] -> $$ ... $$
    """
    if not text:
        return ""

    parts = re.split(r'(```[\s\S]*?```)', text)
    processed = []

    for part in parts:
        if part.startswith("```"):
            processed.append(part)
            continue

        p = re.sub(r'\\\[([\s\S]*?)\\\]', r'$$\n\1\n$$', part)
        p = re.sub(r'\\\(([\s\S]*?)\\\)', r'$\1$', p)

        def _bracket_repl(m):
            inner = m.group(1).strip()
            indicators = ["=", "\\text", "\\frac", "\\times", "^", "_", "\\cdot", "\\left", "\\sum", "softmax", "ReLU"]
            if any(tok in inner for tok in indicators):
                return f"\n$$\n{inner}\n$$\n"
            return m.group(0)

        p = re.sub(r'(?m)^\s*\[\s*([^\n\]]+)\s*\]\s*$', _bracket_repl, p)
        p = re.sub(r'\${3,}', '$$', p)
        processed.append(p)

    return "".join(processed)


def is_casual_or_personal(text: str) -> bool:
    """Detects whether prompt is a greeting, casual chat, idea sharing, or personal profile declaration."""
    p = text.strip().lower()
    conversational_starters = [
        "whats up", "what's up", "sup", "hello", "hi", "hey", "how are you",
        "how's it going", "good morning", "good evening", "who are you",
        "kanka", "kankam", "hocam", "dostum", "reis", "selam", "merhaba",
        "naber", "nasılsın", "günaydın", "iyi akşamlar", "iyi günler",
        "kolay gelsin", "merhabalar", "selamlar", "selamün aleyküm", "bak şimdi",
        "sen kimsin", "kimsin sen", "beni tanıyor musun", "adın ne", "ne yapıyorsun",
        "ne haber", "ne var ne yok"
    ]
    personal_cues = [
        "benim adım", "adım ", "aklında tut", "hafızana kaydet",
        "beni hatırla", "projem", "çalışmaya başladım", "çalışıyorum",
        "düşünüyorum", "planlıyorum", "yapmak istiyorum", "öğrenciyim", "okuyorum",
        "fikrim var", "bir model eğiteceğim", "model eğitmeyi", "başlayacağım",
        "yapacağım", "benim hakkımda", "hakkımda ne biliyorsun", "beni tanıyor musun",
        "geçmişim", "okulum", "bölümüm", "tezim", "ben ", "benim "
    ]
    if any(p == g or p.startswith(g) or g in p.split()[:3] for g in conversational_starters):
        return True
    if any(cue in p for cue in personal_cues) and len(p.split()) < 40:
        return True
    return False


def is_challenge_request(text: str) -> bool:
    """Detects whether prompt is requesting a Devil's Advocate or hypothesis stress-test."""
    p = text.strip().lower()
    if p.startswith("/challenge") or p.startswith("/test-hypo"):
        return True
    triggers = [
        "hipotezimi test et", "hipotezimi sına", "bu fikrimi çürüt",
        "fikrimi eleştir", "fikrimi sına", "şeytanın avukatı",
        "devil's advocate", "failure mode analizi", "hipotezi test et",
        "hipotezimi değerlendir", "fikrimi değerlendir"
    ]
    return any(trig in p for trig in triggers)


def contextualize_query(current_query: str, history: List[Dict[str, str]]) -> str:
    """
    Augments genuine follow-up questions or quoted replies with conversation context.
    NEVER overrides standalone or topic-changing questions.
    """
    if not history or len(history) <= 1:
        return current_query

    # If query contains an explicit blockquote reply (> ...):
    if ">" in current_query:
        lines = current_query.split("\n")
        quote_lines = [l.replace(">", "").strip() for l in lines if l.strip().startswith(">")]
        user_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith(">")]
        quote_text = " ".join(quote_lines)[:80]
        user_q = " ".join(user_lines)
        return f"{quote_text} : {user_q}"

    lower_q = current_query.strip().lower()

    # Explicit triggers that signify a dependent follow-up referencing previous turn
    followup_starters = [
        "peki ", "peki ya ", "peki bu ", "peki bunun ", "peki onun ",
        "what about ", "and what about ", "how about ", "and why is "
    ]
    referential_keywords = [
        "bunun", "bunu", "buna", "bunda", "ondan", "onun", "ona", "onu",
        "aralarındaki fark", "yukarıdaki", "bahsettiğin", "bu formül", "bu kod",
        "the former", "the latter", "difference between them"
    ]

    is_starter = any(lower_q.startswith(s) for s in followup_starters)
    is_referential = any(re.search(r'\b' + re.escape(w) + r'\b', lower_q) for w in referential_keywords)

    if is_starter or is_referential:
        last_user_query = ""
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user_query = msg.get("content", "")
                break

        if last_user_query and last_user_query != current_query:
            clean_topic = last_user_query
            for pfx in ["explain", "how does", "why do", "what is", "what are", "anlat", "açıkla"]:
                clean_topic = re.sub(r"^" + re.escape(pfx) + r"\s+", "", clean_topic, flags=re.I)
            clean_topic = clean_topic[:50].strip()
            return f"{clean_topic} : {current_query}"

    return current_query
