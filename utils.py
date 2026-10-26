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


def is_academic_or_technical_inquiry(text: str) -> bool:
    """
    Intelligently identifies if the user's prompt is an academic, theoretical,
    mathematical, architecture-oriented, or technical research question that
    warrants deep literature RAG (ChromaDB + arXiv + textbook masterclass synthesis).
    """
    p = text.strip().lower()

    # Technical AI / Robotics / Math entities & keywords
    technical_entities = [
        "diffusion", "ddpm", "ddim", "score-based", "flow matching", "langevin",
        "robotik", "robotics", "action trajectory", "visuomotor", "manipulation", "end-effector",
        "reinforcement learning", "pekiştirmeli öğrenme", "pekiştirmeli", "ppo", "dpo", "kto", "orpo", "grpo",
        "q-learning", "actor-critic", "sac", "td3", "policy gradient", "bellman", "markov", "mdp",
        "transformer", "attention", "self-attention", "cross-attention", "flashattention", "coordinate attention",
        "mha", "mqa", "gqa", "roformer", "rope", "alibi",
        "yolo", "yolov8", "yolov11", "backbone", "neck", "head", "c2f", "c3k2", "fpn", "pan", "bifpn", "anchor", "nms", "ciou",
        "vit", "vision transformer", "swin", "convnext", "resnet", "vgg", "mobilenet", "efficientnet", "deit",
        "vae", "gan", "autoencoder", "latent space", "clip", "sam", "segmentation", "detection", "segmentasyon",
        "loss fonksiyonu", "kayıp fonksiyonu", "surrogate objective", "cross-entropy", "contrastive", "triplet",
        "tensör", "tensor", "gradient", "gradyan", "backprop", "türev", "integral", "ispat", "matris", "matrix",
        "sram", "hbm", "cuda", "tiling", "flops", "bandwidth", "vram", "quantization", "qlora", "lora", "peft", "4-bit", "fp16", "bf16",
        "makale", "paper", "arxiv", "literatür", "derin öğrenme", "deep learning", "yapay zeka", "ai mimarisi", "mimarisi", "mimari",
        "multimodal", "token", "embedding", "rag", "retrieval", "vektör veritabanı", "chromadb", "llm",
        "overfitting", "regularization", "dropout", "batchnorm", "layernorm", "rmsnorm", "optimizer", "adamw", "sgd"
    ]

    # Match entities with word boundary or clean substring for multi-word phrases
    has_tech_entity = False
    for ent in technical_entities:
        if " " in ent or "-" in ent:
            if ent in p:
                has_tech_entity = True
                break
        else:
            if re.search(r'\b' + re.escape(ent) + r'\b', p):
                has_tech_entity = True
                break

    # Architectural, mechanistic, and comparative analytical query patterns
    analytical_triggers = [
        "nasıl üretir", "nasıl çalışır", "neden üstündür", "çalışma prensibi", "çalışma mantığı",
        "karşılaştır", "farkı nedir", "farkları nelerdir", "avantajı", "dezavantajı",
        "matematiksel formülü", "formülünü açıkla", "denklemi", "tensör boyutları", "boyut analizi",
        "ispatı", "türetimi", "algoritması"
    ]
    has_analytical_trigger = any(trig in p for trig in analytical_triggers)

    # If it contains clear technical concepts or analytical requests
    if has_tech_entity or has_analytical_trigger:
        return True

    return False


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
    continuation_triggers = [
        "devam", "devam et", "devam edelim", "sonraki", "sıradaki", "sonraki parça",
        "sıradaki parça", "evet devam", "evet", "geçelim", "sonrakine geç", "sıradaki bölüme geç",
        "sıradaki adım", "sonraki bölüm", "bölüm", "2. parça", "3. parça", "ikinci parça", "üçüncü parça",
        "continue", "next", "go on"
    ]

    is_starter = any(lower_q.startswith(s) for s in followup_starters)
    is_referential = any(re.search(r'\b' + re.escape(w) + r'\b', lower_q) for w in referential_keywords)
    is_continuation = lower_q in continuation_triggers or any(lower_q.startswith(ct) for ct in continuation_triggers)

    if is_starter or is_referential or is_continuation:
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
            if is_continuation:
                return f"{clean_topic} : sıradaki parça ve derinlemesine devamı ({current_query})"
            return f"{clean_topic} : {current_query}"

    return current_query


def extract_next_section(text: str) -> str:
    """
    Asistan yanıtındaki bölüm kontrol noktasından sıradaki bölüm başlığını otonom olarak ayıklar.
    Örn: '➡️ **Sıradaki Adım: Bölüm 2/5 (PGI Mimarisi ve Tensör Akışı)**' -> 'Bölüm 2/5: PGI Mimarisi ve Tensör Akışı'
    """
    if not text:
        return ""
    
    patterns = [
        r'(?:Sıradaki|Sonraki)\s*(?:Adım|Bölüm|Parça)[^:\n]*[:\-]\s*([^\n\r*]+)',
        r'(?:devam edelim mi|geçelim mi)[^(\n]*\((?:\*\*)?([^)\*]+)(?:\*\*)?\)',
        r'Bölüm\s*(\d+/\d+[^:\n]*:[^\n\r*]+)',
        r'🎯\s*(?:Bölüm|Parça)\s*\d+/\d+\s*tamamlandı[^\n]*\n+[^:\n]*:\s*([^\n\r*]+)'
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            clean = m.group(1).strip().strip('*').strip()
            # trailing punctuations
            clean = re.sub(r'[\?\.!:]+$', '', clean).strip()
            if clean.count('(') > clean.count(')'):
                clean += ')'
            # If length is reasonable
            if 3 < len(clean) < 80:
                return clean
    return ""
