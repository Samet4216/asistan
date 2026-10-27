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


def strip_unwanted_cjk(text: str) -> str:
    """Strips rogue Chinese/CJK characters that multilingual models might accidentally bleed."""
    if not text:
        return ""
    cleaned = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf]+', '', text)
    return re.sub(r'[ \t]{2,}', ' ', cleaned)


def normalize_latex(text: str) -> str:
    """
    Cleans LaTeX delimiters so KaTeX in Streamlit renders formulas crisply:
    - Protects code blocks (```...```) so Python code parentheses are NEVER corrupted.
    - Converts \\[ ... \\] -> $$ ... $$
    - Converts \\( ... \\) -> $ ... $
    - Converts standalone bracketed lines [ \\text{...} ] -> $$ ... $$
    - Strips rogue CJK/Chinese character bleeds.
    """
    if not text:
        return ""

    text = strip_unwanted_cjk(text)

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
        # Deep Learning & Neural Architectures
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
        "overfitting", "regularization", "dropout", "batchnorm", "layernorm", "rmsnorm", "optimizer", "adamw", "sgd",
        # Neural Network Layers, Activations & Training Dynamics
        "neuron", "nöron", "dead neuron", "dying neuron", "activation function", "aktivasyon fonksiyonu",
        "relu", "leaky relu", "prelu", "gelu", "silu", "swish", "sigmoid", "tanh",
        "initialization", "random initialization", "weight initialization", "xavier", "glorot", "kaiming", "he initialization",
        "bias weight", "bias", "weights", "neural network", "yapay sinir ağı", "ann", "cnn", "rnn", "lstm",
        "vanishing gradient", "exploding gradient", "gradient clipping", "learning rate", "backpropagation",
        # Aerospace, Aviation, Flight Mechanics & UAV
        "havacılık", "aerodinamik", "aerodynamics", "uçuş dinamiği", "flight dynamics", "hücum açısı", "angle of attack",
        "lift", "drag", "taşıma kuvveti", "sürükleme kuvveti", "itki", "thrust", "kanat", "airfoil", "kanat profili",
        "iha", "siha", "uav", "drone", "mach", "ses hızı", "aviyonik", "avionics", "pitot", "altimetre", "gyro", "jiroskop",
        "akışkanlar mekaniği", "navier-stokes", "cfd", "türbülans", "bernoulli", "roket", "itki sistemi", "itme kuvveti",
        # Physics, Math & Fundamental Sciences
        "kuantum", "quantum", "fizik", "physics", "termodinamik", "thermodynamics", "matematik", "mathematics",
        "istatistik", "olasılık", "diferansiyel", "biyoloji", "kimya", "astronomi", "elektromanyetik"
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
        "ispatı", "türetimi", "algoritması", "literatürde", "makalede", "akademik", "bilimsel",
        "teorik olarak", "deney", "ablation", "benchmark"
    ]
    has_analytical_trigger = any(trig in p for trig in analytical_triggers)

    # Quoted replies referencing previous technical discussion are always technical inquiries
    if p.startswith(">") or "\n>" in p:
        return True

    # If it contains clear technical concepts or analytical requests
    if has_tech_entity or has_analytical_trigger:
        return True

    return False


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
        "continue", "next", "go on", "go advantage", "advantage level", "advanced", "advanced level",
        "ileri seviye", "ileri seviyeye geç", "daha detaylı", "detaylı anlat", "çok detaylı", "bütün bilgileri",
        "daha fazla detay", "ayrıntılı açıkla", "derinleştir"
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
    """Parses '➡️ Sıradaki Adım: Bölüm X/N (Başlık)' or similar transitions from assistant response."""
    if not text:
        return ""
    m = re.search(r'➡️\s*(?:Sıradaki\s*Adım|Sonraki\s*Adım)?[:\s]*(Bölüm\s*\d+/\d+[^\n]+)', text, flags=re.I)
    if m:
        return m.group(1).strip()
    m2 = re.search(r'(Bölüm\s*\d+/\d+[^:\n]+)', text, flags=re.I)
    return m2.group(1).strip() if m2 else ""


def highlight_fosforlu(chunk_text: str, assistant_text: str = "", query: str = "") -> str:
    """
    Highlights cited passages and key evidence in paper chunk text with authentic fluorescent yellow marker.
    1. Extracts verbatim quotes from assistant response (text in "..." or '...') that match in chunk_text.
    2. If no exact quote matches, finds sentences in chunk_text with highest query/concept overlap.
    3. Wraps matched text in glowing <mark class="fosfor-highlighter">...</mark>.
    """
    if not chunk_text:
        return ""

    processed_text = chunk_text
    highlight_applied = False

    # 1. Look for explicit verbatim quotes from assistant_text
    if assistant_text:
        # Match quotes between double quotes or single quotes
        raw_quotes = re.findall(r'["“]([^"”]{10,})["”]', assistant_text)
        # Sort by length descending to match longest phrases first
        raw_quotes = sorted(set(raw_quotes), key=lambda x: len(x), reverse=True)

        for q in raw_quotes:
            q_clean = q.strip()
            if len(q_clean) >= 10 and q_clean in processed_text:
                pattern = re.escape(q_clean)
                replacement = f'<mark class="fosfor-highlighter">{q_clean}</mark>'
                processed_text = re.sub(pattern, replacement, processed_text, count=1)
                highlight_applied = True

    # 2. If no verbatim quote matched or as secondary highlighting, highlight key evidence sentences
    if not highlight_applied:
        # Split chunk into sentences while preserving delimiters
        sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
        search_terms = set()
        if query:
            # Extract meaningful words (>= 4 chars)
            words = re.findall(r'\b[A-Za-z0-9_ğüşıöçĞÜŞİÖÇ\-]{4,}\b', query.lower())
            search_terms.update(words)

        scored_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 25:
                continue
            s_lower = s_clean.lower()
            overlap_score = sum(1 for term in search_terms if term in s_lower)
            if overlap_score > 0:
                scored_sentences.append((overlap_score, len(s_clean), s_clean))

        if scored_sentences:
            scored_sentences.sort(key=lambda x: (x[0], x[1]), reverse=True)
            # Pick top 1-2 most relevant sentences
            for _, _, target_s in scored_sentences[:2]:
                if target_s in processed_text:
                    pattern = re.escape(target_s)
                    replacement = f'<mark class="fosfor-highlighter">{target_s}</mark>'
                    processed_text = re.sub(pattern, replacement, processed_text, count=1)
                    highlight_applied = True

    return processed_text


def inject_citation_anchors(content: str, sources: List[Dict[str, Any]] = None) -> str:
    """
    Injects interactive 🔗 citation anchor links next to headings and citation tags [S1], [S2]
    so users can click them to inspect the cited paper page and fluorescent quote in the right panel.
    """
    if not content:
        return ""

    sources = sources or []
    default_tag = sources[0].get("source_tag", "S1") if sources else "S1"
    default_id = sources[0].get("id", default_tag) if sources else "S1"

    # 1. Turn markdown headings (### Heading) into clickable headings with 🔗
    def replace_heading(match):
        level = match.group(1)
        title = match.group(2).strip()
        # Avoid double injecting if 🔗 is already present
        if "🔗" in title:
            return match.group(0)
        anchor_html = f' <a href="?inspect_chunk={default_id}" class="cite-anchor-link" data-inspect-chunk="{default_id}" target="_self" title="Bu konunun alıntılandığı makale sayfasını ve fosforlu alıntıyı sağ panelde gör">🔗</a>'
        return f"{level} {title}{anchor_html}"

    content = re.sub(r'^(#{1,4})\s+(.+)$', replace_heading, content, flags=re.MULTILINE)

    # 2. Turn inline citation tags [S1], [S2], [W1] into interactive clickable pills
    for idx, s in enumerate(sources):
        tag = s.get("source_tag", f"S{idx+1}")
        cid = s.get("id", tag)
        tag_pattern = rf'\[{tag}\]'
        pill_html = f'<a href="?inspect_chunk={cid}" class="cite-pill-link" data-inspect-chunk="{cid}" target="_self" title="Kaynak [{tag}] sayfasını sağ panelde fosforlu alıntıyla aç">[{tag} 🔗]</a>'
        content = re.sub(tag_pattern, pill_html, content)

    return content


