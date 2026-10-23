"""
Local Deep Learning Lab Assistant — Cyber-Lab AI Research Studio (Streamlit).
Provides a high-end, futuristic dark glassmorphism research laboratory interface:
- Inter & Fira Code typography with KaTeX crisp mathematical rendering
- Real-time streaming generation, multi-turn conversational context, Obsidian Vault integration
- Two-Stage Chain-of-Thought deep reasoning with live animated pulse
- Interactive text selection & quotation popover
- Sub-Query Decomposition & HyDE retrieval controls
- Live NVIDIA RTX 4050 GPU VRAM & thermal gauges
- Quick-action research prompt cards
"""

import sys
import re
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components
import psutil
import config
import llm
import budget
from retrieval.search import searcher
from retrieval.git_context import get_git_diff
from answer.prompts import (
    build_answer_prompt,
    build_thought_plan_prompt,
    build_casual_response_prompt,
    build_general_expert_prompt
)
from answer.challenge import challenger
from memory.approval import approval_manager
from memory.errors import error_manager
from memory.user_profile import profile_manager
from memory.auto_memory import auto_memory
from ingest.single_ingest import ingest_arxiv_paper
import lang_guard


# Page configuration
st.set_page_config(
    page_title="Deep Learning Lab Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cyber-Lab CSS & Typography
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Global Base Canvas */
    .stApp {
        background: radial-gradient(circle at 10% 10%, #0d1527 0%, #060911 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #e2e8f0 !important;
    }

    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 29, 0.88) !important;
        backdrop-filter: blur(28px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
    }

    /* Sleek Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(56, 189, 248, 0.28);
        border-radius: 999px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(56, 189, 248, 0.6);
    }

    /* Typography & Line Spacing */
    p, li, span {
        line-height: 1.72 !important;
    }

    /* Modern Chat Bubble Styling */
    .stChatMessage {
        border-radius: 14px !important;
        padding: 20px 24px !important;
        margin-bottom: 20px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    /* Assistant Message Glass Card */
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.75) 0%, rgba(20, 30, 55, 0.45) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-left: 4px solid #38bdf8 !important;
        box-shadow: 0 8px 32px -4px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    }

    /* User Message Gradient Card */
    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.55) 0%, rgba(15, 23, 42, 0.65) 100%) !important;
        border: 1px solid rgba(129, 140, 248, 0.3) !important;
        border-left: 4px solid #818cf8 !important;
        box-shadow: 0 6px 24px -2px rgba(0, 0, 0, 0.4) !important;
    }

    /* KaTeX Mathematical Formula Styling */
    .katex-display {
        margin: 1.3em 0 !important;
        padding: 1.1em 1.4em !important;
        background: rgba(3, 7, 18, 0.65) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 10px !important;
        box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.5) !important;
        overflow-x: auto;
    }

    /* Code Blocks */
    pre, code {
        font-family: 'Fira Code', 'JetBrains Mono', monospace !important;
    }
    pre {
        background: #040711 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(22, 31, 53, 0.5) 0%, rgba(10, 15, 29, 0.7) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }

    /* Sleek Blockquote Styling for Quoted Replies */
    blockquote {
        border-left: 3px solid #38bdf8 !important;
        background: rgba(56, 189, 248, 0.07) !important;
        padding: 10px 18px !important;
        border-radius: 6px;
        margin: 10px 0 16px 0 !important;
        color: #cbd5e1 !important;
        font-style: italic;
    }

    /* Status Widgets / Extended Thinking */
    div[data-testid="stStatusWidget"] {
        background: rgba(13, 19, 35, 0.8) !important;
        border: 1px solid rgba(168, 85, 247, 0.35) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(168, 85, 247, 0.1) !important;
    }

    /* Live Header Status Pill */
    .status-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 22px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 13px;
        color: #94a3b8;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Quick Action Cards */
    .quick-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(30, 41, 59, 0.35) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 14px 18px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .quick-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.15);
    }

    /* Hide Streamlit Debug / Ask ChatGPT buttons */
    [data-testid="stExceptionActionButtons"],
    .stException > div:last-child {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# Interactive JavaScript for ChatGPT-style text selection and quote popover
components.html("""
<script>
const parentDoc = window.parent.document;

function setupQuoteButton() {
    if (parentDoc.getElementById("agy-quote-btn")) return;

    const btn = parentDoc.createElement("div");
    btn.id = "agy-quote-btn";
    btn.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
        Alıntı Yap
    `;
    btn.style.position = "fixed";
    btn.style.display = "none";
    btn.style.zIndex = "999999";
    btn.style.background = "#0f172a";
    btn.style.color = "#38bdf8";
    btn.style.border = "1px solid #38bdf8";
    btn.style.borderRadius = "20px";
    btn.style.padding = "6px 14px";
    btn.style.fontSize = "13px";
    btn.style.fontWeight = "600";
    btn.style.fontFamily = "-apple-system, sans-serif";
    btn.style.cursor = "pointer";
    btn.style.boxShadow = "0 6px 20px rgba(0,0,0,0.6)";
    btn.style.transition = "transform 0.12s ease, background 0.12s ease";

    btn.onmouseenter = () => {
        btn.style.backgroundColor = "#1e293b";
        btn.style.transform = "scale(1.06)";
    };
    btn.onmouseleave = () => {
        btn.style.backgroundColor = "#0f172a";
        btn.style.transform = "scale(1.0)";
    };

    parentDoc.body.appendChild(btn);

    let activeSelectedText = "";

    function handleSelection() {
        const selection = parentDoc.getSelection();
        const selectedText = selection ? selection.toString().trim() : "";

        if (selectedText && selectedText.length > 2) {
            const anchorNode = selection.anchorNode;
            const chatMsg = anchorNode ? (anchorNode.nodeType === 3 ? anchorNode.parentElement : anchorNode).closest(".stChatMessage") : null;
            if (chatMsg) {
                const range = selection.getRangeAt(0);
                const rect = range.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0) {
                    activeSelectedText = selectedText;
                    btn.style.top = Math.max(10, rect.top - 42) + "px";
                    btn.style.left = Math.max(10, rect.left + (rect.width / 2) - 65) + "px";
                    btn.style.display = "block";
                    return;
                }
            }
        }
        btn.style.display = "none";
    }

    parentDoc.addEventListener("mouseup", () => setTimeout(handleSelection, 30));
    parentDoc.addEventListener("keyup", () => setTimeout(handleSelection, 30));
    parentDoc.addEventListener("selectionchange", () => {
        if (!parentDoc.getSelection().toString().trim()) {
            btn.style.display = "none";
        }
    });

    btn.addEventListener("mousedown", (e) => {
        e.preventDefault();
        e.stopPropagation();

        if (!activeSelectedText) return;

        const chatInput = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
        if (chatInput) {
            const quoteLines = activeSelectedText.split("\\n").map(l => "> " + l).join("\\n");
            const existingVal = chatInput.value ? chatInput.value.trim() + "\\n\\n" : "";
            const newVal = existingVal + quoteLines + "\\n\\n";

            const nativeSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, "value").set;
            nativeSetter.call(chatInput, newVal);
            chatInput.dispatchEvent(new Event("input", { bubbles: true }));

            chatInput.focus();
            chatInput.selectionStart = chatInput.selectionEnd = chatInput.value.length;
            chatInput.scrollIntoView({ behavior: "smooth" });
        }

        btn.style.display = "none";
    });
}

setTimeout(setupQuoteButton, 500);
</script>
""", height=0)


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
    """Detects whether prompt is a greeting, casual chat, or personal profile declaration."""
    p = text.strip().lower()
    greetings = [
        "naber", "nasılsın", "whats up", "what's up", "selam", "merhaba",
        "günaydın", "iyi akşamlar", "iyi günler", "kolay gelsin", "hey",
        "kimsin", "sen kimsin", "kimsin sen", "neler yapabilirsin", "ne yapabilirsin",
        "ne işe yararsın", "beni tanıyor musun", "adın ne", "naber?", "nasılsın?",
        "ne haber", "ne yapıyorsun", "merhabalar", "selamlar", "selamün aleyküm"
    ]
    personal_cues = [
        "benim adım", "adım ", "aklında tut", "hafızana kaydet",
        "beni hatırla", "projem", "çalışmaya başladım", "çalışıyorum"
    ]
    if any(p.startswith(g) or p == g for g in greetings):
        return True
    if any(cue in p for cue in personal_cues) and len(p.split()) < 30:
        return True
    return False


def contextualize_query(current_query: str, history: list) -> str:
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


def get_gpu_status():
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
                    "util": int(p[5])
                }
    except Exception:
        pass
    return {"available": False}


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### 🔬 Laboratuvar Donanımı")
    
    # GPU Monitor Card
    gpu = get_gpu_status()
    if gpu.get("available"):
        st.caption(f"⚡ **{gpu['name']}**")
        col1, col2 = st.columns(2)
        col1.metric("GPU Isı", f"{gpu['temp']} °C")
        col2.metric("GPU Yük", f"{gpu['util']} %")
        vram_pct = int((gpu['used'] / gpu['total']) * 100)
        st.progress(vram_pct, text=f"VRAM: {gpu['used']} / {gpu['total']} MiB")
    else:
        st.info("GPU Telemetrisi Beklemede")

    # CPU & RAM Monitor
    cpu_percent = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    st.caption("💻 **İşlemci & Sistem Belleği**")
    ccol1, ccol2 = st.columns(2)
    ccol1.metric("CPU", f"{cpu_percent} %")
    ccol2.metric("RAM", f"{ram.percent} %")
    st.progress(ram.percent / 100.0, text=f"RAM: {ram.used // (1024**2)} / {ram.total // (1024**2)} MiB")

    # Unload & Clear Buttons
    st.write("")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("❄️ VRAM'i Boşalt", help="Model ağırlıklarını VRAM'den siler, GPU'yu soğutur."):
            llm.unload()
            st.success("VRAM boşaltıldı.")
            st.rerun()
    with col_btn2:
        if st.button("🗑️ Sohbeti Sıfırla", help="Sohbet geçmişini sıfırlar ve yeni bir seans açar."):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Merhaba! Ben yerel Derin Öğrenme Araştırma Partnerinizim. İndekslenmiş 11 makalemiz ve 91 sinaptik kavram notumuz hakkında teknik sorular sorabilir, fareyle herhangi bir satırı seçip '💬 Alıntı Yap' diyerek soru sorabilirsiniz."
                }
            ]
            st.rerun()

    st.divider()

    st.markdown("### ⚙️ Akıl Yürütme Modu")
    mode = st.radio(
        "Yanıt Derinliği:",
        [
            "🎓 İki Aşamalı Derin Akıl Yürütme (Two-Stage CoT)",
            "⚡ Hızlı Akademik Soru-Cevap",
            "🎯 Devil's Advocate (/challenge)"
        ],
        index=0
    )

    st.divider()

    st.markdown("### 🔍 Arama Motoru Ayarları")
    use_decomp = st.checkbox("🔀 Alt-Sorgu Ayrıştırma (Decomposition)", value=True, help="Karmaşık veya karşılaştırmalı soruları otomatik 2-3 bağımsız sorguya bölerek arar.")
    use_hyde = st.checkbox("💡 HyDE (Hipotetik Belge Vektörü)", value=False, help="Soyut sorular için önce akademik bir varsayımsal yanıt üretip onun vektörüyle arar.")

    st.divider()

    # On-demand arXiv Paper Ingestion Section
    st.markdown("### 📥 Hafızaya Yeni Makale Ekle")
    with st.form("form_add_paper"):
        new_arxiv_id = st.text_input("arXiv Kodu:", placeholder="örn: 2103.02907 veya 1706.03762")
        submit_paper = st.form_submit_button("📥 İndir & İndeksle")
        if submit_paper and new_arxiv_id.strip():
            with st.spinner(f"[{new_arxiv_id.strip()}] arXiv'den indiriliyor, parçalanıyor ve ChromaDB'ye ekleniyor..."):
                res_ing = ingest_arxiv_paper(new_arxiv_id.strip())
                if res_ing.get("success"):
                    st.success(f"✅ '{res_ing.get('title')}' ({res_ing.get('chunks_count')} parça) başarıyla hafızaya eklendi!")
                    st.rerun()
                else:
                    st.error(f"❌ Hata: {res_ing.get('error', 'Makale indirilemedi.')}")

    st.divider()

    # Researcher Profile / About Me Panel
    with st.expander("👤 Araştırmacı Profili (Benim Hakkımda)", expanded=False):
        current_about = profile_manager.get_about_me()
        st.caption("Asistan sizi bu bilgilerle tanır. Sohbette kendinizi, projelerinizi veya tercihlerinizi anlattıkça burası otomatik güncellenir.")
        st.markdown(current_about)

    # Obsidian Pending Drafts
    drafts = approval_manager.list_drafts()
    st.markdown(f"### 📝 Obsidian Taslakları ({len(drafts)})")
    if drafts:
        for d in drafts[:5]:
            with st.expander(f"{d['title'][:25]}"):
                st.caption(f"Kategori: {d['category']} | Tarih: {d['created_at'][:16]}")
                st.write(d["preview"])
                if st.button("✓ Notu Onayla & Arşive Ekle", key=f"app_{d['id']}"):
                    approval_manager.approve_draft(d["id"])
                    st.success("Kalıcı Vault'a taşındı ve indekslendi!")
                    st.rerun()
    else:
        st.caption("Bekleyen taslak yok (Vault/Drafts/)")


# --- MAIN CHAT AREA ---

# Top Hero Status Bar
gpu_temp_str = f"{gpu['temp']}°C" if gpu.get("available") else "Normal"
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:flex-end; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:14px; margin-bottom:18px;">
    <div>
        <h1 style="margin:0; font-size:1.85rem; font-weight:700; background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Local Deep Learning Lab Assistant
        </h1>
        <p style="margin:4px 0 0 0; font-size:0.88rem; color:#94a3b8;">
            Çevrimdışı Akademik Araştırma, Tensör Laboratuvarı & İkinci Beyin
        </p>
    </div>
    <div class="status-badge-container" style="margin:0;">
        <div class="status-pill"><span class="status-dot"></span> Ollama CUDA</div>
        <div class="status-pill">⚡ RTX 4050 ({gpu_temp_str})</div>
        <div class="status-pill">📚 606 RAG Parçası</div>
        <div class="status-pill">🧠 91 Sinaps</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Merhaba! Ben yerel Derin Öğrenme Araştırma Partnerinizim. İndekslenmiş 11 makalemiz ve 91 sinaptik kavram notumuz hakkında teknik sorular sorabilir, fareyle herhangi bir satırı seçip '💬 Alıntı Yap' diyerek soru sorabilirsiniz."
        }
    ]

# One-Click Quick Action Cards (Shown at start or above input)
if len(st.session_state.messages) <= 1:
    st.markdown("##### 🚀 Hızlı Başlangıç Araştırma Başlıkları")
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        if st.button("⚖️ PPO vs DPO: Ödül Modelleme Farkları", use_container_width=True):
            st.session_state.pending_prompt = "Compare PPO clipped surrogate objective with DPO policy optimization in terms of reward modeling and memory."
        if st.button("⚡ FlashAttention: SRAM Tiling & IO-Awareness", use_container_width=True):
            st.session_state.pending_prompt = "How does FlashAttention avoid materializing the NxN attention matrix in HBM using SRAM tiling?"
    with q_col2:
        if st.button("🔍 Self-RAG: Reflection Token Mekanizması", use_container_width=True):
            st.session_state.pending_prompt = "Explain Self-RAG reflection tokens ([Retrieve], [IsRel], [IsSup]) and how they prevent hallucination."
        if st.button("📐 Coordinate Attention: 1D Uzamsal Pooling", use_container_width=True):
            st.session_state.pending_prompt = "Coordinate Attention yatay ve dikey 1D pooling ile konumsal bilgiyi nasıl korur? Formüllerini ve tensör boyutlarını açıkla."
    st.write("")

# Render chat history with normalized LaTeX and Obsidian Save buttons
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(normalize_latex(msg["content"]))
        
        # If assistant message (and not the initial welcome message), provide Save button
        if msg["role"] == "assistant" and i > 0:
            btn_col1, _ = st.columns([1, 4])
            with btn_col1:
                if st.button("💾 Obsidian'a Kaydet", key=f"save_draft_{i}", help="Bu cevabı Obsidian Vault/Drafts/ klasörüne taslak olarak kaydeder."):
                    user_q = "Theory_Note"
                    for prev_msg in reversed(st.session_state.messages[:i]):
                        if prev_msg["role"] == "user":
                            user_q = prev_msg["content"]
                            break
                    safe_title = re.sub(r'[^A-Za-z0-9]+', '_', user_q[:35]).strip('_') or "Note"
                    
                    draft_path = approval_manager.create_draft(
                        title=safe_title,
                        content=msg["content"],
                        category="Theory_Notes",
                        tags=["qa-chat", "dl-lab"]
                    )
                    st.toast(f"✅ '{safe_title}' Obsidian Vault/Drafts/ klasörüne kaydedildi!", icon="📝")
                    st.rerun()

# User input handling (or pending quick prompt)
input_prompt = st.chat_input("Bir soru sorun, selam verin veya yukarıdan bir satırı seçip '💬 Alıntı Yap' ile sorun...")

prompt = None
if input_prompt:
    prompt = input_prompt
elif "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    # Autonomous Memory Extraction: check if user shares personal facts, projects, or preferences
    if auto_memory.has_personal_cues(prompt):
        try:
            update_summary = auto_memory.extract_and_update(prompt)
            if update_summary:
                st.toast(f"🧠 Obsidian Hafızası Güncellendi: {update_summary}", icon="📝")
        except Exception:
            pass

    # Render user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        # 1. Check if Casual Chat or Personal Profile Statement
        if is_casual_or_personal(prompt):
            # Stream warm casual response
            casual_msgs = build_casual_response_prompt(prompt, history=st.session_state.messages[:-1])
            stream = llm.stream_generate(casual_msgs, max_tokens=600)
            full_response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()

        # 2. Check Devil's Advocate Mode
        elif "Devil's Advocate" in mode or prompt.startswith("/challenge"):
            with st.spinner("Hipotez literatürle ve RTX 4050 donanım sınırlarıyla test ediliyor..."):
                clean_q = prompt.replace("/challenge", "").strip()
                res = challenger.challenge_hypothesis(clean_q)
                
                verdict = res.get("verdict", "HIGH_RISK")
                v_color = "🔴" if verdict == "REJECT" else ("🟡" if verdict == "HIGH_RISK" else "🟢")
                
                output = f"### {v_color} Karar: **{verdict}**\n\n"
                output += f"**Yönetici Özeti:** {res.get('summary')}\n\n"
                output += "#### Kritik Zayıf Noktalar & Literatür Çelişkileri:\n"
                for c in res.get("criticisms", []):
                    output += f"- **{c.get('vulnerability')}:** {c.get('evidence_or_rationale')}\n"
                output += f"\n#### ⚠️ Donanım & VRAM Darboğazı (RTX 4050 6GB):\n{res.get('hardware_and_vram_risk')}\n\n"
                output += f"#### 🧪 Hipotezi Çürütecek Asgari Deney:\n{res.get('minimal_disproving_experiment')}"
                
                clean_output = normalize_latex(output)
                st.markdown(clean_output)
                st.session_state.messages.append({"role": "assistant", "content": clean_output})

        # 3. Technical Deep Learning Question
        else:
            is_deep_cot = "Two-Stage CoT" in mode or prompt.startswith("/explain")
            clean_q = prompt.replace("/explain", "").strip()

            # Multi-turn context resolution: augment search query with prior conversational topic or quote
            augmented_search_query = contextualize_query(clean_q, st.session_state.messages[:-1])

            # Step 1: Hybrid Retrieval
            with st.status("📚 Literatür ve yerel hafıza taranıyor...", expanded=False) as status:
                search_res = searcher.search(
                    augmented_search_query,
                    top_k=config.RETRIEVAL_TOP_K,
                    use_decomposition=use_decomp,
                    use_hyde=use_hyde
                )
                retrieved_context = search_res.get("packed_context", "")
                has_sufficient_evidence = search_res.get("has_sufficient_evidence", False)
                error_evidence = error_manager.get_relevant_errors(augmented_search_query)
                
                budget_parts = {
                    "error_history": error_evidence,
                    "paper_chunks": retrieved_context
                }
                packed_report = budget.pack_context(budget_parts)
                safe_packed = packed_report["packed"]
                status.update(label="Literatür tarandı.", state="complete")

            # Display decomposed sub-queries if any were generated for technical questions
            sub_qs = search_res.get("sub_queries", [])
            if len(sub_qs) > 1 and has_sufficient_evidence:
                st.info("🔀 **Ayrıştırılan Alt-Sorgular:**\n" + "\n".join([f"- `{sq}`" for sq in sub_qs]))

            # Honest Abstention: If no evidence in local papers/notes, abstain politely and state 'Bilmiyorum'
            if not has_sufficient_evidence:
                abstain_msg = (
                    "### ⚠️ Yerel Hafızada / Makalelerde Bulunamadı\n\n"
                    f"Sorduğunuz soru (*\"{clean_q}\"*) ile ilgili yerel kütüphanemdeki akademik makalelerde ve notlarda herhangi bir bilgi veya kanıt bulunamadı.\n\n"
                    "Yanıltıcı ve uydurma (halüsinasyon) bilgi vermemek prensibim olduğu için **bu sorunun cevabını bilmiyorum**.\n\n"
                    "💡 *Eğer bu bir Derin Öğrenme / AI konusu ise, sol menüdeki **'📥 Hafızaya Yeni Makale Ekle'** kutusundan ilgili makalenin arXiv ID'sini girerek hafızama ekleyebilirsiniz.*"
                )
                st.markdown(abstain_msg)
                st.session_state.messages.append({"role": "assistant", "content": abstain_msg})
                st.rerun()

            # Display source citations box if available
            sources = search_res.get("results", [])
            if sources:
                with st.expander(f"📚 Referans Alınan Makale Parçaları ({len(sources)} Bölüm)"):
                    for s in sources:
                        meta = s.get("metadata", {})
                        title = meta.get("title", "Akademik Makale")
                        section = meta.get("section_name", "")
                        score = s.get("rerank_score", 0.0)
                        sec_info = f" • Bölüm: *{section}*" if section else ""
                        st.markdown(f"- 📄 **{title}**{sec_info} (`{s.get('id')}`) — *Alaka Skoru: `{score:.2f}`*")

            thought_plan = ""
            # Step 2: Extended Thinking Mode - Model Thinks Deeply in Public
            if is_deep_cot:
                import time
                t_start = time.time()
                with st.status("💭 Derin Düşünme & Analiz (Model kendi kendine muhakeme yapıyor...)", expanded=True) as status_cot:
                    cot_prompt_msgs = build_thought_plan_prompt(clean_q, safe_packed.get("paper_chunks", ""))
                    thought_stream = llm.stream_generate(cot_prompt_msgs, max_tokens=650, temperature=0.2)
                    thought_plan = st.write_stream(thought_stream)
                    elapsed_sec = max(1, int(time.time() - t_start))
                    status_cot.update(label=f"💭 Düşünce Süreci ({elapsed_sec} saniye boyunca derin analiz yapıldı)", state="complete", expanded=False)

            # Step 3: Stage 2 - Comprehensive Masterclass Synthesis
            messages = build_answer_prompt(
                query=clean_q,
                packaged_context=safe_packed.get("paper_chunks", ""),
                telemetry_context="",
                git_diff="",
                error_context=safe_packed.get("error_history", ""),
                thought_plan=thought_plan,
                deep_dive=is_deep_cot,
                history=st.session_state.messages[:-1]
            )

            target_max_tokens = config.TOKEN_BUDGET.get("max_answer", 2500) if is_deep_cot else 1000

            stream = llm.stream_generate(
                messages,
                max_tokens=target_max_tokens
            )
            full_response = st.write_stream(stream)
            clean_full = normalize_latex(full_response)
            st.session_state.messages.append({"role": "assistant", "content": clean_full})
            st.rerun()
