"""
Local Deep Learning Lab Assistant — Modern Web Chat Interface (Streamlit).
Provides a ChatGPT/Open-WebUI-grade browser interface with KaTeX math rendering,
real-time streaming generation, multi-turn conversational context, Obsidian Vault integration,
Two-Stage Chain-of-Thought deep reasoning, and live RTX 4050 & CPU hardware telemetry.
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
import psutil
import config
import llm
import budget
from retrieval.search import searcher
from retrieval.git_context import get_git_diff
from answer.prompts import build_answer_prompt, build_thought_plan_prompt
from answer.challenge import challenger
from memory.approval import approval_manager
from memory.errors import error_manager
import lang_guard


# Page configuration
st.set_page_config(
    page_title="Deep Learning Lab Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for ChatGPT/Ollama-like Typography
st.markdown("""
<style>
    /* Dark comfortable font & line height */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 16px;
        line-height: 1.7;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 16px;
    }
    p {
        margin-bottom: 1.1rem !important;
    }
    /* Formula block styling */
    .katex-display {
        margin: 1.2em 0 !important;
        padding: 0.9em 1.2em;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 8px;
        overflow-x: auto;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 8px 12px;
        border-radius: 8px;
    }
    /* Hide Streamlit exception debug buttons (Copy, Ask Google, Ask ChatGPT) */
    [data-testid="stExceptionActionButtons"],
    .stException > div:last-child {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


def normalize_latex(text: str) -> str:
    """
    Cleans LaTeX delimiters so KaTeX in Streamlit renders formulas crisply:
    - \\[ ... \\] -> $$ ... $$
    - \\( ... \\) -> $ ... $
    - Standalone bracketed lines [ ... ] -> $$ ... $$
    - Mathematical terms inside parentheses ( Q ), ( B \\times T \\times D ) -> $...$
    """
    if not text:
        return ""

    # 1. Standard display math: \[ ... \] -> $$ ... $$
    text = re.sub(r'\\\[([\s\S]*?)\\\]', r'$$\n\1\n$$', text)

    # 2. Standard inline math: \( ... \) -> $ ... $
    text = re.sub(r'\\\(([\s\S]*?)\\\)', r'$\1$', text)

    # 3. Standalone bracket lines [ ... ] that contain math indicators
    def _bracket_repl(m):
        inner = m.group(1).strip()
        indicators = ["=", "\\text", "\\frac", "\\times", "^", "_", "\\cdot", "\\left", "\\sum", "softmax", "ReLU"]
        if any(tok in inner for tok in indicators):
            return f"\n$$\n{inner}\n$$\n"
        return m.group(0)

    text = re.sub(r'(?m)^\s*\[\s*([^\n\]]+)\s*\]\s*$', _bracket_repl, text)

    # 4. Math variables or dimensions inside parentheses: ( Q ), ( K ), ( B \times T \times D ), ( x )
    def _paren_repl(m):
        inner = m.group(1).strip()
        # Single variable like ( Q ), ( K ), ( V ), ( x ), ( z )
        if len(inner) == 1 and inner.isalpha():
            return f"${inner}$"
        # Equations or dimension shapes like ( B \times T \times D ) or ( F(x) = x + G(x) )
        indicators = ["\\times", "\\cdot", "\\sum", "\\in", "=", "^", "_", "\\frac", "\\approx", "\\leq", "\\geq"]
        if any(tok in inner for tok in indicators):
            return f"${inner}$"
        return m.group(0)

    text = re.sub(r'\(\s*([A-Za-z0-9_\\^+\-*\/=, \t]+)\s*\)', _paren_repl, text)

    # 5. Clean excessive $$$ to $$
    text = re.sub(r'\${3,}', '$$', text)
    return text


def contextualize_query(current_query: str, history: list) -> str:
    """
    Augments follow-up questions with conversation context for accurate hybrid retrieval.
    Example: "What is its computational complexity?" -> "ResNet residual connections : What is its computational complexity?"
    """
    if not history or len(history) <= 1:
        return current_query

    followup_triggers = [
        "it", "its", "this", "these", "they", "that", "the same",
        "bu", "bunun", "bunu", "onun", "ona", "peki", "neden", "nasıl",
        "farkı", "avantajı", "dezavantajı", "karmaşıklığı", "nedir"
    ]
    words = set(current_query.lower().split())

    is_followup = len(current_query.split()) < 10 or bool(words.intersection(followup_triggers))

    if is_followup:
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
    st.title("🔬 Donanım & Bellek")
    
    # GPU Monitor
    gpu = get_gpu_status()
    if gpu.get("available"):
        st.markdown("**NVIDIA RTX 4050 Laptop (GPU)**")
        col1, col2 = st.columns(2)
        col1.metric("GPU Sıcaklık", f"{gpu['temp']} °C")
        col2.metric("GPU Yükü", f"{gpu['util']} %")
        vram_pct = int((gpu['used'] / gpu['total']) * 100)
        st.progress(vram_pct, text=f"VRAM: {gpu['used']} / {gpu['total']} MiB")
    else:
        st.info("GPU Telemetrisi Beklemede")

    # CPU & RAM Monitor
    cpu_percent = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    st.markdown("**İşlemci (CPU) & Sistem Belleği**")
    ccol1, ccol2 = st.columns(2)
    ccol1.metric("CPU Yükü", f"{cpu_percent} %")
    ccol2.metric("RAM", f"{ram.percent} %")
    st.progress(ram.percent / 100.0, text=f"RAM: {ram.used // (1024**2)} / {ram.total // (1024**2)} MiB")

    st.caption("ℹ️ *Laptop fanları hem CPU hem de GPU ısısına göre hızlanır. Gösterilen sıcaklık NVIDIA GPU çipinin sıcaklığıdır.*")

    # Unload Button
    if st.button("❄️ VRAM'i Boşalt (Modeli Kapat)", help="Modeli VRAM'den siler, laptop sıcaklığını ve güç tüketimini anında düşürür."):
        llm.unload()
        st.success("Model VRAM'den kaldırıldı, GPU boşa alındı.")
        st.rerun()

    st.divider()

    st.subheader("⚙️ Yanıt Derinliği & Mod")
    mode = st.radio(
        "Mod Seçimi:",
        [
            "🎓 Derin Akıl Yürütme & Ders Kitabı (Two-Stage CoT)",
            "⚡ Hızlı Akademik Soru-Cevap",
            "🎯 Devil's Advocate (/challenge)"
        ],
        index=0
    )

    st.divider()

    # Obsidian Pending Drafts
    drafts = approval_manager.list_drafts()
    st.subheader(f"📝 Obsidian Taslakları ({len(drafts)})")
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
st.title("Local Deep Learning Lab Assistant")
st.caption("Donanım: RTX 4050 6GB | Motor: Ollama qwen2.5:7b | Vektör Veritabanı: ChromaDB | Bellek: Obsidian Vault")

# Initialize session messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Merhaba! Ben yerel Derin Öğrenme Laboratuvar Asistanınızım. İndekslenmiş makalelerimiz (Transformer, ResNet, ViT, Coordinate Attention, Swin) hakkında teknik sorular sorabilir, iki aşamalı derin akıl yürütme (Two-Stage CoT) ile ders kitabı kalitesinde kapsamlı analizler alabilir veya yeni fikirlerinizi Devil's Advocate modunda test edebilirsiniz."
        }
    ]

# Render chat history with normalized LaTeX and Obsidian Save buttons
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(normalize_latex(msg["content"]))
        
        # If assistant message (and not the initial welcome message), provide Save to Vault button
        if msg["role"] == "assistant" and i > 0:
            btn_col, _ = st.columns([1, 4])
            with btn_col:
                if st.button("💾 Obsidian Vault'a Kaydet", key=f"save_draft_{i}", help="Bu cevabı Obsidian Vault/Drafts/ klasörüne taslak olarak kaydeder."):
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

# User input box
if prompt := st.chat_input("Bir soru sorun, hipotez belirtin veya konu anlatımı isteyin..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        # Check Mode
        if "Devil's Advocate" in mode or prompt.startswith("/challenge"):
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

        else:
            is_deep_cot = "Two-Stage CoT" in mode or prompt.startswith("/explain")
            clean_q = prompt.replace("/explain", "").strip()

            # Multi-turn context resolution: augment search query with prior conversational topic
            augmented_search_query = contextualize_query(clean_q, st.session_state.messages[:-1])

            # Step 1: Hybrid Retrieval
            with st.status("📚 Literatür ve yerel hafıza taranıyor...", expanded=False) as status:
                search_res = searcher.search(augmented_search_query, top_k=config.RETRIEVAL_TOP_K)
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

            # Display source citations box if available
            sources = search_res.get("results", [])
            if sources:
                with st.expander(f"📚 Referans Alınan Makale Parçaları ({len(sources)} Bölüm)"):
                    for s in sources:
                        st.markdown(f"- **{s.get('id')}** (Skor: `{s.get('score', 0):.2f}`) — *{s.get('title')}*")

            thought_plan = ""
            # Step 2: Stage 1 - Chain-of-Thought Deep Planning (if in Two-Stage CoT mode)
            if is_deep_cot:
                with st.status("🧠 1. Aşama: Derin Muhakeme & Çözüm Planı Çıkarılıyor (Chain-of-Thought)...", expanded=True) as status_cot:
                    cot_prompt_msgs = build_thought_plan_prompt(clean_q, safe_packed.get("paper_chunks", ""))
                    thought_plan = llm.generate(cot_prompt_msgs, max_tokens=450, temperature=0.1)
                    status_cot.update(label="✅ Akıl Yürütme Planı Hazırlandı. 2. Aşama: Kapsamlı Ders Yazılıyor...", state="complete")
                
                with st.expander("🧠 Akıl Yürütme ve Çözüm Planı (Chain-of-Thought Blueprint)", expanded=False):
                    st.markdown(normalize_latex(thought_plan))

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

            # Max tokens: 2500 for deep CoT mode, 1000 for quick mode
            target_max_tokens = config.TOKEN_BUDGET.get("max_answer", 2500) if is_deep_cot else 1000

            stream = llm.stream_generate(
                messages,
                max_tokens=target_max_tokens
            )
            full_response = st.write_stream(stream)
            clean_full = normalize_latex(full_response)
            st.session_state.messages.append({"role": "assistant", "content": clean_full})
            st.rerun()
