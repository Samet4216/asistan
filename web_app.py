"""
Local Deep Learning Lab Assistant — Modern Web Chat Interface (Streamlit).
Provides a ChatGPT/Open-WebUI-grade browser interface with KaTeX math rendering,
real-time streaming generation, live RTX 4050 & CPU thermals, and Obsidian memory integration.
"""

import sys
import subprocess
from pathlib import Path

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
from answer.prompts import build_answer_prompt
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
        margin-bottom: 1.2rem !important;
    }
    /* Formula block styling */
    .katex-display {
        margin: 1.2em 0 !important;
        padding: 0.8em;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 8px;
        overflow-x: auto;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 8px 12px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


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

    st.subheader("⚙️ Yanıt Modu")
    mode = st.radio(
        "Ayrıntı Seviyesi Seçin:",
        ["Akademik Soru-Cevap (Hızlı & Kaynaklı)", "🎓 Detaylı Konu Anlatımı (/explain)", "🎯 Devil's Advocate (/challenge)"],
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
            "content": "Merhaba! Ben yerel Derin Öğrenme Laboratuvar Asistanınızım. İndekslenmiş makalelerimiz (Transformer, ResNet, ViT, Coordinate Attention, Swin) hakkında teknik sorular sorabilir, kapsamlı konu anlatımı isteyebilir veya yeni fikirlerinizi Devil's Advocate modunda test edebilirsiniz."
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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
                
                st.markdown(output)
                st.session_state.messages.append({"role": "assistant", "content": output})

        else:
            is_deep_dive = "Detaylı Konu Anlatımı" in mode or prompt.startswith("/explain")
            clean_q = prompt.replace("/explain", "").strip()

            # Step 1: Hybrid Retrieval
            with st.status("Literatür ve yerel hafıza taranıyor...", expanded=False) as status:
                search_res = searcher.search(clean_q, top_k=config.RETRIEVAL_TOP_K)
                retrieved_context = search_res.get("packed_context", "")
                has_sufficient_evidence = search_res.get("has_sufficient_evidence", False)
                error_evidence = error_manager.get_relevant_errors(clean_q)
                
                budget_parts = {
                    "error_history": error_evidence,
                    "paper_chunks": retrieved_context
                }
                packed_report = budget.pack_context(budget_parts)
                safe_packed = packed_report["packed"]
                status.update(label="Literatür tarandı, yanıt gerçek zamanlı üretiliyor...", state="complete")

            # Display source citations box if available
            sources = search_res.get("results", [])
            if sources:
                with st.expander(f"📚 Referans Alınan Makale Parçaları ({len(sources)} Bölüm)"):
                    for s in sources:
                        st.markdown(f"- **{s.get('id')}** (Skor: `{s.get('score', 0):.2f}`) — *{s.get('title')}*")

            # Step 2: Build prompt
            messages = build_answer_prompt(
                query=clean_q,
                packaged_context=safe_packed.get("paper_chunks", ""),
                telemetry_context="",
                git_diff="",
                error_context=safe_packed.get("error_history", ""),
                deep_dive=is_deep_dive
            )

            # Step 3: Stream output token-by-token
            stream = llm.stream_generate(
                messages,
                max_tokens=config.TOKEN_BUDGET.get("max_answer", 1200)
            )
            full_response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
