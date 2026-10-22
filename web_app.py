"""
Local Deep Learning Lab Assistant — Modern Web Chat Interface (Streamlit).
Provides a ChatGPT/Open-WebUI-grade browser interface with KaTeX math rendering,
syntax-highlighted copyable code blocks, live RTX 4050 thermals, and Obsidian memory integration.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import config
from answer.pipeline import pipeline
from answer.challenge import challenger
from memory.approval import approval_manager
import lang_guard


# Page configuration
st.set_page_config(
    page_title="Deep Learning Lab Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for ChatGPT-like Typography
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
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def get_gpu_status():
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
    st.title("🔬 Lab Hardware & Memory")
    
    # GPU Monitor
    gpu = get_gpu_status()
    if gpu.get("available"):
        st.subheader("RTX 4050 Laptop GPU")
        col1, col2 = st.columns(2)
        col1.metric("Temperature", f"{gpu['temp']} °C")
        col2.metric("GPU Load", f"{gpu['util']} %")
        vram_pct = int((gpu['used'] / gpu['total']) * 100)
        st.progress(vram_pct, text=f"VRAM: {gpu['used']} / {gpu['total']} MiB")
    else:
        st.info("GPU Telemetry Idle")

    st.divider()

    st.subheader("⚙️ Query Mode")
    mode = st.radio(
        "Select Response Depth:",
        ["Standard Academic Q&A", "🎓 Deep-Dive Tutorial (/explain)", "🎯 Devil's Advocate (/challenge)"],
        index=0
    )

    st.divider()

    # Obsidian Pending Drafts
    drafts = approval_manager.list_drafts()
    st.subheader(f"📝 Obsidian Drafts ({len(drafts)})")
    if drafts:
        for d in drafts[:5]:
            with st.expander(f"{d['title'][:25]}"):
                st.caption(f"Category: {d['category']} | Created: {d['created_at'][:16]}")
                st.write(d["preview"])
                if st.button("✓ Approve Note", key=f"app_{d['id']}"):
                    approval_manager.approve_draft(d["id"])
                    st.success("Moved to permanent Vault & indexed!")
                    st.rerun()
    else:
        st.caption("No pending drafts in Vault/Drafts/")


# --- MAIN CHAT AREA ---
st.title("Local Deep Learning Lab Assistant")
st.caption("Hardware: RTX 4050 6GB | Engine: Ollama qwen2.5:7b | Embeddings: BAAI/bge-base-en-v1.5 (CPU) | Storage: Obsidian Vault")

# Initialize session messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your local Deep Learning Lab Assistant. Ask technical questions about our indexed papers (Transformer, ResNet, ViT, Coordinate Attention, Swin), request a deep-dive tutorial, or stress-test your ideas with Devil's Advocate mode."
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input box
if prompt := st.chat_input("Ask a question, propose a hypothesis, or enter a topic..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving literature, checking past errors, and synthesizing..."):
            if "Devil's Advocate" in mode or prompt.startswith("/challenge"):
                clean_q = prompt.replace("/challenge", "").strip()
                res = challenger.challenge_hypothesis(clean_q)
                
                verdict = res.get("verdict", "HIGH_RISK")
                v_color = "🔴" if verdict == "REJECT" else ("🟡" if verdict == "HIGH_RISK" else "🟢")
                
                output = f"### {v_color} Verdict: **{verdict}**\n\n"
                output += f"**Executive Summary:** {res.get('summary')}\n\n"
                output += "#### Critical Vulnerabilities & Literature Conflicts:\n"
                for c in res.get("criticisms", []):
                    output += f"- **{c.get('vulnerability')}:** {c.get('evidence_or_rationale')}\n"
                output += f"\n#### ⚠️ Hardware & VRAM Bottlenecks (RTX 4050 6GB):\n{res.get('hardware_and_vram_risk')}\n\n"
                output += f"#### 🧪 Minimal Disproving Experiment:\n{res.get('minimal_disproving_experiment')}"
                
            elif "Deep-Dive" in mode or prompt.startswith("/explain"):
                clean_q = prompt.replace("/explain", "").strip()
                res = pipeline.run(clean_q, top_k=config.RETRIEVAL_TOP_K, deep_dive=True)
                output = res.get("answer", "")
            else:
                res = pipeline.run(prompt, top_k=config.RETRIEVAL_TOP_K, deep_dive=False)
                output = ""
                
                # Render quotes if verified
                quotes = res.get("quotes", [])
                if quotes:
                    output += "> **📚 Grounded Citations & Verbatim Quotes:**\n"
                    for q in quotes:
                        output += f"> - *\"{q.get('text')}\"* — `[{q.get('source')}]`\n"
                    output += "\n---\n\n"
                
                output += res.get("answer", "")
                
                takeaways = res.get("key_takeaways", [])
                if takeaways:
                    output += "\n\n#### 💡 Key Takeaways:\n"
                    for t in takeaways:
                        output += f"- {t}\n"

            st.markdown(output)
            st.session_state.messages.append({"role": "assistant", "content": output})
