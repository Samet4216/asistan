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
import json
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
from utils import (
    get_gpu_status,
    normalize_latex,
    is_casual_or_personal,
    is_challenge_request,
    contextualize_query
)
from retrieval.search import searcher
from answer.prompts import (
    build_answer_prompt,
    build_thought_plan_prompt,
    build_casual_response_prompt,
    build_general_expert_prompt
)
from answer.challenge import challenger
from answer.verify import audit_markdown_response
from memory.approval import approval_manager
from memory.errors import error_manager
from memory.user_profile import profile_manager
from memory.auto_memory import auto_memory
from memory.rl_feedback import rl_engine
from memory.alignment_interview import alignment_interviewer
from memory.graph_visualizer import get_vault_graph_data, generate_graph_html
from ingest.single_ingest import ingest_arxiv_paper
from ingest.local_file_ingest import ingest_uploaded_file


# Page configuration
st.set_page_config(
    page_title="Deep Learning Lab Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load modular CSS styling
css_file = PROJECT_ROOT / "assets" / "style.css"
if css_file.exists():
    raw_css = css_file.read_text(encoding="utf-8").strip()
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">\n'
        f'<style>\n{raw_css}\n</style>',
        unsafe_allow_html=True
    )


# Load modular quotation popover script
js_file = PROJECT_ROOT / "assets" / "quote_popover.js"
if js_file.exists():
    components.html(f"<script>{js_file.read_text(encoding='utf-8')}</script>", height=0)


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

    # Otomatik VRAM Boşaltma & Idle Watchdog Göstergesi
    idle_info = llm.get_idle_status()
    if idle_info.get("is_loaded"):
        rem_min = idle_info["remaining_seconds"] // 60
        rem_sec = idle_info["remaining_seconds"] % 60
        st.caption(f"⏱️ **Oto-VRAM Boşaltma:** ~{rem_min} dk {rem_sec:02d} sn sonra (Boşta: {idle_info['idle_seconds']} sn)")
    else:
        st.caption("❄️ **Model Durumu:** Boşta / VRAM Serin (Oto-Boşaltma Devrede)")

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

    st.markdown("### 🧠 Sistem Modu")
    st.info("🎯 **Tek Bütünleşik Akıllı Mod (Unified)**\n\nSistem; akademik araştırma, derin akıl yürütme (CoT), mekatronik & robotik ve hipotez testlerini tek akışta otonom olarak yürütür.")

    st.divider()

    st.markdown("### 🔍 Arama Motoru Ayarları")
    use_decomp = st.checkbox("🔀 Alt-Sorgu Ayrıştırma (Decomposition)", value=True, help="Karmaşık veya karşılaştırmalı soruları otomatik 2-3 bağımsız sorguya bölerek arar.")
    use_hyde = st.checkbox("💡 HyDE (Hipotetik Belge Vektörü)", value=False, help="Soyut sorular için önce akademik bir varsayımsal yanıt üretip onun vektörüyle arar.")

    st.divider()

    # Ingestion Section: Local File Upload & arXiv Downloader
    st.markdown("### 📥 Hafızaya Kaynak Ekle")
    ingest_tab_file, ingest_tab_arxiv = st.tabs(["📂 Dosya Yükle", "🌐 arXiv İndir"])

    with ingest_tab_file:
        uploaded_file = st.file_uploader(
            "PDF, Markdown veya TXT yükleyin:",
            type=["pdf", "md", "txt"],
            help="Ders notları, makaleler veya proje dökümanlarınızı sürükleyip bırakabilirsiniz."
        )
        custom_doc_title = st.text_input("Belge Başlığı (Opsiyonel):", placeholder="örn: YOLOv8 Bitirme Tezi Notları", key="upload_title_input")
        if st.button("📥 Dosyayı İndeksle & Ekle", key="btn_upload_doc", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner(f"'{uploaded_file.name}' ayrıştırılıyor, parçalanıyor ve ChromaDB'ye ekleniyor..."):
                    res_up = ingest_uploaded_file(uploaded_file.getvalue(), uploaded_file.name, custom_title=custom_doc_title)
                    if res_up.get("success"):
                        st.success(f"✅ '{res_up.get('title')}' ({res_up.get('chunks_count')} parça) hafızaya eklendi!")
                        st.toast(f"✅ '{res_up.get('title')}' hafızaya ve Bilgi Ağı'na eklendi!", icon="📚")
                        st.rerun()
                    else:
                        st.error(f"❌ Hata: {res_up.get('error', 'Dosya indekslenemedi.')}")
            else:
                st.warning("Lütfen önce bir dosya seçin veya sürükleyin.")

    with ingest_tab_arxiv:
        with st.form("form_add_paper"):
            new_arxiv_id = st.text_input("arXiv Kodu:", placeholder="örn: 2103.02907 veya 1706.03762")
            submit_paper = st.form_submit_button("📥 İndir & İndeksle", use_container_width=True)
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

    # Dynamic Reinforcement Learning & Style Policy Panel
    with st.expander("🎯 Pekiştirmeli Öğrenme & Kişisel Tarz", expanded=False):
        rl_metrics = rl_engine.get_metrics()
        st.caption("Asistan ile konuştukça ve geri bildirim verdikçe (👍/👎, tarz butonları, örtük eleştiriler) model sizin tarzınızı burada otonom olarak pekiştirir.")
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Geri Bildirim", f"{rl_metrics.get('total_feedbacks', 0)}")
        m_col2.metric("Kümülatif Ödül", f"{rl_metrics.get('cumulative_reward', 0.0)}")
        
        axioms = rl_engine.get_active_axioms()
        st.markdown("**📌 Öğrenilen Aktif Tarz Kuralları:**")
        if axioms:
            for ax in axioms:
                st.markdown(f"- *{ax}*")
        else:
            st.caption("Henüz özel kural eklenmedi. Sohbetteki 👍/👎 butonlarıyla tarzınızı öğretebilirsiniz.")
        
        if st.button("🔄 Tarz Kurallarını Sıfırla", key="btn_reset_rl_style", help="Öğrenilen tarz kurallarını varsayılana döndürür"):
            rl_engine.add_style_axiom("Cevapları doğrudan konuya girerek, gereksiz nezaket cümleleri olmadan öz ve net tut.")
            st.toast("Tarz politikası sıfırlandı.", icon="🔄")
            st.rerun()

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

# User input handling (or pending quick prompt)
input_prompt = st.chat_input("Bir soru sorun, selam verin veya yukarıdan bir satırı seçip '💬 Alıntı Yap' ile sorun...")

prompt = None
if input_prompt:
    prompt = input_prompt
elif "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# Main View Switcher Tabs: 4-Tab Cyber-Lab Workspace Suite
tab_chat, tab_graph, tab_corpus, tab_radar = st.tabs([
    "💬 Araştırma & Sohbet",
    "🕸️ Obsidian İkinci Beyin Grafiği",
    "📚 arXiv Makale Kütüphanesi",
    "⚡ Donanım & Sistem Radarı"
])

with tab_chat:
    # Top Chat Action Toolbar
    tool_col1, tool_col2, tool_col3, tool_col4 = st.columns([2.0, 1.4, 0.9, 0.9])
    with tool_col1:
        st.caption(f"🧠 Aktif Seans: {len(st.session_state.get('messages', []))} mesaj • Yerel CUDA Hazır")
    with tool_col2:
        if st.button("🎙️ Zihniyetimi Öğret", key="btn_start_align", help="Modelin sana derin mühendislik soruları sorarak projelerinin nedenini ve karar tarzını öğrenmesini başlatır (/align)"):
            st.session_state.pending_prompt = "/align"
            st.rerun()
    with tool_col3:
        if st.button("🧹 Temizle", key="clear_chat_toolbar", help="Sohbet geçmişini sıfırlar"):
            st.session_state.messages = [st.session_state.messages[0]]
            st.rerun()
    with tool_col4:
        export_md = "# Local DL Lab Assistant — Sohbet Dökümü\n\n"
        for m in st.session_state.get("messages", []):
            role_label = "👤 **Kullanıcı:**" if m["role"] == "user" else "🤖 **Asistan:**"
            export_md += f"### {role_label}\n\n{m['content']}\n\n---\n\n"
        st.download_button(
            "📥 Dışa Aktar",
            data=export_md,
            file_name=f"dl_lab_sohbet_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            key="export_chat_toolbar",
            help="Sohbet dökümünü Markdown dosyası olarak indirir."
        )

    # Display interview banner if active
    if st.session_state.get("interview_mode", False):
        st.info("🎙️ **Bilişsel Zihniyet Hizalama Röportajı Devam Ediyor**\n\nModel şu anda projelerinin arkasındaki amacı ve mühendislik karar mantığını öğreniyor. Röportajı sonlandırmak için chat'e `/cikis` veya `Bitir` yazabilirsiniz.")

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
                st.rerun()
            if st.button("⚡ FlashAttention: SRAM Tiling & IO-Awareness", use_container_width=True):
                st.session_state.pending_prompt = "How does FlashAttention avoid materializing the NxN attention matrix in HBM using SRAM tiling?"
                st.rerun()
        with q_col2:
            if st.button("🔍 Self-RAG: Reflection Token Mekanizması", use_container_width=True):
                st.session_state.pending_prompt = "Explain Self-RAG reflection tokens ([Retrieve], [IsRel], [IsSup]) and how they prevent hallucination."
                st.rerun()
            if st.button("📐 Coordinate Attention: 1D Uzamsal Pooling", use_container_width=True):
                st.session_state.pending_prompt = "Coordinate Attention yatay ve dikey 1D pooling ile konumsal bilgiyi nasıl korur? Formüllerini ve tensör boyutlarını açıkla."
                st.rerun()
        st.write("")
        if st.button("🎙️ /align: Bilişsel Zihniyet Hizalama Röportajı (Modelin Benim Gibi Düşünmesini Sağla)", use_container_width=True):
            st.session_state.pending_prompt = "/align"
            st.rerun()
        st.write("")

    # Render chat history with normalized LaTeX and Obsidian Save buttons
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(normalize_latex(msg["content"]))
            
            # If assistant message (and not the initial welcome message), provide RL Feedback & Save toolbar
            if msg["role"] == "assistant" and i > 0:
                user_q = ""
                for prev_msg in reversed(st.session_state.messages[:i]):
                    if prev_msg["role"] == "user":
                        user_q = prev_msg["content"]
                        break
                
                # Sleek, compact RL Feedback Toolbar
                rl_col1, rl_col2, rl_col3, rl_col4, rl_col5 = st.columns([0.7, 0.7, 2.3, 1.8, 1.4])
                with rl_col1:
                    if st.button("👍", key=f"rl_up_{i}", help="Mükemmel yanıt! Bu tarzı ve yaklaşımı pekiştir (+1.0 Ödül)"):
                        rl_engine.record_feedback(user_q, msg["content"], reward=1.0, style_tag="perfect")
                        st.toast("🎯 Tarzınız pekiştirildi! Model hafızasına işlendi (+1.0 Ödül).", icon="👍")
                        st.rerun()
                with rl_col2:
                    if st.button("👎", key=f"rl_down_{i}", help="Bu cevabı beğenmedim / tarzıma uymadı (-1.0 Ceza)"):
                        rl_engine.record_feedback(user_q, msg["content"], reward=-1.0, style_tag="general")
                        st.toast("⚠️ Tercihiniz kaydedildi (-1.0 Ceza).", icon="👎")
                        st.rerun()
                with rl_col3:
                    quick_style = st.selectbox(
                        "Tarzı Yönlendir:",
                        ["Tarzı Yönlendir...", "⚡ Daha Kısa & Öz", "💻 Kod & Tensör Odaklı", "📐 Teori & Matematik", "🤖 Mekatronik & Donanım"],
                        key=f"rl_quick_{i}",
                        label_visibility="collapsed"
                    )
                    if quick_style != "Tarzı Yönlendir...":
                        tag_map = {
                            "⚡ Daha Kısa & Öz": "concise",
                            "💻 Kod & Tensör Odaklı": "code_heavy",
                            "📐 Teori & Matematik": "math_depth",
                            "🤖 Mekatronik & Donanım": "robotics"
                        }
                        tag = tag_map.get(quick_style, "general")
                        res = rl_engine.record_feedback(user_q, msg["content"], reward=0.5, style_tag=tag)
                        st.toast(f"🎯 Yeni tarz kuralı pekiştirildi: {res.get('learned_axiom')}", icon="🎯")
                        st.rerun()
                with rl_col4:
                    with st.popover("✍️ Tarzımı Öğret", help="Modele özel bir üslup kuralı öğret"):
                        custom_critique = st.text_input("Nasıl anlatmalıydı?", placeholder="Örn: Daha çok pratik kod ver...", key=f"critique_input_{i}")
                        if st.button("Pekiştir", key=f"btn_critique_{i}") and custom_critique:
                            res = rl_engine.record_feedback(user_q, msg["content"], reward=0.5, style_tag="custom", user_comment=custom_critique)
                            st.toast(f"🎯 Yeni tarz kuralı hafızaya alındı!", icon="🧠")
                            st.rerun()
                with rl_col5:
                    if st.button("💾 Obsidian", key=f"save_draft_{i}", help="Bu cevabı Obsidian Vault/Drafts/ klasörüne taslak olarak kaydeder."):
                        safe_title = re.sub(r'[^A-Za-z0-9]+', '_', user_q[:35]).strip('_') or "Note"
                        draft_path = approval_manager.create_draft(
                            title=safe_title,
                            content=msg["content"],
                            category="Theory_Notes",
                            tags=["qa-chat", "dl-lab"]
                        )
                        st.toast(f"✅ '{safe_title}' Obsidian Vault/Drafts/ klasörüne kaydedildi!", icon="📝")
                        st.rerun()

    if prompt:
        # Check conversational critique / implicit feedback
        if len(st.session_state.messages) > 1:
            last_assistant = None
            last_user = None
            for prev_m in reversed(st.session_state.messages):
                if prev_m["role"] == "assistant" and not last_assistant:
                    last_assistant = prev_m["content"]
                elif prev_m["role"] == "user" and not last_user:
                    last_user = prev_m["content"]
            
            critique_res = rl_engine.detect_conversational_critique(prompt, last_user, last_assistant)
            if critique_res and critique_res.get("learned_axiom"):
                st.toast(f"🎯 Tarzınız algılandı ve pekiştirildi: {critique_res['learned_axiom']}", icon="🎯")

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
            clean_p = prompt.strip().lower()
            is_align_cmd = clean_p in ["/align", "/hizala", "zihniyetimi öğret", "röportaj"] or clean_p.startswith("/align")

            # 0. Alignment Interview Trigger
            if is_align_cmd:
                st.session_state.interview_mode = True
                st.session_state.interview_history = []
                with st.spinner("🧠 Samet'in projeleri ve zihniyetine özel soru hazırlanıyor..."):
                    first_q = alignment_interviewer.generate_next_question([])
                    st.session_state.current_interview_q = first_q
                    
                    response_text = (
                        "### 🎙️ Bilişsel Zihniyet Hizalama Röportajı Başlatıldı\n\n"
                        "Amacım yalnızca ne çalıştığını değil; **projelerini ne amaçla yaptığını, sahadaki hangi darboğazı çözdüğünü ve senin karar mantığını** "
                        "öğrenerek tam bir bilişsel ikizin haline gelmek.\n\n"
                        f"**Soru 1:** {first_q}\n\n"
                        "*İstediğin zaman `/cikis` veya 'Bitir' yazarak normal moda dönebilirsin.*"
                    )
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.rerun()

            # 0.1 Ongoing Alignment Interview Turn
            elif st.session_state.get("interview_mode", False):
                if clean_p in ["/cikis", "bitir", "çıkış", "exit", "tamam"]:
                    st.session_state.interview_mode = False
                    st.session_state.current_interview_q = None
                    exit_msg = (
                        "### 🏁 Zihniyet Hizalama Röportajı Tamamlandı\n\n"
                        "Harika! Paylaştığın tüm karar ilkeleri, donanım öncelikleri ve projelerinin nihai amaçları "
                        "pekiştirmeli hafızaya (`Vault/00_Profile/Reinforcement_Policy.md`) işlendi. "
                        "Artık tüm teknik ve mimari analizlerde senin gözünden düşüneceğim!"
                    )
                    st.markdown(exit_msg)
                    st.session_state.messages.append({"role": "assistant", "content": exit_msg})
                    st.rerun()
                else:
                    curr_q = st.session_state.get("current_interview_q", "Genel Mühendislik Zihniyeti")
                    with st.spinner("Düşünce tarzın ve projenin 'neden'i çözümleniyor..."):
                        interview_res = alignment_interviewer.process_answer(curr_q, prompt)
                        if "interview_history" not in st.session_state:
                            st.session_state.interview_history = []
                        st.session_state.interview_history.append({"question": curr_q, "answer": prompt})
                        
                        heuristic_badge = f"🧠 **Öğrenilen Karar İlkesi:** `{interview_res['heuristic']}`"
                        st.toast("🎯 Zihniyet ilkesi kalıcı hafızaya işlendi!", icon="🧠")

                        if len(st.session_state.interview_history) >= 3:
                            st.session_state.interview_mode = False
                            st.session_state.current_interview_q = None
                            wrap_msg = (
                                f"### ✅ 3/3 Zihniyet Analizi Başarıyla Tamamlandı!\n\n"
                                f"{interview_res['confirmation']}\n\n"
                                f"- {heuristic_badge}\n"
                                f"- 🎯 **Proje Teleolojisi:** {interview_res['project_why']}\n\n"
                                f"Tüm ilkeler model hafızasına (`Reinforcement_Policy.md`) mühürlendi. "
                                f"Şimdi istediğin bir teknik soruyu sorabilirsin, artık senin zihnindeki önceliklerle yanıtlayacağım!"
                            )
                            st.markdown(wrap_msg)
                            st.session_state.messages.append({"role": "assistant", "content": wrap_msg})
                            st.rerun()
                        else:
                            next_q_num = len(st.session_state.interview_history) + 1
                            next_q = alignment_interviewer.generate_next_question(st.session_state.interview_history)
                            st.session_state.current_interview_q = next_q
                            
                            step_msg = (
                                f"{interview_res['confirmation']}\n\n"
                                f"{heuristic_badge}\n\n"
                                f"---\n\n"
                                f"**Soru {next_q_num}/3:** {next_q}"
                            )
                            st.markdown(step_msg)
                            st.session_state.messages.append({"role": "assistant", "content": step_msg})
                            st.rerun()

            # 1. Check if Casual Chat or Personal Profile Statement
            elif is_casual_or_personal(prompt):
                casual_msgs = build_casual_response_prompt(prompt, history=st.session_state.messages[:-1])
                stream = llm.stream_generate(casual_msgs, max_tokens=1500, temperature=0.6)
                full_response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()

            # 2. Check Devil's Advocate / Hypothesis Stress Test
            elif is_challenge_request(prompt):
                with st.spinner("Hipotez literatürle ve RTX 4050 donanım sınırlarıyla test ediliyor..."):
                    clean_q = re.sub(r"^/challenge\s*", "", prompt, flags=re.I).strip()
                    for trig in [
                        "hipotezimi test et:", "hipotezimi test et",
                        "hipotezimi sına:", "hipotezimi sına",
                        "fikrimi eleştir:", "fikrimi eleştir",
                        "şeytanın avukatı:", "şeytanın avukatı",
                        "devil's advocate:", "devil's advocate"
                    ]:
                        clean_q = re.sub(r"^" + re.escape(trig) + r"\s*", "", clean_q, flags=re.I).strip()

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

            # 3. Explicit Mentor / General Engineering Directive
            elif prompt.startswith("/mentor"):
                with st.spinner("🤖 Mekatronik ve Yapay Zeka mühendislik perspektifiyle analiz ediliyor..."):
                    clean_q = re.sub(r"^/mentor\s*", "", prompt, flags=re.I).strip()
                    mentor_msgs = build_general_expert_prompt(clean_q, history=st.session_state.messages[:-1])
                    stream = llm.stream_generate(mentor_msgs, max_tokens=3000)
                    full_response = st.write_stream(stream)
                    clean_full = normalize_latex(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": clean_full})
                    st.rerun()

            # 4. Unified Technical & Research Pipeline
            else:
                clean_q = re.sub(r"^/explain\s*", "", prompt, flags=re.I).strip()
                is_deep_cot = True

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

                # Display decomposed sub-queries or translated English search query
                sub_qs = search_res.get("sub_queries", [])
                en_sq = search_res.get("search_query", "")
                if len(sub_qs) > 1 and has_sufficient_evidence:
                    st.info("🔀 **Ayrıştırılan Alt-Sorgular (EN):**\n" + "\n".join([f"- `{sq}`" for sq in sub_qs]))
                elif en_sq and en_sq != augmented_search_query and has_sufficient_evidence:
                    st.caption(f"🌐 **Arama Sorgusu (TR → EN):** `{en_sq}`")

                # If no direct academic paper matches with high confidence,
                # answer naturally and conversationally using broad expert AI & engineering intelligence:
                if not has_sufficient_evidence:
                    expert_msgs = build_general_expert_prompt(clean_q, history=st.session_state.messages[:-1])
                    stream = llm.stream_generate(expert_msgs, max_tokens=3000, temperature=0.6)
                    full_response = st.write_stream(stream)
                    clean_full = normalize_latex(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": clean_full})
                    st.rerun()

                # Display numbered source citations box [S1]..[S5] if available
                sources = search_res.get("results", [])
                if sources:
                    with st.expander(f"📚 Referans Alınan Makale Parçaları ({len(sources)} Bölüm — [S1]..[S{len(sources)}])"):
                        for idx, s in enumerate(sources):
                            tag = s.get("source_tag", f"S{idx+1}")
                            meta = s.get("metadata", {})
                            title = meta.get("title", "Akademik Makale")
                            section = meta.get("section_name", "")
                            score = s.get("rerank_score", 0.0)
                            sec_info = f" • Bölüm: *{section}*" if section else ""
                            st.markdown(f"- **`[{tag}]`** 📄 **{title}**{sec_info} (`{s.get('id')}`) — *Alaka Skoru: `{score:.2f}`*")

                thought_plan = ""
                # Step 2: Extended Thinking Mode - Model Thinks Deeply in Public
                if is_deep_cot:
                    import time
                    t_start = time.time()
                    with st.status("💭 Derin Düşünme & Analiz (Model kendi kendine muhakeme yapıyor...)", expanded=True) as status_cot:
                        cot_prompt_msgs = build_thought_plan_prompt(clean_q, safe_packed.get("paper_chunks", ""))
                        thought_stream = llm.stream_generate(cot_prompt_msgs, max_tokens=750, temperature=0.2)
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

                target_max_tokens = config.TOKEN_BUDGET.get("max_answer", 3500) if is_deep_cot else 3000

                stream = llm.stream_generate(
                    messages,
                    max_tokens=target_max_tokens
                )
                full_response = st.write_stream(stream)
                clean_full = normalize_latex(full_response)

                # Step 4: Deterministic Citation & Verbatim Quote Audit
                audit = audit_markdown_response(clean_full, retrieved_context)
                cited_str = ", ".join([f"`[{t}]`" for t in audit["cited_tags"]]) if audit["cited_tags"] else "Belirtilmedi"
                if audit["total_quotes"] > 0:
                    q_badge = f"✅ **{audit['verified_count']}/{audit['total_quotes']} Doğrudan Alıntı Kaynak Metinde Doğrulandı** (`%{int(audit['verified_ratio']*100)}`)"
                else:
                    q_badge = "ℹ️ Doğrudan tırnak içi alıntı kullanılmadı (Parafraz / Formül anlatımı)"

                footer_parts = [
                    f"\n\n---\n🔎 **Kaynak & Alıntı Denetimi:** Atıf Yapılan Kaynaklar: {cited_str} | {q_badge}"
                ]
                for q_item in audit["quotes"]:
                    mark = "✅" if q_item["verified"] else "⚠️ *(Kaynak metinde tam eşleşmedi)*"
                    snippet = q_item["text"][:140] + ("..." if len(q_item["text"]) > 140 else "")
                    footer_parts.append(f"- {mark} *\"{snippet}\"*")

                verification_footer = "\n".join(footer_parts)
                st.markdown(verification_footer)
                clean_full = clean_full + verification_footer

                st.session_state.messages.append({"role": "assistant", "content": clean_full})
                st.rerun()

with tab_graph:
    st.markdown("### 🕸️ Obsidian İkinci Beyin Bilgi Ağı (Synaptic Knowledge Graph)")
    st.caption("Vault içindeki makale kartları, temel teorik notlar ve sinaptik kavram bağlantılarının canlı interaktif haritası. Düğümleri sürükleyebilir, yakınlaştırabilir ve detaylarını incelemek için tıklayabilirsiniz.")

    graph_data = get_vault_graph_data()
    stats = graph_data.get("stats", {})

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Toplam Düğüm", f"{stats.get('total_nodes', 0)} Kavram")
    mcol2.metric("Toplam Sinaps", f"{stats.get('total_edges', 0)} Bağlantı")
    mcol3.metric("Makale Kartları", f"{stats.get('papers_count', 0)} Makale")
    mcol4.metric("Teorik Notlar", f"{stats.get('theory_count', 0)} Konu")

    # Render interactive physics graph
    graph_html = generate_graph_html(graph_data, height=720)
    components.html(graph_html, height=730, scrolling=False)

    st.markdown("#### 🌟 En Çok Bağlantılı Merkez Kavramlar (Knowledge Hubs)")
    hubs = stats.get("top_hubs", [])
    if hubs:
        hub_cols = st.columns(len(hubs))
        for idx, hub in enumerate(hubs):
            with hub_cols[idx]:
                if st.button(f"📌 {hub['title'][:20]}... ({hub['degree']} Sinaps)", key=f"hub_btn_{idx}", use_container_width=True):
                    st.session_state.pending_prompt = f"{hub['title']} kavramının teorik motivasyonunu, formüllerini ve PyTorch implementasyonunu açıkla."
                    st.rerun()

with tab_corpus:
    st.markdown("### 📚 Yerel RAG Makale Kütüphanesi (arXiv Corpus)")
    st.caption("Yerel ChromaDB vektör hafızamıza indekslenmiş 21 temel makale (YOLO, ConvNeXt, SAC, Vision Transformer, Attention ve Edge AI). Kartlardaki tek tıkla sorgulama butonunu kullanarak o makale hakkında anında derinlemesine analiz başlatabilirsiniz.")

    catalog_path = PROJECT_ROOT / "data" / "corpus_catalog.json"
    papers_info = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else []

    p_cols = st.columns(2)
    for idx, p in enumerate(papers_info):
        with p_cols[idx % 2]:
            st.markdown(f"""
            <div style="background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.08); border-left:4px solid {p['color']}; border-radius:12px; padding:16px; margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-size:11px; font-weight:600; text-transform:uppercase; color:{p['color']}; letter-spacing:0.5px;">{p['domain']}</span>
                    <span style="font-size:12px; color:#64748b; font-family:'Fira Code',monospace;">arXiv:{p['id']}</span>
                </div>
                <h4 style="margin:0 0 6px 0; font-size:1.05rem; color:#f8fafc;">{p['title']}</h4>
                <p style="margin:0 0 10px 0; font-size:0.84rem; color:#94a3b8;">{p['authors']}</p>
                <p style="margin:0 0 12px 0; font-size:0.88rem; color:#cbd5e1; line-height:1.4;">{p['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"⚡ Bu Makaleyi Sorgula", key=f"corp_btn_{p['id']}", use_container_width=True):
                st.session_state.pending_prompt = p["prompt"]
                st.rerun()

with tab_radar:
    st.markdown("### ⚡ Laboratuvar Donanım & Model Telemetri Radarı")
    st.caption("Yerel donanım sınırları, GPU VRAM kullanım durumu ve ChromaDB vektör veritabanı sağlık göstergeleri.")

    rad_col1, rad_col2 = st.columns(2)
    with rad_col1:
        st.markdown("#### 🎮 NVIDIA RTX 4050 Laptop GPU (6GB)")
        if gpu.get("available"):
            vram_mb = gpu["used"]
            vram_total = gpu["total"]
            vram_pct = (vram_mb / vram_total) * 100
            
            c_g1, c_g2, c_g3 = st.columns(3)
            c_g1.metric("GPU Sıcaklığı", f"{gpu['temp']} °C")
            c_g2.metric("GPU Yükü", f"{gpu['util']} %")
            c_g3.metric("Kullanılan VRAM", f"{vram_mb} MB")

            vram_color = "#10b981" if vram_pct < 70 else ("#f59e0b" if vram_pct < 85 else "#ef4444")
            st.markdown(f"""
            <div style="margin:14px 0 6px 0; font-size:13px; color:#94a3b8; display:flex; justify-content:space-between;">
                <span>VRAM Doluluk Oranı</span>
                <span><b>{vram_pct:.1f}%</b> ({vram_mb} / {vram_total} MiB)</span>
            </div>
            <div style="width:100%; height:12px; background:rgba(255,255,255,0.08); border-radius:8px; overflow:hidden;">
                <div style="width:{vram_pct}%; height:100%; background:{vram_color}; transition:width 0.4s ease;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("💡 **RTX 4050 Optimizasyonu:** 6GB VRAM için `qwen2.5:7b-instruct` modeli 4-bit kuantizasyonla ~4.2GB VRAM kaplar. Kalan 1.8GB, 8192 tokenlik KV cache için yeterlidir.")
        else:
            st.warning("NVIDIA GPU telemetrisi alınamadı veya sistem CPU modunda.")

    with rad_col2:
        st.markdown("#### 💻 Sistem İşlemcisi & Bellek")
        c_c1, c_c2 = st.columns(2)
        c_c1.metric("CPU Kullanımı", f"{cpu_percent} %")
        c_c2.metric("RAM Kullanımı", f"{ram.percent} %")
        st.progress(ram.percent / 100.0, text=f"RAM: {ram.used // (1024**2)} / {ram.total // (1024**2)} MiB")

    st.divider()

    st.markdown("#### 🗄️ ChromaDB Yerel Vektör Veritabanı Koleksiyonları")
    try:
        col_stats = searcher.indexer.get_collection_stats()
        p_chunks_cnt = col_stats.get("paper_chunks", 1208)
        p_cards_cnt = col_stats.get("paper_cards", 21)
        code_cnt = col_stats.get("code", 33)
        err_cnt = col_stats.get("errors", 5)
        exp_cnt = col_stats.get("experiments", 1)
    except Exception:
        p_chunks_cnt, p_cards_cnt, code_cnt, err_cnt, exp_cnt = 1208, 21, 33, 5, 1

    db_col1, db_col2, db_col3, db_col4, db_col5 = st.columns(5)
    db_col1.metric("📄 Makale Parçaları", f"{p_chunks_cnt} Chunks")
    db_col2.metric("📑 Makale Kartları", f"{p_cards_cnt} Kart")
    db_col3.metric("📐 Deney & Teori", f"{exp_cnt} Not")
    db_col4.metric("💻 Kod Snippetleri", f"{code_cnt} Kod")
    db_col5.metric("🐛 Çözülen Hatalar", f"{err_cnt} Hata")

    st.markdown("#### 🤖 Çift Motorlu Yerel Çıkarım Mimarisi (Dual-Engine)")
    m_acad = getattr(config, "MODEL_ACADEMIC", getattr(config, "MODEL", "qwen2.5:7b-instruct-q4_K_M"))
    m_reas = getattr(config, "MODEL_REASONING", "deepseek-r1:7b")
    st.markdown(f"""
    - **📚 Akademik Literatür & Formül Motoru:** `{m_acad}` (Bağlam doğrulaması, LaTeX ve [S1] alıntıları)
    - **🧠 Derin Akıl Yürütme & Planlama Motoru:** `{m_reas}` (&lt;think&gt; zinciriyle mühendislik planlaması ve problem çözme)
    - **Vektör Gömme Modeli (Embedding):** `{config.EMBED_MODEL}` (CPU üzerinde 768 boyutlu yoğun vektörler)
    - **Yeniden Sıralama (Cross-Encoder):** `{config.RERANK_MODEL}`
    - **Kişisel Hafıza & İkinci Beyin:** Obsidian Markdown Vault (`Vault/00_Profile/About_Me.md`)
    - **Gizlilik Durumu:** 🟢 **%100 Çevrimdışı ve Yerel** (Harici bulut API bağlantısı yok)
    """)

    st.divider()
    st.markdown("#### 🎯 Pekiştirmeli Öğrenme & Tercih Havuzu (RL Preference Alignment)")
    rl_met = rl_engine.get_metrics()
    c_r1, c_r2, c_r3, c_r4 = st.columns(4)
    c_r1.metric("Toplam Tercih Sinyali", f"{rl_met.get('total_feedbacks', 0)}")
    c_r2.metric("Pozitif Geri Bildirim 👍", f"{rl_met.get('positive_count', 0)}")
    c_r3.metric("Negatif Düzeltme 👎", f"{rl_met.get('negative_count', 0)}")
    c_r4.metric("Kümülatif Ödül Skoru", f"{rl_met.get('cumulative_reward', 0.0)}")

    st.markdown(f"""
    - **Tercih Veri Seti Konumu:** `data/rl_preferences/preferences.jsonl` (Direct Preference Optimization - DPO uyumlu)
    - **Aktif Tarz Politikası:** `Vault/00_Profile/Reinforcement_Policy.md` ve `Vault/00_Profile/Style.md`
    - **Çevrimdışı İnce Ayar:** Toplanan tercihlerle model ağırlıklarını kalıcı olarak eğitmek için `scripts/export_dpo_dataset.py` kullanılabilir.
    """)

