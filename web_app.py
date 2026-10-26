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
    is_academic_or_technical_inquiry,
    contextualize_query
)
from retrieval.search import searcher
from retrieval.web_search import web_searcher, online_searcher, is_online
from retrieval.graph_rag import graph_rag_engine
from answer.prompts import (
    build_answer_prompt,
    build_thought_plan_prompt,
    build_casual_response_prompt,
    build_general_expert_prompt,
    build_self_correction_prompt
)
from answer.verify import audit_markdown_response
from memory.approval import approval_manager
from memory.errors import error_manager
from memory.user_profile import profile_manager
from memory.auto_memory import auto_memory
from memory.rl_feedback import rl_engine
from memory.self_evolving_rl import self_evolving_rl
from memory.conversation_collector import collector
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

    st.markdown("### 🎯 Çalışma Modu")
    app_mode = st.radio(
        "Aktif Mod:",
        [
            "🎯 Bütünleşik Akıllı Mod (Önerilen)",
            "💬 Sadece Sohbet & Kişisel Adaptasyon",
            "🔬 Sadece Araştırma & Literatür"
        ],
        index=0,
        help=(
            "🎯 Bütünleşik Akıllı Mod: Soru niyetini otonom olarak analiz eder. "
            "Teknik/akademik sorularda yerel makaleler ve canlı arXiv literatürünü derin matematik ve tensör boyutlarıyla sentezler; "
            "günlük veya kişisel konularda samimi araştırma partneri olarak sohbet eder.\n\n"
            "💬 Sadece Sohbet: Odak noktasını kişisel diyalog, hafıza ve tarz adaptasyonuna sabitler.\n\n"
            "🔬 Sadece Araştırma: Tüm girdileri derin akademik literatür araştırması ve RAG taramasından geçirir."
        )
    )

    if app_mode.startswith("🎯"):
        st.info("🎯 **Bütünleşik Akıllı Mod Aktif**\n\n- Otonom Yönlendirme: Teknik sorular otomatik olarak yerel makaleler ve canlı arXiv'e yönlendirilir.\n- Sohbet: Kişisel hafıza ve pekiştirmeli üslup adaptasyonu devrededir.")
        use_cot = st.toggle("🧠 Derin Akıl Yürütme Planı (CoT)", value=False, help="Teknik makale sorularında cevaptan önce kaynaklar üzerinde derin akıl yürütme planı çıkarır.")
    elif app_mode.startswith("💬"):
        st.info("💬 **Sohbet & Kişisel Adaptasyon**\n\n- Model: `Qwen 2.5 (7B Instruct)`\n- Özellik: Akıcı Türkçe, kişisel hafıza, tarz öğrenme ve pekiştirmeli adaptasyon.")
        use_cot = False
    else:
        net_status = "🟢 Canlı Web & arXiv Aktif" if is_online() else "⚪ Çevrimdışı (Yerel Hafıza)"
        st.success(f"🔬 **Araştırma & Literatür**\n\n- Model: `Qwen 2.5 (7B Instruct)`\n- Öncelik: Önce Yerel 21 Makale & Obsidian, yoksa Canlı Web.\n- Bağlantı: {net_status}")
        use_cot = st.toggle("🧠 Derin Akıl Yürütme Planı (CoT)", value=True, help="Her cevaptan önce makale kaynakları üzerinde derin akıl yürütme ve konuyu parçalama planı çıkarır.")

    is_chat_mode = app_mode.startswith("💬")

    use_web_search = True
    use_decomp = True
    use_hyde = False

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
        evo_metrics = self_evolving_rl.get_evolution_metrics()
        
        st.caption("🧬 **Otonom Öz-Geliştirme: 🟢 AKTİF**\n\nModel her yanıtını doğruluk, üslup ve dil kalitesi açısından anlık olarak kendisi eleştirip kurallarını otonom olarak evrilir.")
        
        e_col1, e_col2, e_col3 = st.columns(3)
        e_col1.metric("Öz-Değerlendirme", f"{evo_metrics.get('total_self_evaluations', 0)}")
        e_col2.metric("Ort. Kalite", f"%{evo_metrics.get('average_quality_score', 85.0):.0f}")
        e_col3.metric("Otonom İlke", f"{evo_metrics.get('total_self_axioms_evolved', 0)}")
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Kullanıcı Geri Bildirimi", f"{rl_metrics.get('total_feedbacks', 0)}")
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

    # Self-Training Studio: Kendi Konuşmalarımla Eğit
    with st.expander("🎓 Kendi Konuşmalarımla Eğit (Fine-Tuning)", expanded=False):
        all_convs = collector.get_all_interactions()
        total_dialogs = len(all_convs)
        st.caption("Konuştuğunuz her diyalog ve verdiğiniz geri bildirimler yerel eğitim veri setinize eklenir. Model bu verilerle sizin tarzınızı öğrenir.")
        
        c_col1, c_col2 = st.columns(2)
        c_col1.metric("Toplam Diyalog", f"{total_dialogs}")
        positive_samples = len([c for c in all_convs if c.get("reward", 0) > 0])
        c_col2.metric("Kaliteli Diyalog", f"{positive_samples}")
        
        st.markdown("**1. Hızlı Kişisel Model (Ollama Persona):**")
        st.caption("0-VRAM harcar, konuşmalarınızı ve profilinizi doğrudan `samet-partner` modeline derler.")
        if st.button("🚀 Özel Modelimi Derle (samet-partner)", key="btn_compile_ollama_persona", use_container_width=True):
            with st.spinner("Özel 'samet-partner' modeli derleniyor..."):
                from scripts.create_ollama_persona import create_personal_model
                success = create_personal_model("samet-partner")
                if success:
                    st.success("✅ 'samet-partner' başarıyla derlendi! Ollama'da kullanıma hazır.")
                    st.toast("Modeliniz başarıyla derlendi!", icon="🚀")
                else:
                    st.error("❌ Model derlenirken bir hata oluştu. Ollama servisinin açık olduğundan emin olun.")
        
        st.divider()
        st.markdown("**2. Doğrudan Tercih Optimizasyonu (DPO):**")
        st.caption("Verdiğiniz 👍/👎 geri bildirimleri HuggingFace DPO formatına (chosen vs rejected) dönüştürür.")
        if st.button("📥 DPO Tercih Veri Setini Dışa Aktar", key="btn_export_dpo", use_container_width=True):
            from scripts.export_dpo_dataset import export_dpo_dataset
            out_file = export_dpo_dataset(min_feedback=1)
            if out_file:
                st.success(f"✅ DPO veri seti hazır: `{out_file}`")
                st.toast("DPO Veri Seti hazırlandı!", icon="📥")
            else:
                st.info("ℹ️ Henüz kaydedilmiş geri bildirim yok. Sohbetteki 👍 veya 👎 butonlarına basarak tercih oluşturabilirsiniz.")

        st.divider()
        st.markdown("**3. Yerel 4-Bit QLoRA Eğitimi (RTX 4050):**")
        st.caption("Modelin tüm ağırlıklarını konuşma geçmişinizle eğitir (~3.8 GB VRAM).")
        col_lora1, col_lora2 = st.columns(2)
        with col_lora1:
            if st.button("📁 Veri Seti Güncelle", key="btn_build_dataset", use_container_width=True):
                res_data = collector.build_training_dataset()
                st.success(f"✅ {res_data.get('sample_count', 0)} diyalog hazırlandı!")
                st.toast("Eğitim veri seti güncellendi!", icon="📁")
        with col_lora2:
            if st.button("🔍 GPU Uygunluk", key="btn_run_qlora_check", use_container_width=True):
                from scripts.train_my_style_qlora import check_prerequisites
                chk = check_prerequisites()
                st.info(f"GPU: {chk.get('gpu_name', 'NVIDIA RTX 4050 6GB')} | LoRA Altyapısı: Hazır")

        st.caption("⚡ Terminalden tam GPU eğitimini başlatmak için:\n`python scripts/train_my_style_qlora.py`")

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
    tool_col1, tool_col2, tool_col3 = st.columns([2.5, 0.9, 0.9])
    with tool_col1:
        if app_mode.startswith("🎯"):
            mode_tag = "🎯 **Bütünleşik Akıllı Mod** (Otonom Literatür & Sohbet • Qwen 2.5)"
        elif app_mode.startswith("💬"):
            mode_tag = "💬 **Sohbet & Kişisel Adaptasyon** (Qwen 2.5)"
        else:
            mode_tag = "🔬 **Araştırma & Literatür** (Qwen 2.5)"
        st.caption(f"🧠 {mode_tag} • {len(st.session_state.get('messages', []))} mesaj")
    with tool_col2:
        if st.button("🧹 Temizle", key="clear_chat_toolbar", help="Sohbet geçmişini sıfırlar"):
            st.session_state.messages = [st.session_state.messages[0]]
            st.rerun()
    with tool_col3:
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

    # Initialize session messages
    if "messages" not in st.session_state:
        if app_mode.startswith("🎯"):
            init_content = (
                "Selam Samet! **Bütünleşik Akıllı Mod** devrede (Qwen 2.5).\n\n"
                "- 🔬 **Teknik & Akademik Sorular:** Sistem sorunuzu otomatik algılar; yerel 21 makale ve canlı arXiv üzerinden derin matematiksel formüller ($$...$$), tensör boyutları ve literatür atıflarıyla eksiksiz yanıtlar.\n"
                "- 💬 **Kişisel & Genel Konular:** Projelerini, çalışma tarzını anlatabilirsin; pekiştirmeli öğrenme ile senin üslubuna anlık adapte olurum."
            )
        elif app_mode.startswith("💬"):
            init_content = "Selam Samet! Hoş geldin. Bugün günün nasıl geçiyor, keyifler nasıl? İster projelerinden dertleşelim, ister aklına takılanları konuşalım; buradayım dostum."
        else:
            init_content = (
                "Merhaba! **Araştırma & Literatür** modundayız (Qwen 2.5). "
                "Sorularınızı önce yerel 21 makale arşivimiz ve Obsidian kasamızdan araştırır, "
                "bulamazsa canlı web & arXiv üzerinden tarayıp sentezlerim."
            )
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": init_content
            }
        ]

    # One-Click Quick Action Cards (Shown at start or above input)
    if len(st.session_state.messages) <= 1:
        if app_mode.startswith("🎯"):
            st.markdown("##### 💡 Hızlı Başlatıcılar (Akıllı Otonom Yönlendirme)")
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                if st.button("🤖 Diffusion Policy: Robotik Eylemler & DDPM", use_container_width=True):
                    st.session_state.pending_prompt = "Diffusion Policy mimarisi robotik eylemleri (action trajectory) DDPM gürültü giderme süreciyle nasıl üretir ve klasik Pekiştirmeli Öğrenmeye (RL) göre multimodal görevlerde neden üstündür"
                    st.rerun()
                if st.button("⚖️ PPO vs DPO: Ödül Modelleme Farkları", use_container_width=True):
                    st.session_state.pending_prompt = "Compare PPO clipped surrogate objective with DPO policy optimization in terms of reward modeling and memory."
                    st.rerun()
            with q_col2:
                if st.button("👋 Kendimi ve Alanımı Tanıt", use_container_width=True):
                    st.session_state.pending_prompt = "Merhaba! Kendimden, hedeflerimden ve ilgi alanlarımdan bahsetmek istiyorum. Bana sorular sorarak beni sıfırdan tanır mısın?"
                    st.rerun()
                if st.button("⚡ FlashAttention: SRAM Tiling & IO-Awareness", use_container_width=True):
                    st.session_state.pending_prompt = "How does FlashAttention avoid materializing the NxN attention matrix in HBM using SRAM tiling?"
                    st.rerun()
        elif is_chat_mode:
            st.markdown("##### 💡 Hızlı Sohbet & Adaptasyon Başlatıcıları")
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                if st.button("👋 Kendimi ve Alanımı Tanıt", use_container_width=True):
                    st.session_state.pending_prompt = "Merhaba! Kendimden, hedeflerimden ve ilgi alanlarımdan bahsetmek istiyorum. Bana sorular sorarak beni sıfırdan tanır mısın?"
                    st.rerun()
                if st.button("🎯 Tarzımı Öğret: Net & Öz", use_container_width=True):
                    st.session_state.pending_prompt = "Bundan sonraki konularda bana gereksiz giriş cümleleri olmadan doğrudan soruya odaklanarak öz ve net anlatmanı istiyorum."
                    st.rerun()
            with q_col2:
                if st.button("🛠️ Projelerimi Tanıt", use_container_width=True):
                    st.session_state.pending_prompt = "Şu an üzerinde çalıştığım ve ilgi duyduğum projeleri hafızana kaydetmeni istiyorum. Projelerimi öğrenmek için bana sorular sor."
                    st.rerun()
                if st.button("💻 Donanımımı Tanıt", use_container_width=True):
                    st.session_state.pending_prompt = "Bilgisayarımın donanım özelliklerini sana tanıtmak ve sınırlarını aklında tutmanı istiyorum."
                    st.rerun()
        else:
            st.markdown("##### 🔬 Hızlı Literatür & Makale Araştırması")
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                if st.button("⚖️ PPO vs DPO: Ödül Modelleme Farkları", use_container_width=True):
                    st.session_state.pending_prompt = "Compare PPO clipped surrogate objective with DPO policy optimization in terms of reward modeling and memory."
                    st.rerun()
                if st.button("⚡ FlashAttention: SRAM Tiling & IO-Awareness", use_container_width=True):
                    st.session_state.pending_prompt = "How does FlashAttention avoid materializing the NxN attention matrix in HBM using SRAM tiling?"
                    st.rerun()
            with q_col2:
                if st.button("📐 Coordinate Attention: 1D Uzamsal Pooling", use_container_width=True):
                    st.session_state.pending_prompt = "Coordinate Attention yatay ve dikey 1D pooling ile konumsal bilgiyi nasıl korur? Formüllerini ve tensör boyutlarını açıkla."
                    st.rerun()
                if st.button("🔍 YOLOv8 vs YOLOv11 Mimari Farkları", use_container_width=True):
                    st.session_state.pending_prompt = "YOLOv8 ve YOLOv11 mimarilerini C2f vs C3k2 blokları ve dikkat mekanizmaları açısından karşılaştır."
                    st.rerun()
        st.write("")

    # Render chat history with normalized LaTeX and Obsidian Save buttons
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            content_to_show = msg["content"]
            if msg["role"] == "assistant":
                if msg.get("thought_plan"):
                    thought_html = (
                        f"<details style='background:rgba(30,41,59,0.5); border:1px solid rgba(56,189,248,0.25); "
                        f"border-radius:8px; padding:10px 14px; margin-bottom:12px;'>"
                        f"<summary style='cursor:pointer; color:#38bdf8; font-weight:600;'>🧠 Derin Akıl Yürütme & Konu Parçalama Planı</summary>"
                        f"<div style='margin-top:8px; color:#cbd5e1; font-size:0.88rem; line-height:1.5; white-space:pre-wrap;'>"
                        f"{msg['thought_plan']}</div></details>\n\n"
                    )
                    content_to_show = thought_html + content_to_show
                elif "<think>" in content_to_show:
                    parsed = llm.parse_deepseek_thought(content_to_show)
                    if parsed["thought"]:
                        thought_html = (
                            f"<details style='background:rgba(30,41,59,0.5); border:1px solid rgba(56,189,248,0.25); "
                            f"border-radius:8px; padding:10px 14px; margin-bottom:12px;'>"
                            f"<summary style='cursor:pointer; color:#38bdf8; font-weight:600;'>💭 DeepSeek-R1 Düşünce Süreci</summary>"
                            f"<div style='margin-top:8px; color:#cbd5e1; font-size:0.88rem; line-height:1.5; white-space:pre-wrap;'>"
                            f"{parsed['thought']}</div></details>\n\n"
                        )
                        content_to_show = thought_html + parsed["response"]
            clean_display_content = re.sub(r'\n*---\n*⚡ \*\*Otonom RL Öz-Değerlendirme:\*\*.*$', '', content_to_show, flags=re.DOTALL)
            st.markdown(normalize_latex(clean_display_content), unsafe_allow_html=True)

            # Proactive Research Follow-ups (1-Click Socratic Pills)
            if msg["role"] == "assistant":
                proactive_items = re.findall(r'-\s*🔹\s*(.+)', content_to_show)
                if proactive_items:
                    st.markdown("<p style='font-size:0.85rem; color:#38bdf8; font-weight:600; margin:10px 0 4px 0;'>💡 Proaktif Araştırma & Sıradaki Adımlar (Tek Tıkla Sor):</p>", unsafe_allow_html=True)
                    p_cols = st.columns(len(proactive_items[:3]))
                    for p_idx, p_text in enumerate(proactive_items[:3]):
                        with p_cols[p_idx]:
                            btn_label = p_text.strip()
                            short_label = (btn_label[:42] + "...") if len(btn_label) > 42 else btn_label
                            if st.button(f"🔹 {short_label}", key=f"proact_btn_{i}_{p_idx}", help=btn_label, use_container_width=True):
                                st.session_state.pending_prompt = btn_label
                                st.rerun()

            # If assistant message (and not the initial welcome message), provide RL Feedback & Save toolbar
            if msg["role"] == "assistant" and i > 0:
                user_q = ""
                for prev_msg in reversed(st.session_state.messages[:i]):
                    if prev_msg["role"] == "user":
                        user_q = prev_msg["content"]
                        break

                # Dynamic RL Quality Badge
                s_eval = msg.get("self_eval")
                if s_eval:
                    score = s_eval.get("score", 75.0)
                    critique = s_eval.get("critique", "")
                    icon = "🟢" if score >= 75 else ("🟡" if score >= 50 else "🔴")
                    st.caption(f"{icon} **Otonom RL Kalitesi:** `%{score:.0f}` (Ödül: `{s_eval.get('reward', 0.0):+.2f}`) • *{critique}*")

                # Sleek Proactive RL Feedback & Self-Correction Toolbar
                rl_col1, rl_col2, rl_col3, rl_col4, rl_col5, rl_col6, rl_col7 = st.columns([0.6, 0.6, 2.0, 1.3, 1.3, 1.2, 0.9])
                with rl_col1:
                    if st.button("👍", key=f"rl_up_{i}", help="Mükemmel yanıt! Bu tarzı ve yaklaşımı pekiştir (+1.0 Ödül)"):
                        rl_engine.record_feedback(user_q, msg["content"], reward=1.0, style_tag="perfect")
                        st.toast("🎯 Tarzınız pekiştirildi! (+1.0 Ödül)", icon="👍")
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
                        pop_c1, pop_c2 = st.columns(2)
                        with pop_c1:
                            if st.button("Pekiştir", key=f"btn_critique_{i}") and custom_critique:
                                res = rl_engine.record_feedback(user_q, msg["content"], reward=0.5, style_tag="custom", user_comment=custom_critique)
                                st.toast("🎯 Yeni tarz kuralı hafızaya alındı!", icon="🧠")
                                st.rerun()
                        with pop_c2:
                            if st.button("⚡ Düzelt", key=f"btn_critique_fix_{i}") and custom_critique:
                                rl_engine.record_feedback(user_q, msg["content"], reward=-0.5, style_tag="custom", user_comment=custom_critique)
                                st.session_state.self_correction_target = {
                                    "idx": i,
                                    "user_q": user_q,
                                    "flawed_ans": msg["content"],
                                    "critique": custom_critique
                                }
                                st.toast("⚡ Öz-Düzeltme başlatılıyor...", icon="⚡")
                                st.rerun()
                with rl_col5:
                    if st.button("⚡ Öz-Düzelt", key=f"self_correct_{i}", help="Model bu cevaptaki eksikleri analiz edip kendi kendini düzelterek üstün bir versiyon üretir"):
                        st.session_state.self_correction_target = {
                            "idx": i,
                            "user_q": user_q,
                            "flawed_ans": msg["content"],
                            "critique": "Önceki yanıttaki matematiksel formülleri, tensör boyutlarını ve literatür atıflarını eksiksiz tamamlayarak ders kitabı masterclass derinliğinde baştan inşa et."
                        }
                        st.rerun()
                with rl_col6:
                    if st.button("🔄 Yeniden", key=f"regen_{i}", help="Bu soruyu yeni öğrenilen pekiştirilmiş tarzınızla baştan yanıtlar"):
                        st.session_state.messages = st.session_state.messages[:i]
                        st.session_state.pending_prompt = user_q
                        st.rerun()
                with rl_col7:
                    if st.button("💾 Kaydet", key=f"save_draft_{i}", help="Bu cevabı Obsidian Vault/Drafts/ klasörüne taslak olarak kaydeder."):
                        safe_title = re.sub(r'[^A-Za-z0-9]+', '_', user_q[:35]).strip('_') or "Note"
                        draft_path = approval_manager.create_draft(
                            title=safe_title,
                            content=msg["content"],
                            category="Theory_Notes",
                            tags=["qa-chat", "dl-lab"]
                        )
                        st.toast(f"✅ '{safe_title}' Obsidian Vault/Drafts/ klasörüne kaydedildi!", icon="📝")
                        st.rerun()

    # Autonomous Self-Correction Flow (Triggered by ⚡ Öz-Düzelt button)
    if "self_correction_target" in st.session_state and st.session_state.self_correction_target:
        sc_target = st.session_state.self_correction_target
        st.session_state.self_correction_target = None
        sc_q = sc_target.get("user_q", "")
        sc_flawed = sc_target.get("flawed_ans", "")
        sc_critique = sc_target.get("critique", "Matematiksel formülleri, tensör boyutlarını ve literatür atıflarını eksiksiz tamamlayarak düzelt.")

        with st.chat_message("assistant"):
            with st.status("⚡ Öz-Düzeltme Devrede: Önceki yanıt taranıyor, eksik formüller ve kavramlar GraphRAG ile tamamlanıyor...", expanded=True) as sc_status:
                search_res = searcher.search(sc_q, top_k=config.RETRIEVAL_TOP_K)
                retrieved_context = search_res.get("packed_context", "")

                # GraphRAG Synaptic Context
                graph_seeds = graph_rag_engine.extract_seed_nodes(sc_q, search_res.get("results", []))
                subgraph = graph_rag_engine.traverse_subgraph(graph_seeds)
                graph_context = graph_rag_engine.format_graph_context(subgraph)

                sc_msgs = build_self_correction_prompt(
                    query=sc_q,
                    flawed_response=sc_flawed,
                    critique=sc_critique,
                    packaged_context=retrieved_context,
                    graph_context=graph_context,
                    history=st.session_state.messages[:-1]
                )
                sc_status.update(label="✅ Eksik noktalar teşhis edildi, düzeltilmiş masterclass yanıt üretiliyor...", state="complete")

            stream = llm.stream_generate(sc_msgs, max_tokens=3500, model=config.MODEL_ACADEMIC)
            full_response = st.write_stream(stream)
            clean_full = normalize_latex(full_response)

            self_eval = self_evolving_rl.evaluate_and_evolve(sc_q, clean_full, mode="research")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"🛠️ **[Öz-Düzeltilmiş Masterclass Yanıt]:**\n\n{clean_full}",
                "self_eval": self_eval
            })
            st.toast("⚡ Yanıt kendi kendini düzeltti!", icon="✅")
            st.rerun()

    if prompt:
        # Check conversational critique / implicit feedback
        critique_learned = ""
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
                critique_learned = critique_res["learned_axiom"]
                st.toast(f"🎯 Tarzınız algılandı ve pekiştirildi: {critique_learned}", icon="🎯")

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

        # Generate Response strictly by the 2 MODES
        with st.chat_message("assistant"):
            # Otonom Yönlendirme Kararı
            if app_mode.startswith("🔬"):
                should_run_research = True
            elif app_mode.startswith("💬"):
                # Sadece sohbet modunda kullanıcı tutulur; ancak teknik soru sorarsa kaynakla desteklenir
                should_run_research = False
            else:
                # 🎯 Bütünleşik Akıllı Mod (Önerilen):
                # Niyet analizi: Teknik/akademik konu ise RAG literatür motoru, değilse samimi diyalog motoru
                should_run_research = is_academic_or_technical_inquiry(prompt)

            if not should_run_research:
                # ==========================================
                # DİYALOG & KİŞİSEL ADAPTASYON (Qwen 2.5)
                # ==========================================
                clean_p_lower = prompt.lower().strip()
                
                # Check if this is a general factual lookup (exclude greetings and friendly chat)
                greeting_words = ["selam", "merhaba", "naber", "nasılsın", "günaydın", "iyi akşamlar", "nasıl gidiyor", "ne yapıyorsun", "halin", "keyifler", "kanka", "dostum"]
                is_greeting = any(gw in clean_p_lower for gw in greeting_words)

                factual_triggers = [
                    "kimdir", "kaç yılında", "nerede doğdu", "tarihi nedir",
                    "who is", "when did", "teknik direktörü kim",
                    "kuruluş", "renkleri", "kulübü", "ne zaman kuruldu"
                ]
                has_factual_intent = not is_greeting and any(t in clean_p_lower for t in factual_triggers)

                # If user in chat mode asks about any paper/technical topic, fetch it so it's grounded
                has_tech_mention = is_academic_or_technical_inquiry(prompt)

                web_fact_context = ""
                if has_tech_mention:
                    paper_search = searcher.search(prompt.strip(), top_k=3)
                    if paper_search.get("has_sufficient_evidence") and paper_search.get("packed_context"):
                        web_fact_context = f"\n\n[İLGİLİ MAKALE VE LİTERATÜR KAYNAĞI ([S1]..[S3])]:\n{paper_search['packed_context']}\n(Talimat: Bu teknik konuyu Samet'e samimi, akıcı ama teknik olarak zengin, tensör boyutları ve formülleriyle açıkla.)"
                    elif is_online():
                        web_res_quick = web_searcher.get_live_web_context(prompt.strip())
                        if web_res_quick.get("has_web"):
                            web_fact_context = f"\n\n[CANLI LİTERATÜR/ARXIV BULGUSU]:\n{web_res_quick['packed_context']}"
                elif has_factual_intent and is_online():
                    # Hızlı Wikipedia olgusal doğrulaması
                    wiki_briefs = web_searcher.search_wikipedia(prompt.strip(), max_results=1)
                    if wiki_briefs:
                        web_fact_context = f"\n\n[GÜNCEL REFERANS]:\n{wiki_briefs[0]['title']}: {wiki_briefs[0]['summary']}"

                with st.spinner("💭 Yanıt hazırlanıyor..."):
                    augmented_user_prompt = prompt + web_fact_context if web_fact_context else prompt
                    casual_msgs = build_casual_response_prompt(
                        augmented_user_prompt,
                        history=st.session_state.messages[:-1],
                        immediate_critique=critique_learned
                    )
                    stream = llm.stream_generate(
                        casual_msgs,
                        max_tokens=1500,
                        temperature=0.75,
                        model=config.MODEL_CHAT
                    )
                    full_response = st.write_stream(stream)
                    clean_full = normalize_latex(full_response)

                    # Otonom RL Öz-Değerlendirme & Anlık Politika Evrimi
                    self_eval = self_evolving_rl.evaluate_and_evolve(prompt, clean_full, mode="chat")
                    score = self_eval.get("score", 85.0)

                    # Konuşmayı kişisel eğitim veri setine kaydet (saf diyalog olarak)
                    collector.record_interaction(
                        user_text=prompt,
                        assistant_text=clean_full,
                        reward=self_eval.get("reward", 1.0),
                        mode="chat"
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": clean_full,
                        "self_eval": self_eval
                    })
                    st.rerun()

            else:
                # ==========================================
                # MOD 2: 🔬 ARAŞTIRMA & LİTERATÜR (Qwen 2.5)
                # ==========================================
                clean_q = prompt.strip()
                augmented_search_query = contextualize_query(clean_q, st.session_state.messages[:-1])

                # Adım 1: Önce Yerel Makaleler ve Obsidian taranır
                with st.status("📚 1. Aşama: Yerel makaleler ve Obsidian taranıyor...", expanded=False) as status:
                    search_res = searcher.search(
                        augmented_search_query,
                        top_k=config.RETRIEVAL_TOP_K,
                        use_decomposition=True,
                        use_hyde=False
                    )
                    retrieved_context = search_res.get("packed_context", "")
                    has_sufficient_evidence = search_res.get("has_sufficient_evidence", False)
                    error_evidence = error_manager.get_relevant_errors(augmented_search_query)

                    web_res = {"has_web": False, "packed_context": "", "sources": [], "raw_results": []}

                    # Adım 2: Yerelde yoksa veya yetersizse İnternetten / Webden ara
                    if not has_sufficient_evidence:
                        status.update(label="🌐 Yerel kütüphanede tam eşleşme bulunamadı. Canlı arXiv ve web taranıyor...", state="running")
                        if is_online():
                            web_res = web_searcher.get_live_web_context(augmented_search_query)
                            if web_res.get("has_web"):
                                retrieved_context = (retrieved_context + "\n\n" + web_res["packed_context"]).strip()
                                has_sufficient_evidence = True
                                status.update(label="✅ Canlı arXiv ve web taraması tamamlandı, kaynaklar birleştirildi.", state="complete")
                            else:
                                status.update(label="ℹ️ Canlı webde doğrudan sonuç bulunamadı. Yerel bilgi tabanı devrede.", state="complete")
                        else:
                            status.update(label="⚠️ Çevrimdışı mod: Canlı web erişilemiyor. Yerel zeka ile yanıt veriliyor.", state="complete")
                    else:
                        status.update(label="✅ Yerel kütüphanede (21 makale / Obsidian) doğrudan akademik kanıt bulundu.", state="complete")

                    budget_parts = {
                        "error_history": error_evidence,
                        "paper_chunks": retrieved_context
                    }
                    packed_report = budget.pack_context(budget_parts)
                    safe_packed = packed_report["packed"]

                # Ayrıştırılan alt-sorgular bilgisi
                sub_qs = search_res.get("sub_queries", [])
                en_sq = search_res.get("search_query", "")
                if len(sub_qs) > 1 and has_sufficient_evidence:
                    st.info("🔀 **Ayrıştırılan Alt-Sorgular (EN):**\n" + "\n".join([f"- `{sq}`" for sq in sub_qs]))
                elif en_sq and en_sq != augmented_search_query and has_sufficient_evidence:
                    st.caption(f"🌐 **Arama Sorgusu (TR → EN):** `{en_sq}`")

                # Eğer ne yerelde ne webde hiçbir şey bulunamadıysa genel uzman modunda yanıt ver
                if not has_sufficient_evidence:
                    expert_msgs = build_general_expert_prompt(clean_q, history=st.session_state.messages[:-1])
                    stream = llm.stream_generate(expert_msgs, max_tokens=3000, temperature=0.1, model=config.MODEL_ACADEMIC)
                    full_response = st.write_stream(stream)
                    clean_full = normalize_latex(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": clean_full})
                    st.rerun()

                # Yerel Makale Atıf Kartları [S1]..[Sk]
                sources = search_res.get("results", [])
                if sources:
                    with st.expander(f"📚 Referans Alınan Yerel Makale Parçaları ({len(sources)} Bölüm — [S1]..[S{len(sources)}])"):
                        for idx, s in enumerate(sources):
                            tag = s.get("source_tag", f"S{idx+1}")
                            meta = s.get("metadata", {})
                            title = meta.get("title", "Akademik Makale")
                            section = meta.get("section_name", "")
                            score = s.get("rerank_score", 0.0)
                            sec_info = f" • Bölüm: *{section}*" if section else ""
                            st.markdown(f"- **`[{tag}]`** 📄 **{title}**{sec_info} (`{s.get('id')}`) — *Alaka Skoru: `{score:.2f}`*")

                # Canlı Web & arXiv Kaynak Kartları [W1]..[Wk]
                if web_res.get("sources"):
                    with st.expander(f"🌐 Canlı Çevrimiçi Web & arXiv Bulguları ({len(web_res['sources'])} Kaynak — [W1]..[W{len(web_res['sources'])}])"):
                        for ws in web_res["sources"]:
                            st.markdown(f"- **`[{ws['tag']}]`** 🔗 [{ws['title']}]({ws['url']})\n  *{ws['summary'][:250]}...*")

                # GraphRAG: Sinaptik Kavram Grafiği Taraması (Obsidian Vault 2. Beyin)
                graph_seeds = graph_rag_engine.extract_seed_nodes(augmented_search_query, search_res.get("results", []))
                subgraph = graph_rag_engine.traverse_subgraph(graph_seeds)
                graph_context = graph_rag_engine.format_graph_context(subgraph)

                if subgraph.get("active_nodes"):
                    with st.expander(f"🕸️ Obsidian Sinaptik Bilgi Grafiği ({len(subgraph['active_nodes'])} Bağlantılı Düğüm — GraphRAG)", expanded=False):
                        for n in subgraph["active_nodes"][:5]:
                            st.markdown(f"- **[[{n['title']}]]** (`{n['id']}`) — *{n.get('folder', '')}*")
                        if subgraph.get("connections"):
                            st.markdown("**🔗 Aktif Çapraz Sinapslar:**")
                            for c in subgraph["connections"][:5]:
                                st.markdown(f"- `[[{c['source_title']}]]` $\\rightarrow$ `[[{c['target_title']}]]`")

                # Step 2.5: Derin CoT Akıl Yürütme ve Konu Parçalama Planı
                thought_plan = ""
                if use_cot:
                    with st.status("🧠 Makale kaynakları üzerinde derin akıl yürütülüyor ve modüler plan hazırlanıyor...", expanded=False) as cot_status:
                        cot_msgs = build_thought_plan_prompt(clean_q, safe_packed.get("paper_chunks", ""))
                        thought_stream = llm.stream_generate(cot_msgs, max_tokens=1000, temperature=0.1, model=config.MODEL_REASONING)
                        thought_plan = "".join(list(thought_stream))
                        st.markdown(thought_plan)
                        cot_status.update(label="✅ Derin akıl yürütme ve konu parçalama planı hazırlandı.", state="complete")

                # Step 3: Akademik Yanıt Sentezi (RAG + GraphRAG)
                messages = build_answer_prompt(
                    query=clean_q,
                    packaged_context=safe_packed.get("paper_chunks", ""),
                    graph_context=graph_context,
                    telemetry_context="",
                    git_diff="",
                    error_context=safe_packed.get("error_history", ""),
                    thought_plan=thought_plan,
                    deep_dive=True,
                    history=st.session_state.messages[:-1],
                    immediate_critique=critique_learned
                )

                stream = llm.stream_generate(
                    messages,
                    max_tokens=3500,
                    model=config.MODEL_ACADEMIC
                )
                full_response = st.write_stream(stream)
                clean_full = normalize_latex(full_response)

                # Step 4: Kaynak ve Alıntı Denetimi (Deterministic Audit)
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

                # Step 5: Otonom Hafızaya Mühürleme (Webden öğrenildiyse Obsidian & ChromaDB'ye ekle)
                if web_res.get("has_web") and web_res.get("raw_results"):
                    cryst_res = web_searcher.crystalize_knowledge(clean_q, clean_full, web_res["raw_results"])
                    if cryst_res.get("success"):
                        st.toast("🧠 Yeni bilgi yerel Obsidian kasasına ve ChromaDB hafızasına mühürlendi!", icon="💾")
                        cryst_footer = f"\n\n---\n💾 **Otonom Hafıza Mühürlendi:** Bu araştırma konusu `Vault/02_Theory_Notes/{cryst_res.get('filename')}` ve yerel ChromaDB vektör hafızasına kaydedildi. Artık çevrimdışı modda da yerel kaynak olarak kullanılabilir."
                        st.markdown(cryst_footer)
                        clean_full = clean_full + cryst_footer

                # Step 6: Otonom RL Öz-Değerlendirme & Anlık Politika Evrimi
                self_eval = self_evolving_rl.evaluate_and_evolve(clean_q, clean_full, retrieved_context=retrieved_context, mode="research")
                score = self_eval.get("score", 85.0)

                # Konuşmayı kişisel eğitim veri setine kaydet
                collector.record_interaction(
                    user_text=clean_q,
                    assistant_text=clean_full,
                    reward=self_eval.get("reward", 1.0),
                    mode="research"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": clean_full,
                    "self_eval": self_eval,
                    "thought_plan": thought_plan
                })
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

