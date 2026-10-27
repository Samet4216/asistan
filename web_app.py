"""
Local Deep Learning Lab Assistant — Minimalist & Elegant Research Studio (Streamlit).
Designed for distraction-free research, neural network engineering, and offline second-brain thinking:
- Ultra-clean dark UI with Inter & Fira Code typography
- Centered 840px reading container (Claude / ChatGPT layout)
- Crisp KaTeX mathematical formula rendering ($...$ and $$...$$)
- Real-time streaming generation with local Qwen 2.5 (7B) via Ollama
- Autonomous Hybrid RAG (ChromaDB + BM25) with 57 Landmark Papers
"""

import sys
import re
import json
import importlib
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.resolve() if Path(__file__).parent.name == "scripts" else Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components
import config
import llm
import budget
import utils
importlib.reload(utils)
from utils import (
    get_gpu_status,
    normalize_latex,
    is_academic_or_technical_inquiry,
    contextualize_query,
    highlight_fosforlu,
    inject_citation_anchors
)

from retrieval.search import searcher
from retrieval.web_search import web_searcher, is_online
from retrieval.graph_rag import graph_rag_engine
import answer.prompts as prompts_mod
import memory.user_profile as up_mod
import memory.teacher_correction as tc_mod
import memory.conversation_collector as cc_mod
import memory.rl_feedback as rl_mod

importlib.reload(prompts_mod)
importlib.reload(up_mod)
importlib.reload(tc_mod)
importlib.reload(cc_mod)
importlib.reload(rl_mod)

from answer.prompts import (
    build_answer_prompt,
    build_casual_response_prompt
)
from memory.approval import approval_manager
from memory.user_profile import profile_manager
from memory.auto_memory import auto_memory
from memory.rl_feedback import rl_engine
from memory.self_evolving_rl import self_evolving_rl
from memory.conversation_collector import collector
from memory.teacher_correction import teacher_engine
from ingest.single_ingest import ingest_arxiv_paper
from ingest.local_file_ingest import ingest_uploaded_file


def safe_sync_self_training():
    """Safely executes self-training sync even under Streamlit hot-reload cache."""
    try:
        if hasattr(collector, "sync_self_training"):
            return collector.sync_self_training()
        else:
            import memory.conversation_collector as _cc
            importlib.reload(_cc)
            if hasattr(_cc.collector, "sync_self_training"):
                return _cc.collector.sync_self_training()
    except Exception as e:
        print(f"[!] Warning during self-training sync: {e}")
    return {}


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
    st.markdown(f"<style>\n{raw_css}\n</style>", unsafe_allow_html=True)

# Load modular quotation popover script
js_file = PROJECT_ROOT / "assets" / "quote_popover.js"
if js_file.exists():
    components.html(f"<script>{js_file.read_text(encoding='utf-8')}</script>", height=0)

# Load Corpus Catalog
catalog_path = PROJECT_ROOT / "data" / "corpus_catalog.json"
papers_info = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else []


# ==============================================================================
# SIDEBAR — Minimalist & Functional
# ==============================================================================
with st.sidebar:
    # 1. New Chat (Primary Action)
    if st.button("➕ Yeni Sohbet", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.session_state.current_view = "💬 Sohbet"
        st.rerun()

    # 2. View Switcher (Sohbet vs Kütüphane)
    if "current_view" not in st.session_state:
        st.session_state.current_view = "💬 Sohbet"

    selected_view = st.radio(
        "Görünüm Seç:",
        ["💬 Sohbet", f"📚 Kütüphane ({len(papers_info)})"],
        index=0 if st.session_state.current_view == "💬 Sohbet" else 1,
        label_visibility="collapsed",
        horizontal=True
    )
    if selected_view != st.session_state.current_view:
        st.session_state.current_view = selected_view
        st.rerun()

    st.write("")

    # 3. Compact Hardware Status Card
    gpu = get_gpu_status()
    if gpu.get("available"):
        vram_used = gpu['used']
        vram_total = gpu['total']
        vram_pct = int((vram_used / vram_total) * 100)
        
        st.markdown(f"""
        <div style="background:rgba(18,24,38,0.5); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:10px 12px; margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span style="font-size:0.8rem; font-weight:600; color:#f1f5f9;">⚡ {gpu['name']}</span>
                <span style="font-size:0.75rem; color:#94a3b8;">{gpu['temp']} °C</span>
            </div>
            <div style="width:100%; height:4px; background:rgba(255,255,255,0.06); border-radius:4px; overflow:hidden; margin:6px 0;">
                <div style="width:{vram_pct}%; height:100%; background:#38bdf8; border-radius:4px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#64748b;">
                <span>VRAM: {vram_used} / {vram_total} MB</span>
                <span>%{vram_pct}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("💻 Sistem: CPU Modunda")

    # VRAM Cleanup button (compact)
    if st.button("❄️ VRAM'i Boşalt", use_container_width=True, help="Model ağırlıklarını VRAM'den siler, GPU'yu soğutur."):
        llm.unload()
        st.toast("VRAM serbest bırakıldı.", icon="❄️")
        st.rerun()

    st.divider()

    # 4. Add Knowledge (Clean Collapsible Expander)
    with st.expander("📥 Hafızaya Kaynak Ekle", expanded=False):
        tab_up_file, tab_up_arxiv = st.tabs(["Dosya Yükle", "arXiv"])
        
        with tab_up_file:
            uploaded_file = st.file_uploader(
                "PDF, Markdown veya TXT:",
                type=["pdf", "md", "txt"],
                label_visibility="collapsed"
            )
            custom_title = st.text_input("Başlık (Opsiyonel):", placeholder="Örn: Tez Notları", key="upload_title")
            if st.button("İndeksle & Ekle", key="btn_ingest_file", use_container_width=True):
                if uploaded_file is not None:
                    with st.spinner("Dosya işleniyor..."):
                        res = ingest_uploaded_file(uploaded_file.getvalue(), uploaded_file.name, custom_title=custom_title)
                        if res.get("success"):
                            st.success(f"'{res.get('title')}' eklendi ({res.get('chunks_count')} parça).")
                            st.rerun()
                        else:
                            st.error(f"Hata: {res.get('error')}")
                else:
                    st.warning("Lütfen bir dosya seçin.")

        with tab_up_arxiv:
            arxiv_id_input = st.text_input("arXiv Kodu:", placeholder="Örn: 2106.09685", key="arxiv_id_input")
            if st.button("İndir & İndeksle", key="btn_ingest_arxiv", use_container_width=True):
                if arxiv_id_input.strip():
                    with st.spinner(f"[{arxiv_id_input.strip()}] indiriliyor..."):
                        res = ingest_arxiv_paper(arxiv_id_input.strip())
                        if res.get("success"):
                            st.success(f"'{res.get('title')}' eklendi ({res.get('chunks_count')} parça).")
                            st.rerun()
                        else:
                            st.error(f"Hata: {res.get('error')}")

    # 5. User Profile & Reinforcement Memory (Clean Collapsible Expander)
    with st.expander("👤 Araştırmacı Profili & Hafıza", expanded=False):
        current_about = profile_manager.get_about_me()
        metrics = rl_engine.get_metrics()
        st.caption(f"🎯 **Pekiştirilmiş Kural:** {metrics.get('learned_axioms_count', 0)} | ⭐ **Altın Örnek:** {metrics.get('gold_exemplars_count', 0)}")
        st.markdown(f"<div style='font-size:0.8rem; color:#cbd5e1; max-height:130px; overflow-y:auto; margin-bottom:8px;'>{current_about}</div>", unsafe_allow_html=True)
        col_prof1, col_prof2 = st.columns(2)
        with col_prof1:
            if st.button("🔄 Senkronize Et", key="btn_sync_self_train", use_container_width=True, help="Öğrenilen kuralları, DPO eğitim çiftlerini ve Modelfile'ı günceller"):
                s_res = safe_sync_self_training()
                st.toast(f"✅ Hafıza senkronize edildi! ({s_res.get('sharegpt_dialogs', 0)} diyalog)", icon="🧠")
                st.rerun()
        with col_prof2:
            if st.button("Tarzı Sıfırla", key="btn_reset_style", use_container_width=True):
                rl_engine.add_style_axiom("Cevapları doğrudan konuya girerek, gereksiz nezaket cümleleri olmadan öz ve net tut.")
                st.toast("Tarz politikası sıfırlandı.", icon="🔄")
                st.rerun()

    # 6. Obsidian Drafts (Only shown if pending drafts exist)
    drafts = approval_manager.list_drafts()
    if drafts:
        with st.expander(f"📝 Bekleyen Taslaklar ({len(drafts)})", expanded=False):
            for d in drafts[:3]:
                st.caption(f"**{d['title'][:25]}**")
                if st.button("Arşive Ekle", key=f"draft_{d['id']}", use_container_width=True):
                    approval_manager.approve_draft(d["id"])
                    st.toast("Obsidian Vault'a eklendi!", icon="✅")
                    st.rerun()


# ==============================================================================
# VIEW 1: MAKALE KÜTÜPHANESİ
# ==============================================================================
if st.session_state.current_view.startswith("📚"):
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; padding:4px 0 16px 0; border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:18px;">
        <div>
            <div style="font-weight:600; font-size:1.2rem; color:#f8fafc;">📚 İndekslenmiş Makale Kütüphanesi ({len(papers_info)} Makale)</div>
            <div style="font-size:0.8rem; color:#94a3b8;">Yerel ChromaDB vektör tabanına ve hibrit aramaya (BM25 + BGE Embeddings) tam metin ve formülleriyle eklenmiş öncü makaleler.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    search_filter = st.text_input("🔍 Kütüphanede Filtrele:", placeholder="Örn: LoRA, Transformer, YOLO, Attention...", label_visibility="collapsed")

    filtered_papers = papers_info
    if search_filter.strip():
        q_term = search_filter.strip().lower()
        filtered_papers = [
            p for p in papers_info
            if q_term in p.get("title", "").lower() or q_term in p.get("domain", "").lower() or q_term in p.get("desc", "").lower() or q_term in p.get("id", "").lower()
        ]

    st.caption(f"Gösterilen: {len(filtered_papers)} makale")

    cols_per_row = 2
    rows = [filtered_papers[i:i + cols_per_row] for i in range(0, len(filtered_papers), cols_per_row)]
    for row in rows:
        grid_cols = st.columns(cols_per_row)
        for col_idx, p in enumerate(row):
            with grid_cols[col_idx]:
                domain_color = p.get('color', '#38bdf8')
                st.markdown(f"""
                <div style="background:rgba(18,24,38,0.5); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:14px; margin-bottom:10px; min-height:160px; display:flex; flex-direction:column; justify-content:space-between;">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-size:10px; font-weight:600; text-transform:uppercase; color:{domain_color}; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.07); padding:2px 8px; border-radius:999px; letter-spacing:0.4px;">{p.get('domain', 'AI')}</span>
                            <span style="font-size:11px; color:#64748b; font-family:'Fira Code',monospace;">arXiv:{p.get('id', '')}</span>
                        </div>
                        <div style="font-size:0.92rem; font-weight:600; color:#f1f5f9; margin-bottom:5px; line-height:1.35;">
                            {p.get('title', '')}
                        </div>
                        <div style="font-size:0.78rem; color:#94a3b8; line-height:1.45;">
                            {p.get('desc', '')}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("⚡ Bu Makaleyi Sor", key=f"btn_ask_lib_{p.get('id')}", use_container_width=True):
                    st.session_state.pending_prompt = p.get("prompt", f"{p.get('title')} makalesinin mimari yapısını, temel formüllerini ve avantajlarını açıkla.")
                    st.session_state.current_view = "💬 Sohbet"
                    st.rerun()


# ==============================================================================
# VIEW 2: SOHBET & ASİSTAN (Pure Centered Chat)
# ==============================================================================
else:
    # Top Minimalist Header
    col_hdr_l, col_hdr_r = st.columns([6, 1.2])
    with col_hdr_l:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; padding:4px 0 12px 0;">
            <span style="font-size:1.25rem;">🔬</span>
            <span style="font-weight:600; font-size:1.05rem; color:#f8fafc;">Yapay Zeka Laboratuvarı</span>
            <span style="display:inline-flex; align-items:center; gap:5px; font-size:0.72rem; color:#10b981; background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); padding:2px 8px; border-radius:999px; margin-left:6px;">
                <span style="width:5px; height:5px; background:#10b981; border-radius:50%;"></span> Çevrimdışı & Yerel
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_hdr_r:
        if "messages" in st.session_state and st.session_state.messages:
            export_md = "# Deep Learning Lab Assistant — Sohbet Dökümü\n\n"
            for m in st.session_state.messages:
                role_label = "👤 **Kullanıcı:**" if m["role"] == "user" else "🤖 **Asistan:**"
                export_md += f"### {role_label}\n\n{m['content']}\n\n---\n\n"
            st.download_button(
                "📥 Döküm",
                data=export_md,
                file_name=f"asistan_sohbet_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                key="btn_export_chat",
                use_container_width=True
            )

    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Empty State: Minimalist Welcome Greeting
    if not st.session_state.messages:
        st.markdown(f"""
        <div style="text-align:center; padding:48px 16px 28px 16px;">
            <div style="font-size:2.4rem; margin-bottom:10px;">🔬</div>
            <h2 style="font-weight:600; font-size:1.4rem; color:#f8fafc; margin-bottom:6px;">Size nasıl yardımcı olabilirim?</h2>
            <p style="font-size:0.88rem; color:#94a3b8; max-width:520px; margin:0 auto 24px auto;">
                Sinir ağı mimarileri, optimizasyon problemleri veya yerel kütüphanedeki {len(papers_info)} makale hakkında dilediğinizi sorabilirsiniz.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            if st.button("⚡ LoRA ile İnce Ayar", use_container_width=True):
                st.session_state.pending_prompt = "LoRA düşük dereceli adaptasyon matrislerini ($W = W_0 + B \cdot A$) nasıl kurgular ve VRAM tasarrufunu matematiksel olarak nasıl sağlar?"
                st.rerun()
        with col_q2:
            if st.button("🔍 Grad-CAM Haritalama", use_container_width=True):
                st.session_state.pending_prompt = "Grad-CAM konvolüsyon katmanlarındaki gradyanları kullanarak modelin odaklandığı bölgeleri nasıl haritalandırır?"
                st.rerun()
        with col_q3:
            if st.button("📐 Kaiming / He İlklendirme", use_container_width=True):
                st.session_state.pending_prompt = "Kaiming He ilklendirmesi ReLU aktivasyonunda varyansı ($Var(w) = 2/n$) nasıl korur ve vanishing gradient problemini nasıl çözer?"
                st.rerun()
        st.write("")

    # Render Chat History
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            content = msg["content"]
            # Clean LaTeX normalization and render
            st.markdown(normalize_latex(content), unsafe_allow_html=True)

            # If sources were recorded for this assistant message, show a clean collapsed expander
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📚 Kaynakça ({len(msg['sources'])} Referans)", expanded=False):
                    for s in msg["sources"]:
                        meta = s.get("metadata", {})
                        title = meta.get("title", "Makale")
                        tag = s.get("source_tag", "Kaynak")
                        score = s.get("rerank_score", 0.0)
                        st.markdown(f"- **`[{tag}]`** 📄 {title} *(Skor: {score:.2f})*")

            # Sleek action row under assistant messages
            if msg["role"] == "assistant":
                user_query_for_msg = ""
                for prev in reversed(st.session_state.messages[:idx]):
                    if prev["role"] == "user":
                        user_query_for_msg = prev["content"]
                        break

                act_col1, act_col2, act_col3, act_col4, _ = st.columns([0.45, 0.45, 1.4, 0.8, 4.5])
                with act_col1:
                    if st.button("👍", key=f"btn_up_{idx}", help="Bu tarzı ve yaklaşımı pekiştir"):
                        rl_engine.record_feedback(user_query_for_msg, content, reward=1.0, style_tag="positive")
                        safe_sync_self_training()
                        st.toast("🎯 Yanıt altın örnek olarak mühürlendi ve model güncellendi! (+1.0)", icon="👍")
                        st.rerun()
                with act_col2:
                    if st.button("👎", key=f"btn_down_{idx}", help="Bu cevabı beğenmedim / hatalı buldum"):
                        rl_engine.record_feedback(user_query_for_msg, content, reward=-1.0, style_tag="negative")
                        safe_sync_self_training()
                        st.toast("Geri bildirim kaydedildi (-1.0). İstersen '✏️ Doğrusunu Öğret' ile doğrusunu mühürleyebilirsin.", icon="👎")
                        st.rerun()
                with act_col3:
                    with st.popover("✏️ Doğrusunu Öğret", help="Cevap yanlışsa doğrusunu yazın; model bunu kalıcı hafızaya alsın ve bir daha yanlış yapmasın"):
                        st.markdown("**🎓 Öğretmen Düzeltmesi (Ground Truth)**")
                        st.caption("Verdiğiniz doğru cevap Obsidian'a ve ChromaDB vektör tabanına 'Doğrulanmış Bilgi' olarak kaydedilir; DPO eğitim çifti oluşturulur.")
                        teacher_ans = st.text_area("Doğru cevap nedir?", placeholder="Örn: Bu modelde X kullanılır, formülü şudur...", key=f"teacher_input_{idx}", height=120)
                        teacher_critique = st.text_input("Neyi yanlış bildi? (Opsiyonel)", placeholder="Örn: Formülü uydurmuş...", key=f"teacher_crit_{idx}")
                        if st.button("💾 Doğru Cevabı Mühürle & Eğit", key=f"btn_seal_truth_{idx}", use_container_width=True):
                            if teacher_ans.strip():
                                try:
                                    res = teacher_engine.record_correction(
                                        query=user_query_for_msg,
                                        wrong_response=content,
                                        ground_truth=teacher_ans.strip(),
                                        critique=teacher_critique.strip()
                                    )
                                    if res.get("success"):
                                        safe_sync_self_training()
                                        st.session_state.messages[idx]["content"] = (
                                            content + f"\n\n---\n✅ **[Doğrulanmış Öğretmen Yanıtı (Ground Truth)]:**\n\n{teacher_ans.strip()}"
                                        )
                                        st.toast("🎯 Doğru cevap hafızaya mühürlendi ve modele öğretildi!", icon="🎓")
                                        st.rerun()
                                    else:
                                        st.error(res.get("error", "Kayıt başarısız."))
                                except Exception as e_teach:
                                    st.error(f"Kayıt sırasında hata: {e_teach}")
                            else:
                                st.warning("Lütfen doğru cevabı girin.")
                with act_col4:
                    if st.button("💾 Kaydet", key=f"btn_save_{idx}", help="Obsidian Vault'a not olarak kaydet"):
                        safe_title = re.sub(r'[^A-Za-z0-9]+', '_', user_query_for_msg[:30]).strip('_') or "Not"
                        approval_manager.create_draft(
                            title=safe_title,
                            content=content,
                            category="Theory_Notes",
                            tags=["qa-chat", "lab-notes"]
                        )
                        st.toast(f"'{safe_title}' Obsidian taslaklarına kaydedildi!", icon="📝")

    # Chat Input Handling
    user_input = st.chat_input("Bir soru sorun veya bir konu danışın...")
    prompt_to_process = None
    if user_input:
        prompt_to_process = user_input
    elif "pending_prompt" in st.session_state and st.session_state.pending_prompt:
        prompt_to_process = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt_to_process:
        # Check conversational critique
        if len(st.session_state.messages) > 1:
            last_asst, last_usr = None, None
            for prev_m in reversed(st.session_state.messages):
                if prev_m["role"] == "assistant" and not last_asst:
                    last_asst = prev_m["content"]
                elif prev_m["role"] == "user" and not last_usr:
                    last_usr = prev_m["content"]
            critique_res = rl_engine.detect_conversational_critique(prompt_to_process, last_usr, last_asst)
            if critique_res and critique_res.get("learned_axiom"):
                st.toast(f"Tarzınız algılandı: {critique_res['learned_axiom']}", icon="🎯")

        # Auto-memory cues
        if auto_memory.has_personal_cues(prompt_to_process):
            try:
                auto_memory.extract_and_update(prompt_to_process)
            except Exception:
                pass

        # Append and render user message
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            clean_q = prompt_to_process.strip()
            augmented_query = contextualize_query(clean_q, st.session_state.messages[:-1])
            is_technical = is_academic_or_technical_inquiry(clean_q) or is_academic_or_technical_inquiry(augmented_query)

            # Context Continuity: If the previous discussion was technical/academic and user continues, stay in academic mode
            if not is_technical and len(st.session_state.messages) > 1:
                last_asst_msg = next((m for m in reversed(st.session_state.messages[:-1]) if m["role"] == "assistant"), None)
                if last_asst_msg:
                    asst_content = last_asst_msg.get("content", "")
                    if any(marker in asst_content for marker in ["$$", "Tensör", "Tensor", "Bölüm", "Parça", "[S", "ReLU", "Layer", "Gradient", "Loss", "Initialization", "Neuron"]):
                        if not utils.is_casual_or_personal(clean_q):
                            is_technical = True

            retrieved_sources = []
            if is_technical:
                with st.spinner("💭 Yanıt hazırlanıyor..."):
                    search_res = searcher.search(
                        augmented_query,
                        top_k=config.RETRIEVAL_TOP_K,
                        use_decomposition=True,
                        use_hyde=False
                    )
                    retrieved_context = search_res.get("packed_context", "")
                    has_sufficient = search_res.get("has_sufficient_evidence", False)
                    retrieved_sources = search_res.get("results", [])

                    # Fallback to web/arXiv/Wikipedia if insufficient
                    if not has_sufficient and is_online():
                        web_res = web_searcher.get_live_web_context(augmented_query)
                        if web_res.get("has_web"):
                            retrieved_context = (retrieved_context + "\n\n" + web_res["packed_context"]).strip()
                            has_sufficient = True

                    if has_sufficient:
                        # GraphRAG Context
                        graph_seeds = graph_rag_engine.extract_seed_nodes(augmented_query, search_res.get("results", []))
                        subgraph = graph_rag_engine.traverse_subgraph(graph_seeds)
                        graph_context = graph_rag_engine.format_graph_context(subgraph)

                        budget_parts = {"paper_chunks": retrieved_context}
                        packed_report = budget.pack_context(budget_parts)
                        safe_packed = packed_report["packed"]

                        messages = build_answer_prompt(
                            query=clean_q,
                            packaged_context=safe_packed.get("paper_chunks", ""),
                            graph_context=graph_context,
                            telemetry_context="",
                            git_diff="",
                            error_context="",
                            thought_plan="",
                            deep_dive=True,
                            history=st.session_state.messages[:-1]
                        )
                        stream = llm.stream_generate(messages, max_tokens=3500, temperature=0.0, model=config.MODEL_ACADEMIC)
                        full_response = st.write_stream(stream)
                        clean_full = normalize_latex(full_response)
                    else:
                        # SIFIR HALÜSİNASYON / KESİN DÜRÜSTLÜK (ZERO-HALLUCINATION STRICT ABSTENTION)
                        # Literatürde veya yerel makalelerde doğrulanmış kanıt yoksa ASLA spekülasyon veya uydurma bilgi üretilmez.
                        clean_full = (
                            "⚠️ **Bu konu hakkında yerel makale arşivimizde ve taranan literatürde doğrulanmış bir akademik kanıt veya bilgi bulunmamaktadır; bu yüzden bilmiyorum.**\n\n"
                            "Halüsinasyon, uydurma veya yanıltıcı bilgi üretmemek adına varsayımsal tahminde bulunmuyorum. "
                            "Eğer bu konuyla ilgili doğru bilgiye sahipseniz, aşağıdaki **'✏️ Doğrusunu Öğret'** butonuna tıklayarak doğrudan gerçeği sisteme öğretebilirsiniz."
                        )
                        st.markdown(clean_full)

                    # Silent interaction recording
                    self_eval = self_evolving_rl.evaluate_and_evolve(clean_q, clean_full, mode="research")
                    collector.record_interaction(clean_q, clean_full, reward=self_eval.get("reward", 1.0), mode="research")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": clean_full,
                        "sources": retrieved_sources
                    })
                    st.rerun()

            else:
                # Casual / General Chat
                with st.spinner("💭 Yanıt hazırlanıyor..."):
                    # Check factual lookup if needed
                    web_fact_context = ""
                    clean_p_lower = clean_q.lower()
                    factual_triggers = ["kimdir", "kaç yılında", "tarihi nedir", "who is", "when did"]
                    if any(t in clean_p_lower for t in factual_triggers) and is_online():
                        wiki_briefs = web_searcher.search_wikipedia(clean_q, max_results=1)
                        if wiki_briefs:
                            web_fact_context = f"\n\n[GÜNCEL REFERANS]:\n{wiki_briefs[0]['title']}: {wiki_briefs[0]['summary']}"

                    augmented_user_prompt = clean_q + web_fact_context if web_fact_context else clean_q
                    casual_msgs = build_casual_response_prompt(augmented_user_prompt, history=st.session_state.messages[:-1])
                    stream = llm.stream_generate(casual_msgs, max_tokens=1500, temperature=0.7, model=config.MODEL_CHAT)
                    full_response = st.write_stream(stream)
                    clean_full = normalize_latex(full_response)

                    # Silent interaction recording
                    self_eval = self_evolving_rl.evaluate_and_evolve(clean_q, clean_full, mode="chat")
                    collector.record_interaction(clean_q, clean_full, reward=self_eval.get("reward", 1.0), mode="chat")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": clean_full
                    })
                    st.rerun()
