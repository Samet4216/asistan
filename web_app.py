"""
Local Deep Learning Lab Assistant — Minimalist & Elegant Research Studio (Streamlit).
Designed for distraction-free research, neural network engineering, and offline second-brain thinking:
- Ultra-clean dark UI with Inter & Fira Code typography
- Crisp KaTeX mathematical formula rendering ($...$ and $$...$$)
- Real-time streaming generation with local Qwen 2.5 (7B) via Ollama
- Autonomous Hybrid RAG (ChromaDB + BM25) with 57 Landmark Papers
- Offline-first privacy and low-overhead GPU VRAM monitoring
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
    is_academic_or_technical_inquiry,
    contextualize_query
)
from retrieval.search import searcher
from retrieval.web_search import web_searcher, is_online
from retrieval.graph_rag import graph_rag_engine
from answer.prompts import (
    build_answer_prompt,
    build_casual_response_prompt,
    build_general_expert_prompt
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


# ==============================================================================
# SIDEBAR — Minimalist & Purposeful
# ==============================================================================
with st.sidebar:
    # 1. New Chat (Primary Action)
    if st.button("➕ Yeni Sohbet", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.write("")

    # 2. Compact Hardware Status Card
    gpu = get_gpu_status()
    if gpu.get("available"):
        vram_used = gpu['used']
        vram_total = gpu['total']
        vram_pct = int((vram_used / vram_total) * 100)
        
        st.markdown(f"""
        <div style="background:rgba(15,23,42,0.5); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:12px; margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:0.82rem; font-weight:600; color:#f8fafc;">⚡ {gpu['name']}</span>
                <span style="font-size:0.75rem; color:#94a3b8;">{gpu['temp']} °C</span>
            </div>
            <div style="width:100%; height:6px; background:rgba(255,255,255,0.08); border-radius:4px; overflow:hidden; margin:6px 0;">
                <div style="width:{vram_pct}%; height:100%; background:#38bdf8; border-radius:4px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#64748b;">
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

    # 3. Add Knowledge (Clean Collapsible Expander)
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

    # 4. User Profile & Preferences (Clean Collapsible Expander)
    with st.expander("👤 Araştırmacı Profili & Pekiştirilmiş Hafıza", expanded=False):
        current_about = profile_manager.get_about_me()
        metrics = rl_engine.get_metrics()
        st.caption(f"🎯 **Pekiştirilmiş Kural:** {metrics.get('learned_axioms_count', 0)} | ⭐ **Altın Örnek:** {metrics.get('gold_exemplars_count', 0)}")
        st.markdown(f"<div style='font-size:0.82rem; color:#cbd5e1; max-height:140px; overflow-y:auto; margin-bottom:8px;'>{current_about}</div>", unsafe_allow_html=True)
        col_prof1, col_prof2 = st.columns(2)
        with col_prof1:
            if st.button("🔄 Hafızayı Senkronize Et", key="btn_sync_self_train", use_container_width=True, help="Öğrenilen kuralları, DPO eğitim çiftlerini ve Modelfile'ı günceller"):
                s_res = collector.sync_self_training()
                st.toast(f"✅ Model hafızası senkronize edildi! ({s_res.get('sharegpt_dialogs', 0)} diyalog)", icon="🧠")
                st.rerun()
        with col_prof2:
            if st.button("Tarzı Sıfırla", key="btn_reset_style", use_container_width=True):
                rl_engine.add_style_axiom("Cevapları doğrudan konuya girerek, gereksiz nezaket cümleleri olmadan öz ve net tut.")
                st.toast("Tarz politikası sıfırlandı.", icon="🔄")
                st.rerun()

    # 5. Obsidian Drafts (Only shown if pending drafts exist)
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
# MAIN WORKSPACE — Sleek Header & 2 Simple Views
# ==============================================================================

# Load Corpus Catalog
catalog_path = PROJECT_ROOT / "data" / "corpus_catalog.json"
papers_info = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else []

# Top Minimalist Header Bar
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; padding:6px 0 16px 0; border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:18px;">
    <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-size:1.5rem;">🔬</span>
        <div>
            <div style="font-weight:600; font-size:1.15rem; color:#f8fafc; letter-spacing:-0.02em;">Yapay Zeka Laboratuvar Asistanı</div>
            <div style="font-size:0.78rem; color:#64748b;">Qwen 2.5 (7B) • RTX 4050 • Yerel Hibrit Hafıza ({len(papers_info)} Makale)</div>
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:8px;">
        <span style="display:inline-flex; align-items:center; gap:6px; font-size:0.75rem; color:#10b981; background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.2); padding:4px 10px; border-radius:999px;">
            <span style="width:6px; height:6px; background:#10b981; border-radius:50%;"></span> Çevrimdışı & Yerel
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# 2 Simple Tabs: Asistan Chat & Makale Kütüphanesi
TAB_CHAT = "💬 Sohbet & Asistan"
TAB_LIB = f"📚 Makale Kütüphanesi ({len(papers_info)})"

if "switch_to_tab" in st.session_state and st.session_state.switch_to_tab:
    st.session_state.active_main_tab = st.session_state.switch_to_tab
    st.session_state.switch_to_tab = None

if "active_main_tab" not in st.session_state:
    st.session_state.active_main_tab = TAB_CHAT

tab_chat, tab_library = st.tabs([TAB_CHAT, TAB_LIB], key="active_main_tab", on_change="rerun")

# ------------------------------------------------------------------------------
# TAB 1: SOHBET & ASİSTAN
# ------------------------------------------------------------------------------
with tab_chat:
    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat export & clear actions (minimal row at top of chat)
    if st.session_state.messages:
        col_hdr1, col_hdr2 = st.columns([6, 1])
        with col_hdr2:
            export_md = "# Deep Learning Lab Assistant — Sohbet Dökümü\n\n"
            for m in st.session_state.messages:
                role_label = "👤 **Kullanıcı:**" if m["role"] == "user" else "🤖 **Asistan:**"
                export_md += f"### {role_label}\n\n{m['content']}\n\n---\n\n"
            st.download_button(
                "📥 Dökümü İndir",
                data=export_md,
                file_name=f"asistan_sohbet_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                key="btn_export_chat",
                use_container_width=True
            )

    # Empty State: Clean Welcome & Minimalist Prompt Chips
    if not st.session_state.messages:
        st.markdown(f"""
        <div style="text-align:center; padding:36px 16px 24px 16px;">
            <h2 style="font-weight:600; font-size:1.45rem; color:#f8fafc; margin-bottom:8px;">Nasıl yardımcı olabilirim?</h2>
            <p style="font-size:0.9rem; color:#94a3b8; max-width:560px; margin:0 auto 24px auto;">
                Sinir ağı mimarileri, matematiksel formüller, hiperparametreler veya yerel kütüphanedeki {len(papers_info)} makale hakkında sorular sorabilirsiniz.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            if st.button("⚡ LoRA ile İnce Ayar Mantığı", use_container_width=True):
                st.session_state.pending_prompt = "LoRA düşük dereceli adaptasyon matrislerini ($W = W_0 + B \\cdot A$) nasıl kurgular ve VRAM tasarrufunu matematiksel olarak nasıl sağlar?"
                st.session_state.switch_to_tab = TAB_CHAT
                st.rerun()
        with col_q2:
            if st.button("🔍 Grad-CAM ile Karar Haritalama", use_container_width=True):
                st.session_state.pending_prompt = "Grad-CAM konvolüsyon katmanlarındaki gradyanları kullanarak modelin odaklandığı bölgeleri nasıl haritalandırır?"
                st.session_state.switch_to_tab = TAB_CHAT
                st.rerun()
        with col_q3:
            if st.button("📐 Kaiming / He İlklendirme", use_container_width=True):
                st.session_state.pending_prompt = "Kaiming He ilklendirmesi ReLU aktivasyonunda varyansı ($Var(w) = 2/n$) nasıl korur ve vanishing gradient problemini nasıl çözer?"
                st.session_state.switch_to_tab = TAB_CHAT
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
                        collector.sync_self_training()
                        st.toast("🎯 Yanıt altın örnek olarak mühürlendi ve model güncellendi! (+1.0)", icon="👍")
                        st.rerun()
                with act_col2:
                    if st.button("👎", key=f"btn_down_{idx}", help="Bu cevabı beğenmedim / hatalı buldum"):
                        rl_engine.record_feedback(user_query_for_msg, content, reward=-1.0, style_tag="negative")
                        collector.sync_self_training()
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
                                res = teacher_engine.record_correction(
                                    query=user_query_for_msg,
                                    wrong_response=content,
                                    ground_truth=teacher_ans.strip(),
                                    critique=teacher_critique.strip()
                                )
                                if res.get("success"):
                                    collector.sync_self_training()
                                    st.session_state.messages[idx]["content"] = (
                                        content + f"\n\n---\n✅ **[Doğrulanmış Öğretmen Yanıtı (Ground Truth)]:**\n\n{teacher_ans.strip()}"
                                    )
                                    st.toast("🎯 Doğru cevap hafızaya mühürlendi ve modele öğretildi!", icon="🎓")
                                    st.rerun()
                                else:
                                    st.error(res.get("error", "Kayıt başarısız."))
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
    user_input = st.chat_input("Bir soru sorun, mimari tartışın veya bir konu danışın...")
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
            is_technical = is_academic_or_technical_inquiry(clean_q)

            retrieved_sources = []
            if is_technical:
                augmented_query = contextualize_query(clean_q, st.session_state.messages[:-1])
                
                with st.spinner("💭 Literatür taranıyor ve yanıt hazırlanıyor..."):
                    search_res = searcher.search(
                        augmented_query,
                        top_k=config.RETRIEVAL_TOP_K,
                        use_decomposition=True,
                        use_hyde=False
                    )
                    retrieved_context = search_res.get("packed_context", "")
                    has_sufficient = search_res.get("has_sufficient_evidence", False)
                    retrieved_sources = search_res.get("results", [])

                    # Fallback to web/arXiv if insufficient
                    if not has_sufficient and is_online():
                        web_res = web_searcher.get_live_web_context(augmented_query)
                        if web_res.get("has_web"):
                            retrieved_context = (retrieved_context + "\n\n" + web_res["packed_context"]).strip()
                            has_sufficient = True

                    if not has_sufficient:
                        # DÜRÜST BİLMİYORUM PROTOKOLÜ (ZERO HALLUCINATION & STRICT ABSTENTION)
                        abstain_response = (
                            "⚠️ **Doğrulanmış Kanıt Bulunamadı / Bilmiyorum:**\n\n"
                            "Bu soru hakkında yerel makale arşivimizde ve canlı akademik web taramasında doğrulanmış bir kanıt bulunamadı. "
                            "Yanlış, uydurma veya yanıltıcı bilgi üretmemek adına spekülasyon yapmıyorum; bu konunun gerçek cevabını **bilmiyorum**.\n\n"
                            "💡 *Eğer bu bilginin doğrusunu biliyorsanız, aşağıdaki **'✏️ Doğrusunu Öğret'** butonuna tıklayarak doğru cevabı sisteme mühürleyebilir ve modeli eğitebilirsiniz.*"
                        )
                        st.markdown(abstain_response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": abstain_response
                        })
                        st.rerun()

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

                    stream = llm.stream_generate(messages, max_tokens=3500, temperature=0.1, model=config.MODEL_ACADEMIC)
                    full_response = st.write_stream(stream)
                    clean_full = normalize_latex(full_response)

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


# ------------------------------------------------------------------------------
# TAB 2: MAKALE KÜTÜPHANESİ
# ------------------------------------------------------------------------------
with tab_library:
    st.markdown(f"### 📚 İndekslenmiş Makale Kütüphanesi ({len(papers_info)} Makale)")
    st.caption("Yerel ChromaDB vektör tabanına ve hibrit aramaya (BM25 + BGE Embeddings) tam metin ve formülleriyle eklenmiş öncü makaleler.")

    search_filter = st.text_input("🔍 Kütüphanede Ara:", placeholder="Örn: LoRA, Transformer, YOLO, Reinforcement...")

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
                <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.06); border-left:3px solid {domain_color}; border-radius:10px; padding:14px; margin-bottom:10px; height:180px; display:flex; flex-direction:column; justify-content:space-between;">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                            <span style="font-size:10px; font-weight:600; text-transform:uppercase; color:{domain_color}; letter-spacing:0.5px;">{p.get('domain', 'AI')}</span>
                            <span style="font-size:11px; color:#64748b; font-family:'Fira Code',monospace;">arXiv:{p.get('id', '')}</span>
                        </div>
                        <div style="font-size:0.95rem; font-weight:600; color:#f8fafc; margin-bottom:4px; line-height:1.3; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;">
                            {p.get('title', '')}
                        </div>
                        <div style="font-size:0.8rem; color:#94a3b8; line-height:1.4; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;">
                            {p.get('desc', '')}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("⚡ Bu Makaleyi Sor", key=f"btn_ask_lib_{p.get('id')}", use_container_width=True):
                    paper_prompt = p.get("prompt", f"{p.get('title')} makalesinin mimari yapısını, temel formüllerini ve avantajlarını açıkla.")
                    st.session_state.pending_prompt = paper_prompt
                    st.session_state.switch_to_tab = TAB_CHAT
                    st.toast(f"'{p.get('title')[:32]}...' analizi başlatılıyor, sohbete yönlendiriliyorsunuz.", icon="🚀")
                    st.rerun()
