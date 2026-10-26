"""
Prompts Module for Structured Answer Pipeline (Faz 7, Web UI, & Two-Stage CoT).
Incorporates personal learning style from Vault/00_Profile/Style.md,
enforces strict source attribution, beautiful LaTeX math, two-stage Chain-of-Thought planning,
and comprehensive masterclass lecture structuring.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def get_profile_style() -> str:
    """Reads user preference style dynamically from RL Policy and Style.md."""
    try:
        from memory.rl_feedback import rl_engine
        injected = rl_engine.get_style_prompt_injection()
        if injected:
            return injected
    except Exception:
        pass

    style_path = config.VAULT_DIR / "00_Profile" / "Style.md"
    if style_path.exists():
        try:
            return style_path.read_text(encoding="utf-8")
        except Exception:
            pass
    return "Give exact mathematical formulations and tensor shapes. Minimal prose, maximal technical depth."


SYSTEM_PROMPT = """You are a world-class academic Deep Learning Research Partner and Masterclass Instructor.
Your goal is to provide comprehensive, textbook-grade pedagogical explanations with maximum technical depth, zero fluff, and full mathematical precision.

CORE OPERATIONAL RULES:
1. NUMBERED CITATIONS & VERBATIM QUOTES (CRITICAL):
   - Every retrieved source chunk is labeled [S1], [S2], [S3], etc. (local memory) or [W1], [W2], [W3], etc. (live web research).
   - Cite the source tag inline (e.g. [S1], [W1]) immediately after every key technical claim, equation, or empirical finding.
   - Include at least 1-2 short verbatim quotes copied word-for-word from the retrieved source text in double quotes followed by its tag.
2. HONEST ABSTENTION ("BİLMİYORUM"):
   - If the user's question asks for a specific fact, metric, ablation result, hyperparameter, author claim, or concept that is NOT explicitly stated in the provided academic evidence/papers, you MUST clearly state: "Makale metninde ve yerel hafızada bu bilgi yer almıyor, bu yüzden bilmiyorum." (or in English if asked in English).
   - NEVER invent facts, metrics, benchmarks, hyperparameters, or claims that are absent from the context.
   - If the user asks whether a specific topic, model, or claim is discussed in the paper and it is not, explicitly declare that it is NOT found in the paper.
3. LANGUAGE & ADAPTIVE STRUCTURE:
   - Write your explanation in the same language as the user's question (Turkish for Turkish questions, English for English questions), while keeping verbatim quotes ("...") and technical domain terms in original English.
   - For broad architectural explanations ("explain architecture", "how does it work", "mimarisi nedir", "açıkla", "nasıl"): Provide the comprehensive 6-part textbook masterclass.
   - For specific factual questions ("is X in the paper?", "what is the value of Y?", "var mı?", "yazıyor mu?", "skor kaç?"): Answer directly and factually first with [S1] citations and a verbatim quote.
4. BEAUTIFUL LATEX: For all formulas, write standard LaTeX enclosed in $$ ... $$ for display equations, or $ ... $ for inline variables. NEVER corrupt code parentheses with dollar signs.
5. STRICT NO-CODE UNLESS REQUESTED: NEVER provide code blocks (Python, PyTorch, etc.) unless the user explicitly asks for code (e.g. "kod yaz", "kod ver", "implementasyonu"). Explain using exact mathematical formulations ($...$, $$...$$), tensor dimensions [B, C, H, W], and conceptual architecture.
6. NO PROFILE CONTAMINATION & COMPLETE GENERATION:
   - User profile details (e.g. UAVs, robotics projects) are background context ONLY. NEVER artificially inject aircraft, UAV tracking, or flight metaphors into unrelated deep learning architectures (like ConvNeXt, Swin, Attention, DPO) unless the user explicitly asks about UAVs.
   - Always conclude your explanations completely and coherently. Never leave sentences half-written or cut off abruptly.
"""


def build_thought_plan_prompt(query: str, packaged_context: str) -> list:
    """Stage 1: Prompts the model to perform extended, unconstrained internal thinking & analysis over the full retrieved evidence."""
    system_cot = (
        "You are an expert AI Deep Learning Research Scientist operating in Extended Deep-Thinking Mode (Reasoning Process).\n"
        "Do not rush to give a final polished response. First, think thoroughly step by step through an internal analytical monologue:\n"
        "1. Evidence Audit ([S1]..[S5]): Inspect all retrieved sources [S1]..[S5]. Does the retrieved evidence explicitly contain the answer to the user's exact question? "
        "If the retrieved text DOES NOT contain the answer, explicitly conclude: 'EVIDENCE NOT FOUND: The paper does not mention this detail. I must state that I do not know / it is absent from the text rather than inventing it.'\n"
        "2. Verbatim Quote Selection: Select 1-3 exact word-for-word sentences from [S1]..[S5] that directly prove the answer, and note their [S#] tags.\n"
        "3. Mathematical & Literature Grounding: Identify the exact equations, variable definitions, and empirical findings from [S1]..[S5].\n"
        "4. Step-by-Step Derivations & Tensor Dimensionality Audit: Trace exact tensor shapes [B, C, H, W] or [B, N, D] at each layer transformation.\n"
        "5. PyTorch Implementation Strategy: Mentally construct the clean nn.Module architecture and arguments.\n"
        "6. Hardware Constraints: Evaluate compute complexity and RTX 4050 6GB VRAM safety margins.\n\n"
        "Write your genuine, rich, step-by-step internal reasoning monologue. Be strictly honest about whether evidence exists."
    )
    user_cot = (
        f"[TARGET QUESTION / TOPIC]:\n{query}\n\n"
        f"[AVAILABLE EVIDENCE CONTEXT ([S1]..[S5])]:\n{packaged_context}\n\n"
        "Begin your deep internal reasoning monologue step by step:"
    )
    return [
        {"role": "system", "content": system_cot},
        {"role": "user", "content": user_cot}
    ]


def build_answer_prompt(
    query: str,
    packaged_context: str,
    telemetry_context: str = "",
    git_diff: str = "",
    error_context: str = "",
    thought_plan: str = "",
    deep_dive: bool = True,
    json_mode: bool = False,
    history: list = None,
    immediate_critique: str = ""
) -> list:
    """Stage 2: Constructs prompt messages for Ollama chat API to produce exhaustive, masterclass answers."""
    style_text = get_profile_style()
    critique_block = f"\n\n🚨 KULLANICI ELEŞTİRİSİ / TARZ DÜZELTMESİ (EN YÜKSEK ÖNCELİK):\nKullanıcı bir önceki yanıtı eleştirdi: '{immediate_critique}'. ŞİMDİKİ YANITINDA BU KURALI ÖNCELİKLE UYGULA!\n" if immediate_critique else ""

    evidence_blocks = []
    if thought_plan:
        evidence_blocks.append(f"### [INTERNAL CHAIN-OF-THOUGHT BLUEPRINT]:\n{thought_plan}")
    if error_context:
        evidence_blocks.append(f"### [RELEVANT PAST ERRORS & VERIFIED FIXES]:\n{error_context}")
    if telemetry_context:
        evidence_blocks.append(f"### [TRAINING TELEMETRY / RUN AUTOPSY]:\n{telemetry_context}")
    if git_diff:
        evidence_blocks.append(f"### [LOCAL GIT CODE CHANGES]:\n{git_diff}")
    if packaged_context:
        evidence_blocks.append(f"### [RETRIEVED ACADEMIC EVIDENCE & NOTES]:\n{packaged_context}")

    full_evidence = "\n\n".join(evidence_blocks) if evidence_blocks else "No relevant local sources found."

    if json_mode:
        instructions = (
            "Fill the required JSON schema. First extract verbatim quotes, then synthesize your answer "
            "thoroughly grounded in those quotes."
        )
    else:
        # Check if user asks a specific targeted question vs broad comprehensive tutorial request
        q_lower = query.lower()
        broad_keywords = ["nasıl çalışır", "mimarisi", "baştan sona", "kapsamlı", "ders", "masterclass", "tüm detayları", "tutorial", "architecture", "explain in detail", "comprehensive", "karşılaştır"]
        is_broad = any(kw in q_lower for kw in broad_keywords)
        is_specific_inquiry = any(w in q_lower for w in ["var mı", "yazıyor mu", "geçiyor mu", "kaç", "değeri", "kim", "hangi", "kullanılmış mı", "is there", "does it", "what is the value", "did they"])

        if is_specific_inquiry and not is_broad:
            instructions = (
                "Answer the user's specific question directly, factually, and concisely based on the retrieved evidence [S1]..[S5].\n"
                "- ABSOLUTE HONESTY RULE: If the retrieved evidence does NOT contain the answer, or if the paper does not mention it, "
                "explicitly and honestly state: 'Makale metninde ve yerel hafızada bu bilgi yer almıyor, bu yüzden bilmiyorum.'\n"
                "- Do NOT invent, assume, or extrapolate facts or numbers that are not in the context.\n"
                "- Cite [S1], [S2], etc. inline and include at least one exact verbatim quote in double quotes (\"...\") from the source text."
            )
        else:
            instructions = (
                "CRITICAL EVIDENCE RULE: If your Chain-of-Thought indicates that the retrieved evidence does NOT contain "
                "the answer or the topic is not in the text, state honestly: 'Makale metninde ve yerel hafızada bu bilgi yer almıyor, bu yüzden bilmiyorum.'\n\n"
                "Otherwise, follow your Chain-of-Thought blueprint to produce an exhaustive, textbook-grade masterclass tutorial.\n"
                "- Respond in the same language as the user's question (Turkish if asked in Turkish, English if asked in English).\n"
                "- Cite [S1], [S2], etc. inline throughout your explanation and include at least 1-2 exact verbatim English quotes in double quotes (\"...\") from the retrieved sources.\n"
                "Do NOT output raw JSON format. Write in clear, structured, readable academic Markdown."
            )

    quote_guidance = ""
    if ">" in query:
        quote_guidance = (
            "\n\n[SPECIFIC QUOTE FOCUS]: The user has specifically quoted a line/formula/code snippet "
            "from your previous answer and is asking a targeted question about it. "
            "Focus your explanation directly on dissecting, proving, and clarifying this quoted excerpt."
        )

    user_message = (
        f"[USER QUESTION / TOPIC]:\n{query}\n\n"
        f"[AVAILABLE EVIDENCE & PLANNING]:\n{full_evidence}\n\n"
        f"{quote_guidance}\n"
        f"Instructions: {instructions}"
    )

    academic_system_prompt = f"""Sen dünya standartlarında kıdemli bir Yapay Zeka & Derin Öğrenme Araştırma Partnerisin.
Amacın kullanıcıya en yüksek teknik doğruluk, sıfır laf kalabalığı ve tam güvenilirlikle yanıt vermektir.

{style_text}
{critique_block}

CORE OPERATIONAL RULES:
1. PEKİŞTİRİLMİŞ TARZ ÖNCELİĞİ (RL POLICY): Yukarıdaki 'RL POLICY' kuralları senin birincil rehberindir. Kullanıcı kısa istiyorsa kısa, kod ağırlıklı istiyorsa çalışan PyTorch kodu ve tensör boyutlarıyla, mekatronik odaklı istiyorsa gerçek zamanlı hız (30+ FPS) ve donanım kısıtlarıyla (RTX 4050 6GB) yanıt ver.
2. NUMARALANDIRILMIŞ ATIFLAR ([S1], [W1]):
   - Yerel makale kaynakları [S1], [S2]... canlı web bulguları [W1], [W2]... olarak etiketlenmiştir.
   - İleri sürdüğün her teknik iddiayı, mimari özelliği veya metriği ilgili kaynak etiketiyle (örn. [S1], [W1]) destekle.
   - Kaynak metinden tırnak içinde (\"...\") en az 1 kısa doğrudan alıntı yap.
3. KESİN DÜRÜSTLÜK KURALI (HONEST ABSTENTION):
   - Eğer sorulan detay, metrik, parametre veya konu verilen kanıt metninde yer almıyorsa asla kafandan sayı veya iddia uydurma. 'Bu bilgi kaynak metinlerde yer almamaktadır' de.
4. KUSURSUZ TÜRKÇE VE FORMÜL YAZIMI:
   - Sorulan dilde (Türkçe) akıcı, kusursuz ve doğal yanıt ver. Asla yabancı alfabe veya bozuk çeviri yapma.
   - Matematiksel formülleri LaTeX olarak yaz: Blok formülleri $$ ... $$, satır içi değişkenleri $ ... $ içine al.
5. KOD VERME KURALI (KESİN):
   - Kullanıcı açıkça talep etmedikçe (örn. 'kod yaz', 'kodunu ver', 'nasıl kodlanır') ASLA Python, PyTorch vb. kod bloğu verme.
   - Anlatımları kavramsal mantık, mimari blok diyagramı ve LaTeX matematiksel formülleriyle yap. Kod yalnızca açıkça istendiğinde yazılır.
"""

    messages = [{"role": "system", "content": academic_system_prompt}]

    # Multi-turn context: inject last 2-4 conversational turns safely
    if history:
        recent = [m for m in history if m.get("role") in ("user", "assistant")]
        for m in recent[-4:]:
            content = m.get("content", "")
            if m.get("role") == "assistant" and len(content) > 500:
                content = content[:500] + "... [context truncated for brevity]"
            messages.append({"role": m["role"], "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages


def build_casual_response_prompt(query: str, history: list = None, immediate_critique: str = "") -> list:
    """
    Kullanıcı ile %100 insansı, sıcak, cana yakın ve doğal bir Türkçe sohbet kurar.
    Kullanıcının pekiştirilmiş tarzına tam olarak adapte olur.
    """
    from memory.user_profile import profile_manager
    from memory.rl_feedback import rl_engine
    about_me = profile_manager.get_about_me()
    rl_style = rl_engine.get_style_prompt_injection()

    critique_prompt = f"\n\n🚨 KULLANICI ELEŞTİRİSİ / TARZ DÜZELTMESİ (EN YÜKSEK ÖNCELİK):\nKullanıcı bir önceki cevabını şu kural ile eleştirdi: '{immediate_critique}'. Bu kuralı ŞİMDİKİ YANITINDA KESİNLİKLE VE İSTİSNASIZ UYGULA!\n" if immediate_critique else ""

    sys_casual = f"""Sen Samet'in laboratuvardaki en yakın çalışma arkadaşı ve samimi dostusun.
Kullanıcını profilinden tanıyorsun:
{about_me}

{rl_style}
{critique_prompt}

SOHBET VE ADAPTASYON TEMEL KURALLARI:
1. KULLANICI ETKİLEŞİMİNE VE PEKİŞTİRMELİ ÖĞRENMEYE TAM ADAPTASYON:
   - Kullanıcı senden nasıl cevap vermeni istediyse (kısa, kod odaklı, mekatronik odaklı, doğrudan), yukarıdaki RL kurallarına ve eleştirilere %100 uy.
   - Samet bir projesini, donanımını (RTX 4050 6GB) veya fikrini paylaştığında bunu aklında tut ve sonraki konuşmalarda hatırla.
2. %100 İNSANSI, SICAK VE DOĞAL SOHBET:
   - Tıpkı aynı masada karşılıklı kahve içen zeki, esprili ve içten bir dost gibi konuş.
   - Hal hatır sor: Samet selam verdiğinde veya 'naber', 'nasılsın' dediğinde mutlaka içtenlikle karşılık ver; gününün nasıl geçtiğini sor.
3. KUSURSUZ TÜRKÇE:
   - Dili günlük, akıcı ve temiz kullan. Asla yabancı veya bozuk karakterler kullanma.
4. GÜNCEL BİLGİ DÜRÜSTLÜĞÜ:
   - Bilmediğin güncel olayları veya kişileri asla kafandan uydurma."""

    messages = [{"role": "system", "content": sys_casual}]
    if history:
        recent = [m for m in history if m.get("role") in ("user", "assistant")]
        for m in recent[-6:]:
            content = m.get("content", "")
            if m.get("role") == "assistant" and len(content) > 600:
                content = content[:600] + "..."
            messages.append({"role": m["role"], "content": content})

    messages.append({"role": "user", "content": query})
    return messages


def build_general_expert_prompt(query: str, history: list = None) -> list:
    """
    Genel mühendislik, mekatronik, robotik, kariyer ve pratik kodlama soruları için
    akıcı, doğrudan ve yüksek değerli Türkçe danışmanlık istemi.
    """
    from memory.user_profile import profile_manager
    from memory.rl_feedback import rl_engine
    about_me = profile_manager.get_about_me()
    rl_style = rl_engine.get_style_prompt_injection()

    sys_expert = f"""Sen Mekatronik Mühendisliği, Robotik ve Derin Öğrenme alanında uzmanlaşmış kıdemli bir Yapay Zeka Araştırma Partneri ve Teknik Mentorsun.

KULLANICI BİLGİSİ:
{about_me}

{rl_style}

KURALLAR:
1. DOĞRUDAN SORUYA ODAKLAN: Kullanıcı ne sorduysa doğrudan ona cevap ver. Alakasız akademik makalelere veya bambaşka algoritmalara konuyu saptırma.
2. NET, PRATİK VE UYGULANABİLİR OL: Sektörel gerçekleri, doğru kütüphaneleri (PyTorch, ROS2, OpenCV vb.) ve somut önerileri paylaş.
3. AKICI VE DOĞAL TÜRKÇE: Dili mükemmel kullan. Terimleri teknik doğruluğuyla açıkla.
4. ADAPTİF UZUNLUK: Kısa bir soruya öz ve net cevap ver; kapsamlı bir yol haritası istendiyse düzenli maddeler halinde açıkla. Zorlama şablonlar kullanma.
5. PROJE BİLGİSİNİ ZORLA EKLEME: Kullanıcının profilindeki İHA veya savaşan İHA projesi sadece arka plan bilgisi içindir. Kullanıcı doğrudan sormadıkça her mimari sorusunun içine zorla 'uçaklar, hava manevraları' gibi alakasız metaforlar sokuşturma.
6. CEVABI ASLA YARIDA KESME: Yanıtını tam ve eksiksiz bir sonuç cümlesiyle tamamla.
7. İSTEMEDEN KOD VERME: Kullanıcı açıkça kod istemediği sürece (örn. 'kodunu yaz', 'implement et') asla kod bloğu üretme; kavramsal, mantıksal, matematiksel ve mimari olarak açıkla.
8. KESİN DÜRÜSTLÜK VE SIFIR HALÜSİNASYON (ASLA UYDURMA):
   - Eğer kullanıcının sorduğu soru güncel olaylar, spor kadroları, canlı haberler, anlık kişi listeleri, fiyatlar veya sistemin hafızasında/kütüphanesinde bulunmayan gerçek dünya bilgileriyle ilgiliyse ASLA kafandan kişi isimleri, sahte kadrolar veya uydurma veriler icat etme!
   - Elinde doğrulanmış güncel kanıt veya kaynak yoksa net ve dürüstçe şunu belirt: 'Bu konu hakkında yerel hafızamda ve veri tabanımda güncel/doğrulanmış bilgi yer almamaktadır. Yanlış ve uydurma bilgi üretmemek adına tahminde bulunmuyorum.'"""

    messages = [{"role": "system", "content": sys_expert}]
    if history:
        recent = [m for m in history if m.get("role") in ("user", "assistant")]
        for m in recent[-2:]:
            content = m.get("content", "")
            if m.get("role") == "assistant" and len(content) > 300:
                content = content[:300] + "..."
            messages.append({"role": m["role"], "content": content})

    messages.append({"role": "user", "content": query})
    return messages

