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
    """Reads user preference style from Vault/00_Profile/Style.md."""
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
1. CITATION FIRST: Reference the indexed papers and specific equations from the provided context.
2. NO HALLUCINATIONS: If the context does not contain enough evidence to answer with certainty, state: "Insufficient evidence in local knowledge base to verify this."
3. FORMATTING & READABILITY: Output cleanly structured, beautiful GitHub-flavored Markdown:
   - Use clear markdown headers (e.g. ### 1. Theoretical Motivation, ### 2. Step-by-Step Mathematical Derivations).
   - Use generous spacing and distinct paragraphs for comfortable reading.
   - For all formulas, write standard LaTeX enclosed in $$ ... $$ for display equations, or $ ... $ for inline variables.
   - NEVER output raw JSON or curly brace schemas unless the user explicitly commands you to output JSON.
4. TECHNICAL RIGOR: Preserve exact mathematical formulations, step-by-step derivations, layer configurations, and tensor dimensions.
5. DOMAIN TERMS: Keep domain terms in original technical English (e.g. multi-head attention, bottleneck, patch embedding).
"""


def build_thought_plan_prompt(query: str, packaged_context: str) -> list:
    """Stage 1: Prompts the model to perform extended, unconstrained internal thinking & analysis."""
    system_cot = (
        "You are an expert AI Deep Learning Research Scientist operating in Extended Deep-Thinking Mode (Reasoning Process).\n"
        "Do not rush to give a final polished response. First, think thoroughly step by step through an internal analytical monologue:\n"
        "1. Deconstruct the Question: What is the core theoretical intuition, and what subtle technical traps or edge cases exist?\n"
        "2. Literature Grounding: What specific mathematical equations and empirical findings from the retrieved papers apply?\n"
        "3. Step-by-Step Derivations: Write down the intermediate mathematical steps, proof logic, and notation.\n"
        "4. Tensor Dimensionality Audit: Trace exact tensor shapes [B, C, H, W] or [B, N, D] at each layer transformation to guarantee validity.\n"
        "5. PyTorch Implementation Strategy: Mentally construct the clean nn.Module architecture, dimension annotations, and arguments.\n"
        "6. Hardware Constraints: Evaluate compute complexity (O(N^2) vs O(N)) and RTX 4050 6GB VRAM safety margins.\n\n"
        "Write your genuine, rich, step-by-step internal reasoning monologue. Label your thoughts clearly."
    )
    user_cot = (
        f"[TARGET QUESTION / TOPIC]:\n{query}\n\n"
        f"[AVAILABLE EVIDENCE CONTEXT]:\n{packaged_context[:1800]}\n\n"
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
    history: list = None
) -> list:
    """Stage 2: Constructs prompt messages for Ollama chat API to produce exhaustive, masterclass answers."""
    style_text = get_profile_style()

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
        instructions = (
            "Follow your Chain-of-Thought blueprint to produce an exhaustive, textbook-grade masterclass tutorial. "
            "Teach this concept with maximum technical depth, step-by-step mathematical rigor, and concrete PyTorch code. "
            "Organize your answer with the following comprehensive sections:\n\n"
            "### 1. Theoretical Motivation & The Problem Solved\n"
            "Explain why this architecture was created, what historical bottleneck it resolves, and the core intuition.\n\n"
            "### 2. Rigorous Mathematical Derivations\n"
            "Write all display equations in $$ ... $$ on separate lines. Define every variable, trace derivations, and explain geometric meaning.\n\n"
            "### 3. Tensor Transformation Walkthrough\n"
            "Step-by-step shape transformations (e.g. Input: [B, C, H, W] -> Projection: [B, N, D] -> ...). Explain why dimensions match.\n\n"
            "### 4. Complete PyTorch Implementation\n"
            "Provide a clean, self-contained, modular nn.Module snippet with type annotations, docstrings, and forward pass explanation.\n\n"
            "### 5. Empirical Performance & Ablation Findings\n"
            "Reference original paper experiments, benchmark results, and what happens when key components are ablated.\n\n"
            "### 6. Hardware Optimization & RTX 4050 Limits\n"
            "Discuss VRAM usage (6GB limit), compute complexity (O(N^2) vs O(N)), recommended batch size, and FP16/BF16 stability.\n\n"
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
        f"[USER LEARNING STYLE]:\n{style_text}{quote_guidance}\n\n"
        f"Instructions: {instructions}"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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
