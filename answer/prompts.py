"""
Prompts Module for Structured Answer Pipeline (Faz 7 & Web UI).
Incorporates personal learning style from Vault/00_Profile/Style.md,
enforces strict source attribution, beautiful LaTeX math, and clean textbook formatting.
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


SYSTEM_PROMPT = """You are a rigorous, academic Deep Learning Lab Assistant and research partner.
Your role is to assist with neural network architecture design, training telemetry autopsies, and paper synthesis.

CORE OPERATIONAL RULES:
1. CITATION FIRST: Reference the indexed papers and equations from the provided context.
2. NO HALLUCINATIONS: If the context does not contain enough evidence to answer with certainty, state: "Insufficient evidence in local knowledge base to verify this."
3. FORMATTING & READABILITY: Output cleanly structured, beautiful GitHub-flavored Markdown:
   - Use clear markdown headers (e.g. ### 1. Core Intuition, ### 2. Mathematical Formulation, ### 3. Tensor Shapes).
   - Use generous spacing and distinct paragraphs for comfortable reading.
   - For all formulas, write standard LaTeX enclosed in $$ ... $$ for display equations, or $ ... $ for inline variables.
   - NEVER output raw JSON or curly brace schemas unless the user explicitly commands you to output JSON.
4. TECHNICAL RIGOR: Preserve exact mathematical formulations, layer configurations, and tensor dimensions.
5. DOMAIN TERMS: Keep domain terms in original technical English (e.g. multi-head attention, bottleneck, patch embedding).
"""


def build_answer_prompt(
    query: str,
    packaged_context: str,
    telemetry_context: str = "",
    git_diff: str = "",
    error_context: str = "",
    deep_dive: bool = False,
    json_mode: bool = False
) -> list:
    """Constructs prompt messages for Ollama chat API."""
    style_text = get_profile_style()

    evidence_blocks = []
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
    elif deep_dive:
        instructions = (
            "Provide an exhaustive, textbook-grade pedagogical tutorial. Teach this concept thoroughly from first principles. "
            "Use clear Markdown formatting with generous line spacing and sections:\n"
            "### 1. Core Intuition & The Fundamental Problem it Solves\n"
            "### 2. Mathematical Formulation (write display equations in $$ ... $$ on their own lines)\n"
            "### 3. Tensor Transformation Walkthrough (trace batch, channel, height, width at each step)\n"
            "### 4. PyTorch Code Implementation (complete, clean nn.Module snippet)\n"
            "### 5. Engineering Gotchas & Hardware Limits on RTX 4050 6GB GPU.\n"
            "Do NOT output raw JSON format. Write in clear academic Markdown."
        )
    else:
        instructions = (
            "Provide a well-structured, clear academic answer in GitHub-flavored Markdown. "
            "Organize your explanation with clear sections:\n"
            "### 1. Intuition & Architectural Role\n"
            "### 2. Mathematical Equations (write display equations inside $$ ... $$ on separate lines)\n"
            "### 3. Tensor Dimensions & Dataflow (e.g., shapes like [B, N, C] or [B, C, H, W])\n"
            "### 4. Summary & Key Takeaways\n"
            "Do NOT output raw JSON format. Write in clean, formatted Markdown."
        )

    user_message = (
        f"[USER QUESTION / TOPIC]:\n{query}\n\n"
        f"[AVAILABLE EVIDENCE]:\n{full_evidence}\n\n"
        f"[USER LEARNING STYLE]:\n{style_text}\n\n"
        f"Instructions: {instructions}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
