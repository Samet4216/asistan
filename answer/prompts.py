"""
Prompts Module for Structured Answer Pipeline (Faz 7).
Incorporates personal learning style from Vault/00_Profile/Style.md,
enforces strict source attribution, and demands honest abstention when evidence is lacking.
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
1. CITATION FIRST: You must extract verbatim quotes from the provided context before generating your answer.
2. NO HALLUCINATIONS: If the context does not contain enough evidence to answer with certainty, set "has_evidence": false, and state: "Insufficient evidence in local knowledge base to verify this."
3. TECHNICAL RIGOR: Preserve mathematical equations, layer configurations, and tensor dimensions.
4. LANGUAGE: Answer strictly in technical English. Domain terms must stay in original English.
5. FORMULA READABILITY: Write mathematical equations and tensor dimensions in clear, readable notation (e.g. C × H × W, z_c^h, f = δ(...), σ(...)). Avoid broken escaped macros like \\text{...} or raw \\times that corrupt inside JSON strings.
"""


def build_answer_prompt(
    query: str,
    packaged_context: str,
    telemetry_context: str = "",
    git_diff: str = "",
    error_context: str = "",
    deep_dive: bool = False
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

    if deep_dive:
        instructions = (
            "Provide an exhaustive, textbook-grade, highly structured pedagogical tutorial. "
            "Teach this concept thoroughly from first principles. "
            "Use clear Markdown formatting with generous line spacing:\n"
            "1. Core Intuition & The Fundamental Problem it Solves.\n"
            "2. Mathematical Equations (write equations clearly on their own lines, e.g. z_c^h(h) = 1/W ∑ x_c(h, i)).\n"
            "3. Tensor Transformation Walkthrough (trace batch, channel, height, width at each step).\n"
            "4. PyTorch Code Implementation (complete, self-contained nn.Module snippet).\n"
            "5. Engineering Bottlenecks & Hardware Limits on RTX 4050 6GB GPU."
        )
    else:
        instructions = (
            "Fill the required JSON schema. First extract verbatim quotes, then synthesize your answer "
            "thoroughly grounded in those quotes. Ensure mathematical formulations and tensor dimensions "
            "are completely spelled out and formatted with clean, readable spacing."
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
