"""
Language Guard Module for Deep Learning Lab Assistant.
Detects Turkish query inputs to warn the user (preventing embedding mismatch)
and provides on-demand single-call /tr translations.
"""

import re
from typing import Tuple
import llm


TURKISH_CHARS_PATTERN = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
COMMON_TURKISH_WORDS = {
    "nedir", "nasil", "nasıl", "hata", "neden", "ogrenme", "öğrenme",
    "icin", "için", "modeli", "katman", "agirlik", "ağırlık", "kod", "calis",
    "bu", "bir", "ve", "ile", "soru", "sorusu", "sorusudur", "yardim", "yardım",
    "merhaba", "selam", "ne", "hangi", "hangisi", "anlat", "acikla", "açıkla"
}



def detect_turkish(text: str) -> Tuple[bool, str]:
    """
    Checks if a query contains Turkish specific characters or common Turkish words.
    Returns (is_turkish, warning_message).
    """
    has_chars = bool(TURKISH_CHARS_PATTERN.search(text))
    tokens = {w.strip(".,;:?!'\"").lower() for w in text.split()}
    has_words = bool(tokens.intersection(COMMON_TURKISH_WORDS))

    if has_chars or has_words:
        warning = (
            "[NOTICE]: Turkish input detected. "
            "Our semantic embeddings and academic papers are indexed in English. "
            "For optimal retrieval accuracy, please ask technical queries in English. "
            "(You can always use '/tr' after receiving an answer to get a Turkish breakdown)."
        )
        return True, warning

    return False, ""


def translate_to_turkish(english_text: str) -> str:
    """
    Executes a single LLM call to translate an English technical response into clear Turkish,
    strictly keeping all equations, tensor dimensions, and technical terms in English.
    """
    prompt = [
        {
            "role": "system",
            "content": (
                "You are an academic translation assistant. "
                "Translate the following technical response into natural, concise Turkish. "
                "CRITICAL RULES:\n"
                "1. Keep all mathematical formulas ($...$, equation blocks) untouched.\n"
                "2. Keep all code, variable names, and tensor shapes exactly as they are.\n"
                "3. Keep all deep learning technical terms in original English (e.g., self-attention, "
                "residual connections, learning rate, backpropagation, batch norm, gradient descent, epoch, loss).\n"
                "4. Output only the translation, no extra chatter."
            )
        },
        {
            "role": "user",
            "content": english_text
        }
    ]

    return llm.generate(prompt, max_tokens=800)


if __name__ == "__main__":
    print("[*] Testing lang_guard.py detection...")
    is_tr, msg = detect_turkish("ResNet içindeki residual connection neden önemlidir?")
    print(f"Is Turkish: {is_tr}")
    if is_tr:
        print(f"Warning:\n{msg}\n")

    print("[*] Testing /tr translation...")
    sample_en = "Residual connections mitigate vanishing gradients by providing an identity shortcut $x + F(x)$."
    tr_result = translate_to_turkish(sample_en)
    print(f"[EN Input]: {sample_en}")
    print(f"[TR Output]: {tr_result}")
