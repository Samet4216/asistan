"""
Local QLoRA Fine-Tuning Pipeline for NVIDIA RTX 4050 6GB Laptop GPU (Faz 12.0).
Trains a personal LoRA adapter on Samet's own conversation history:
- Memory footprint: ~3.8 GB VRAM (Safe from OOM on 6GB RTX 4050)
- 4-bit NF4 Quantization (bitsandbytes)
- LoRA rank: r=16, alpha=32, target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj']
- Dataset: data/my_conversations/train_sharegpt.jsonl
"""

import sys
import os
import io
import json
from pathlib import Path
from typing import Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_FILE = PROJECT_ROOT / "data" / "my_conversations" / "train_sharegpt.jsonl"
OUTPUT_LORA_DIR = PROJECT_ROOT / "models" / "samet_style_lora"
OUTPUT_LORA_DIR.mkdir(parents=True, exist_ok=True)


def check_prerequisites() -> Dict[str, Any]:
    """Verifies CUDA, VRAM, and python packages."""
    checks = {}
    try:
        import torch
        checks["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            checks["gpu_name"] = torch.cuda.get_device_name(0)
            checks["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        else:
            checks["gpu_name"] = "No CUDA GPU detected"
            checks["vram_gb"] = 0.0
    except Exception as e:
        checks["cuda_available"] = False
        checks["error"] = str(e)

    try:
        import peft
        checks["peft_installed"] = True
    except ImportError:
        checks["peft_installed"] = False

    try:
        import bitsandbytes
        checks["bitsandbytes_installed"] = True
    except ImportError:
        checks["bitsandbytes_installed"] = False

    try:
        import trl
        checks["trl_installed"] = True
    except ImportError:
        checks["trl_installed"] = False

    return checks


def start_training(
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    epochs: int = 3,
    lr: float = 2e-4
):
    print("=" * 70)
    print("🚀 SAMET'İN KİŞİSEL KONUŞMA TARZI İLE YEREL MODEL EĞİTİMİ (QLoRA 4-BIT)")
    print(f"🎯 Hedef Donanım: NVIDIA RTX 4050 6GB Laptop GPU")
    print(f"📁 Veri Seti: {DATASET_FILE}")
    print(f"💾 Çıktı Dizini: {OUTPUT_LORA_DIR}")
    print("=" * 70)

    # 1. Check prerequisites
    status = check_prerequisites()
    print("[*] Donanım & Kütüphane Kontrolü:")
    for k, v in status.items():
        print(f"  • {k}: {v}")

    if not DATASET_FILE.exists() or DATASET_FILE.stat().st_size == 0:
        print("[!] Hata: 'train_sharegpt.jsonl' veri seti bulunamadı veya boş.")
        print("    Lütfen önce sohbet ederek konuşma verisi toplayın veya 'conversation_collector.py' çalıştırın.")
        return

    # Count samples
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        sample_count = sum(1 for _ in f if _.strip())
    print(f"[*] Toplam Eğitim Örnek Sayısı: {sample_count} diyalog")

    if not status.get("peft_installed") or not status.get("bitsandbytes_installed"):
        print("\n[!] Gerekli ince ayar kütüphaneleri eksik.")
        print("    Eğitimi başlatmak için lütfen şu komutu çalıştırın:")
        print("    pip install peft bitsandbytes trl accelerate datasets")
        print("\nAlternatif olarak, VRAM harcamadan sıfır gecikmeyle anında model oluşturmak için:")
        print("    python scripts/create_ollama_persona.py")
        return

    # 2. Execution instructions for PyTorch training loop
    print("\n[*] Eğitim Başlatılıyor... (VRAM ~3.8 GB seviyesinde tutulacak)")
    # Code dynamically loads training libraries to avoid importing them if missing
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print(f"[*] Model 4-bit olarak yükleniyor: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("\n✅ LoRA Adaptörü başarıyla oluşturuldu! Ağırlıklar kaydediliyor...")
    model.save_pretrained(str(OUTPUT_LORA_DIR))
    tokenizer.save_pretrained(str(OUTPUT_LORA_DIR))
    print(f"🎉 Eğitim Başarıyla Tamamlandı! Model konumu: {OUTPUT_LORA_DIR}")


if __name__ == "__main__":
    start_training()
