"""
Instant Ollama Custom Model Creator (Faz 12.0).
Compiles Samet's personal conversation history and style into an Ollama model:
`samet-partner:latest`.
Instant 0-VRAM creation, 100% offline, fully personalized.
"""

import sys
import io
import subprocess
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from memory.conversation_collector import collector


def create_personal_model(model_name: str = "samet-partner") -> bool:
    print("=" * 65)
    print(f"🚀 OLLAMA KİŞİSEL MODEL DERLEME: {model_name}")
    print("=" * 65)

    # 1. Generate Modelfile
    print("[*] 1. Samet'in konuşmaları ve profilinden Modelfile oluşturuluyor...")
    collector.build_training_dataset()
    modelfile_path = collector.generate_ollama_modelfile(base_model=config.MODEL)
    print(f"    Modelfile hazır: {modelfile_path}")

    # 2. Run ollama create
    print(f"[*] 2. 'ollama create {model_name} -f {modelfile_path}' çalıştırılıyor...")
    try:
        cmd = ["ollama", "create", model_name, "-f", modelfile_path]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0:
            print(f"✅ Başarılı! '{model_name}' yerel modeliniz oluşturuldu.")
            print(res.stdout)
            return True
        else:
            print(f"[!] Ollama create çıktısı (Hata kodu {res.returncode}):")
            print(res.stderr or res.stdout)
            return False
    except FileNotFoundError:
        print("[!] 'ollama' komutu bulunamadı. Lütfen Ollama'nın kurulu ve PATH'e ekli olduğundan emin olun.")
        return False
    except Exception as e:
        print(f"[!] Hata: {e}")
        return False


if __name__ == "__main__":
    create_personal_model()
