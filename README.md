# 🔬 Kişisel Uzman Yapay Zekâ Asistanı ve Model Araştırma Laboratuvarı (`personal-ai`)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1+cu121](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1%20RTX%204050-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://www.trychroma.com/)
[![Tests](https://img.shields.io/badge/Tests-76%2F76%20Passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

NVIDIA RTX 4050 (6 GB VRAM) donanım bütçesi için optimize edilmiş, yerel çalışan, çok ajanlı, deterministik RAG tabanlı ve sıfırdan sinir ağı araştırma yeteneklerine (Scratch Transformer, LoRA, Distillation, DPO, Sparse MoE) sahip kişisel mühendislik yapay zekâ asistanı.

---

## ⚡ Temel Yetenekler ve Mimari Özellikler

- **%100 Yerel ve Gizli:** Dış bulut API'larına bağımlılık olmadan, tüm ağırlıklar ve veriler yerel bilgisayarınızda çalışır.
- **Dört Uzman Ajan (Agent Orchestration):**
  - 🧠 `deep_learning`: Sinir ağları, mimari araştırmaları, backpropagation ve optimizasyon.
  - 👁️ `cv_uav`: Bilgisayarlı görü, nesne tespiti/takibi, İHA güdüm ve edge deployment.
  - 📐 `math_eng`: SymPy destekli deterministik sembolik türev, integral, özdeğer ve lineer cebir çözücü.
  - 💻 `software_eng`: Python AST analizi, C++ sözdizimi ve mimari doğrulama.
- **RAG & Kişisel Bilgi Kasası (Vault):**
  - Sayfa/paragraf referanslı alıntı üretimi (PDF, Markdown, TXT).
  - 93 kişisel makale kartı ve ders notu (2,411 chunk) `data/chroma` koleksiyonunda indekslenmiştir.
  - Sıfır halüsinasyon politikası: Yeterli bağlam yoksa dürüstçe yanıt vermez.
- **Donanım ve Bellek Güvenliği:**
  - RTX 4050 (6 GB VRAM) için `num_ctx: 2048` ve CPU tabanlı embedding ile OOM riski sıfırlanmıştır.
  - Bellek koruma tavanı (`max_vram_mb: 5000.0`).
- **Araştırma Laboratuvarı (Research Modules):**
  - **Phase 7:** RoPE, SwiGLU, RMSNorm ve Weight Tying içeren `ScratchCausalTransformer`.
  - **Phase 8:** Hinton bilgi damıtma (KD) kaybı ve öğretmen/öğrenci telemetrisi (`research/distillation`).
  - **Phase 9:** Rafailov vd. Doğrudan Tercih Optimizasyonu (`DPOLoss`) ve doğrulanabilir ödül motoru (`personal_ai.rl`).
  - **Phase 10:** Top-K gürültülü yönlendirici, Shazeer yük dengeleme kaybı ve paylaşımlı uzmanlı `SparseMoETransformer` (`research/moe`).

---

## 🚀 Hızlı Başlangıç

### 1. Asistanı Çalıştırma
Doğrudan masaüstü / klasör kökünden çift tıklayarak:
```cmd
launch.bat
```
veya terminalden:
```powershell
.\.venv\Scripts\python.exe -m personal_ai.app.cli
```

### 2. Etkileşimli Slash Komutları

| Komut | Açıklama |
| :--- | :--- |
| `/agent auto` | Otomatik niyet sınıflandırma ve uzman yönlendirme (Varsayılan). |
| `/agent dl` | Deep Learning uzman ajan modunu kilitler. |
| `/agent cv` | Computer Vision & UAV uzman ajan modunu kilitler. |
| `/agent math` | Matematik & Mühendislik uzman ajan modunu kilitler. |
| `/agent sw` | Yazılım Mühendisliği & Kodlama uzman ajan modunu kilitler. |
| `/rag on` | RAG doküman erişimini zorunlu kılar (Kişisel makalelerden alıntı yapar). |
| `/rag off` | RAG doküman erişimini kapatır. |
| `/add <yol>` | Yeni bir PDF, MD veya TXT dosyasını vektör hafızasına ekler. |
| `/index <yol>` | Bir klasördeki tüm dokümanları toplu olarak indeksler. |
| `/fb +1 [not]` | Asistanın son cevabını olumlu etiketle hafızaya kaydeder. |
| `/fb -1 [not]` | Asistanın son cevabındaki hatayı critique olarak kaydeder. |
| `/fb verify <id>` | Geri bildirimi kalite onayından geçirerek eğitim havuzuna kabul eder. |
| `/clear` | Sohbet geçmişini sıfırlar. |
| `/exit` | Asistandan çıkar. |

---

## 🧪 Test Doğrulama ve Güvenilirlik

Projenin tüm 11 fazı %100 test güvencesiyle korunmaktadır:
```powershell
.\.venv\Scripts\pytest.exe tests/unit tests/integration tests/hardware -q
```
**Test Özeti:**
- 4 Donanım ve CUDA ortam testi
- 63 Birim testi (Agent, RAG, LoRA, Math, Code, Scratch Transformer, Distillation, DPO, MoE)
- 9 Entegrasyon testi (Pipeline uçtan uca akışları)
- **Toplam:** **76 passed in 22.54s (%100 Başarı)**

---

## 📁 Proje Dizin Yapısı

```text
├── configs/              # Donanım, RAG, Ajan ve Eğitim YAML konfigürasyonları
├── data/
│   ├── chroma/           # Kalıcı ChromaDB vektör veritabanı (2411 chunk indeksli)
│   ├── evaluation/       # 10 görevlik standardize DL/CV benchmark kümesi
│   └── feedback/         # Kullanıcı düzeltmeleri ve onaylı tercih verileri
├── research/
│   ├── scratch_transformer/ # Sıfırdan PyTorch Transformer (RoPE, SwiGLU)
│   ├── distillation/        # Knowledge distillation & edge benchmark motoru
│   └── moe/                 # Sparse Mixture of Experts & Top-K yönlendirici
├── src/personal_ai/
│   ├── app/              # Rich terminal CLI uygulaması
│   ├── agents/           # 4 uzman mühendislik ajanı
│   ├── memory/           # Geri bildirim ve eğitim verisi yönetim sistemi
│   ├── rag/              # Doküman yükleme, chunking ve ChromaDB entegrasyonu
│   ├── rl/               # Doğrulanabilir ödül motoru ve ödül manipülasyonu savunması
│   ├── routing/          # Görev yönlendirme ve orkestrasyon motoru
│   ├── runtime/          # Ollama yerel model streaming istemcisi
│   ├── tools/            # SymPy matematik ve AST kod doğrulama araçları
│   └── training/         # LoRA, Distillation ve DPO eğitim motorları
├── tests/
│   ├── hardware/         # CUDA ve bellek testleri
│   ├── integration/      # Tam sistem entegrasyon testleri
│   └── unit/             # Modüler birim testleri
└── launch.bat            # Tek tıkla yerel asistan başlatıcı
```
