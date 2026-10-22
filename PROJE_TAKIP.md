# Local Deep Learning Lab Assistant — Proje Takip ve Yol Haritası (v2)

> **Mimarisi:** Saf Python modülleri (No heavy blackbox frameworks)  
> **Donanım:** Asus TUF A15 | RTX 4050 (6 GB VRAM) | 16 GB RAM | Python 3.11  
> **Dil Kuralı:** %100 İngilizce (Teknik/Akademik), İstek üzerine `/tr`  
> **LLM Motoru:** Ollama HTTP API (`keep_alive`, sabit `num_ctx`, JSON schema)  
> **Embedding & Reranker:** CPU üzerinde (`bge-base-en-v1.5`, `cross-encoder`)  
> **Son Güncelleme:** 2026-10-22

---

## 📋 Faz Takip Çizelgesi (Final Yol Haritası v2)

| Faz | Açıklama | Durum | Notlar |
| :--- | :--- | :---: | :--- |
| **FAZ 0: Kabul Testi** | 5 gerçek derin öğrenme hatasıyla İngilizce yeterlilik ve ısı testi | 🟢 Hazır | Ollama ve GPU test edildi (4.1 GB VRAM, 53°C). Model hazır. |
| **FAZ 1: İskelet & Temel Katman** | `config.py`, `llm.py`, `budget.py`, `vault.py`, `lang_guard.py` | 🟢 Tamamlandı | Çekirdek 5 modül yazıldı, JSON şeması, token bütçesi ve dil koruması testten geçti. |
| **FAZ 2: Telemetri & Log Otopsisi** | `ingest/log_parser.py` (Ultralytics/PyTorch `results.csv`, `args.yaml`) | 🟢 Tamamlandı | Deterministik eğitim logu otopsi modülü yazıldı. Overfitting ve sağlıklı metrik testleri başarıyla geçti. |
| **FAZ 3: Makale İşleme Hattı** | `arxiv_fetch.py`, `paper_parser.py` (LaTeX/PDF), `chunker.py`, `indexer.py` | 🟢 Tamamlandı | arXiv LaTeX kaynağı çekildi, bölümler ve formüller korundu, 56 chunk ChromaDB'ye indekslendi. |
| **FAZ 4: Değerlendirme Seti** | `eval/questions.yaml`, `run_eval.py` | 🟢 Tamamlandı | 10 soruluk eval seti ve Recall@5 / MRR ölçüm tablosu kuruldu. Temel skor: %40 (İndeksli makalede %100 Rank 1). |
| **FAZ 5: Hibrit Retrieval** | `embed.py` (BGE CPU), `search.py` (Chroma + BM25 RRF), `rerank.py` | 🟢 Tamamlandı | Vektör + BM25 (RRF) + Cross-Encoder Reranker CPU'da kuruldu ve test edildi. |
| **FAZ 6: Makale Kartları** | `ingest/cards.py` (Offline batch, JSON şemalı, onay akışı) | 🟢 Tamamlandı | JSON şemalı makale kartı üretimi tamamlandı. Orijinal denklemlerle `Vault/01_Paper_Cards/` oluşturuldu. |
| **FAZ 7: Cevap Hattı & Atıf** | `answer/pipeline.py`, `verify.py`, alıntı-önce-cevap-sonra | 🟢 Tamamlandı | Çekimser kalabilen, %100 alıntı doğrulama (verbatim quote verification) özellikli cevap hattı tamamlandı. |
| **FAZ 8: Hafıza & Kişiselleştirme** | `memory/errors.py`, `user_profile.py`, `/save`, `/approve` | 🟢 Tamamlandı | `Vault/Drafts/` onay akışı, hata hafızası (`Errors.md`), ve profil yönetimi tamamlandı. |
| **FAZ 9: CLI, Git & Isı Yönetimi** | `app.py` (Rich CLI), `retrieval/git_context.py` (`/diff`), idle unload | 🟢 Tamamlandı | Rich interaktif terminali, GPU termal/VRAM takibi, `/diff`, `/logs`, `/save`, `/approve`, `/tr` komutları tamamlandı. |
| **FAZ 10: Tensör Dry-Run Sandbox** | `answer/shape_check.py` (AST whitelist, CPU sandbox, backward pass) | 🟢 Tamamlandı | AST güvenlik denetimli PyTorch CPU sandbox'ı, boyut ve backward autograd doğrulaması tamamlandı. |
| **FAZ 11: Makale + Kod Deposu** | `ingest/repo_link.py` (Resmi repo AST chunking, Eq-to-Code) | 🟢 Tamamlandı | Resmi repo (CoordAttention) klonlandı, AST ile 33 kod bloğu indekslendi, denklem-kod eşleştirmesi kartlara eklendi. |
| **FAZ 12: `/challenge` Modu** | `answer/challenge.py` (Reverse-RAG, Devil's Advocate, VRAM analizi) | 🟢 Tamamlandı | Hipotez çürütme, donanım darboğaz analizi ve çürütücü minimal deney önerisi tamamlandı. |
| **FAZ 13: Genişleme & Doğrulama** | Korpus büyütme, eval süiti, final donanım & bellek doğrulaması | 🟢 Tamamlandı | 5 temel makale (238 chunk) indekslendi. Recall@5 skoru %90.0, MRR: 0.850'ye ulaştı. GPU 43°C'de stabil. |

---

## 📝 Günlük (Changelog)

### [2026-10-22 - 2026-10-23]
- Proje resmi olarak **"Local Deep Learning Lab Assistant: Final Yol Haritası (v2)"** mimarisine geçirildi.
- Bağımlılıklar tamamlandı: `requests`, `chromadb`, `sentence-transformers`, `torch`, `rank-bm25`, `pandas`, `pymupdf4llm`, `rich`.
- **FAZ 1 Tamamlandı:** 5 temel iskelet modülü yazıldı ve test süiti başarıyla geçti.
- **FAZ 2 Tamamlandı:** `ingest/log_parser.py` yazıldı ve sentetik overfitting testleri geçti.
- **FAZ 3 Tamamlandı:** Makale işleme hattı (LaTeX/PDF parser, hiyerarşik chunker, ChromaDB indexer) kuruldu.
- **FAZ 4 Tamamlandı:** `eval/questions.yaml` ve `eval/run_eval.py` kuruldu (10 soru benchmark'ı).
- **FAZ 5 Tamamlandı:** Hibrit arama ve yeniden sıralama katmanı tamamlandı (BGE CPU + BM25 RRF + Cross-Encoder Reranker).
- **FAZ 6 Tamamlandı:** `ingest/cards.py` yazıldı ve çalıştırıldı:
  - Coordinate Attention (`2103.02907`) için JSON şemalı yapılandırılmış özet kartı üretildi.
  - Orijinal LaTeX denklemleri (`z_c^h(h)`, `z_c^w(w)`) hiçbir halüsinasyon olmadan doğrudan kaynak koddan karta gömüldü.
  - Kart `Vault/01_Paper_Cards/2103.02907.md` konumuna kaydedildi ve ChromaDB `paper_cards` koleksiyonuna indekslendi.
- **FAZ 7 Tamamlandı:** `answer/schemas.py`, `answer/prompts.py`, `answer/verify.py`, `answer/pipeline.py` yazıldı:
  - ChromaDB `EmbeddingFunction` ve `embed_query` uyumluluğu çözüldü.
  - Soru -> Dil Kontrolü -> Hibrit Arama -> Telemetri/Git -> Bütçe Paketleme -> LLM -> Alıntı Doğrulama akışı doğrulandı.
  - Coordinate Attention mimarisi sorusu üzerinde %100 alıntı doğrulama skoru alındı.
- **FAZ 8 Tamamlandı:** Hafıza ve kişiselleştirme katmanı tamamlandı:
  - `Vault/00_Profile/Errors.md` oluşturuldu; donanıma özel (RTX 4050 6GB VRAM) ve mimari hatalar tanımlandı.
  - `memory/errors.py` yazıldı; geçmiş hata hafızası ChromaDB `errors` koleksiyonuna indekslendi ve bütçeye uygun dinamik çağrılma sağlandı.
  - `memory/user_profile.py` yazıldı (Python stdlib `profile` çakışması önlendi); stil ve öğrenme durumu yönetildi.
  - `memory/approval.py` yazıldı; `Vault/Drafts/` üzerinden onay mekanizması kuruldu, onaylanan notlar kalıcı hafızaya taşınarak ChromaDB'ye indekslendi ve Vault git takibine alındı.
  - `answer/pipeline.py` içine geçmiş hata hafızası entegre edildi.
- **FAZ 9 Tamamlandı:** `app.py` interaktif Rich CLI tamamlandı:
  - Canlı GPU VRAM (RTX 4050) ve sıcaklık takibi (`/gpu`, `/vram`).
  - Isı yönetimi ve boşta VRAM boşaltma (`/unload`, exit hook).
  - `/diff`, `/logs`, `/drafts`, `/save`, `/approve`, `/reject`, `/tr`, `/eval` komutları bağlandı.
  - Alıntı ve cevap panelleri, renklendirilmiş doğrulanma oranı ve markdown formatlama test edildi.
- **FAZ 10 Tamamlandı:** `answer/shape_check.py` Tensör Dry-Run Sandbox tamamlandı:
  - AST tabanlı güvenlik beyaz listesi kuruldu (`os`, `sys`, `subprocess`, `eval`, tehlikeli `__import__` engellendi).
  - Yalnızca `torch` ve `math` modüllerine izin veren CPU korumalı çalışma ortamı oluşturuldu.
  - Boyut doğrulaması, forward pass ve backward autograd türevlenebilirlik testi başarıyla doğrulandı.
  - `app.py` içine `/dryrun` komutu entegre edildi.
- **FAZ 11 Tamamlandı:** `ingest/repo_link.py` Makale + Resmi Repo Eşleştirmesi tamamlandı:
  - Resmi GitHub deposu (`Andrew-Qibin/CoordAttention`) klonlandı.
  - AST ile `coordatt.py` ve `mbv2_ca.py` sınıfları ve metodları (33 chunk) ChromaDB `code` koleksiyonuna indekslendi.
  - Eq 4, 5, 6, 7-8, 9 formülleri `coordatt.py` dosyasındaki satırlarla birebir eşleştirilip `Vault/01_Paper_Cards/2103.02907.md` kartına eklendi.
- **FAZ 12 Tamamlandı:** `answer/challenge.py` Devil's Advocate / Ters-RAG Modu tamamlandı:
  - Kullanıcının hipotezlerine körü körüne katılmayan, akademik şüphecilikle yaklaşan değerlendirme motoru kuruldu.
  - Literatür kısıtları, bellek darboğazları (split/cat/permute) ve RTX 4050 6GB VRAM riskleri analiz edilip minimal çürütme deneyi önerildi.
  - `app.py` içerisine `/challenge` komutu entegre edildi.
- **FAZ 13 Tamamlandı:** Genişleme, Toplu İndeksleme ve Doğrulama:
  - `ingest/batch_ingest.py` ile 5 temel makale (Transformer, ResNet, ViT, Coordinate Attention, Swin Transformer) indirildi ve ChromaDB `paper_chunks` 238 parçaya genişletildi.
  - `eval/run_eval.py` benchmark süiti çalıştırıldı: **Recall@5 skoru %90.0 (9/10), MRR: 0.850** olarak doğrulandı.
  - GPU donanımı (RTX 4050) boşta 43°C ve 0% yükle stabil bırakıldı. Proje 14 aşamanın tamamında (Faz 0 - 13) %100 başarıyla tamamlandı.






