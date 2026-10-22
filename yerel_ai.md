# Local Deep Learning Lab Assistant: Final Yol Haritası (v2)

> Hedef: Yapay sinir ağları alanında, seçilmiş İngilizce akademik makalelerle beslenen; senin kodunu, eğitim loglarını ve geçmiş hatalarını bilen; tamamen yerel çalışan, kendi Python kodunla yazılmış bir laboratuvar asistanı.
>
> Donanım: Asus TUF A15, RTX 4050 (6 GB VRAM, ~4.8 GB kullanılabilir), 16 GB RAM.
> Dil kararı: **Sistem %100 İngilizce** (sorular, makaleler, kod, Obsidian notları). Türkçe yalnızca `/tr` komutuyla istek üzerine.

---

## 0. Beklentiyi Doğru Kur

3B-4B bir model genel akıl yürütmede ChatGPT / Claude / Gemini'yi geçemez. Kazanacağın yerler:

| Alan | Neden kazanırsın |
|---|---|
| Kaynağa bağlı cevap | Cevap, seçtiğin makalelerden `paper:section` atfıyla gelir |
| Senin deneylerin | Eğitim loglarını (`results.csv`) ve git değişikliklerini doğrudan okur |
| Kişiselleştirme | Senin kodun, geçmiş hataların, tarzın |
| Dar alan odağı | Sadece seçtiğin alt konudaki makaleler |
| Gizlilik ve süreklilik | Her şey yerelde, hafıza birikir |

Başarı ölçütü "daha akıllı" olmak değil, **"o alanda, senin için, kaynağı gösteren, daha isabetli cevap"** vermektir. Bunu Faz 4'teki değerlendirme setiyle ölçeceksin.

---

## 1. Temel Tasarım Kararları

- [ ] **"Ajan" değil modül:** Bir soru = en fazla 1 LLM çağrısı. Retrieval, telemetri, git bağlamı ve doğrulama saf Python.
- [ ] **Tek dil (İngilizce):** Çeviri katmanı, `query_rewrite.py` ve `Terimler.md` yok. Sorularını kısa ve teknik İngilizceyle yaz.
- [ ] **Kurtarma yolları:** (a) `/tr` komutu son cevabı Türkçe açıklar (1 ekstra çağrı, yalnızca istenince). (b) Soruda Türkçe karakter/kelime algılanırsa "Please ask in English" uyarısı ver (engelleme, sadece uyarı); İngilizce embedding Türkçe sorguda sessizce kötü sonuç verir.
- [ ] **Backend:** Birincil motor **Ollama HTTP API** (`format` ile JSON şema zorlaması, `keep_alive` kontrolü). Tüm kod `llm.generate()` arayüzünü çağırır; `llama-cpp-python` ileride isteğe bağlı eklenir.
- [ ] **Embedding ve reranker CPU'da:** VRAM'de yalnızca LLM durur; model yükleme/boşaltma dansı gerekmez.
- [ ] **Token bütçesi zorunlu:** Her bileşene sabit pay, taşma uyarısı (Ollama sınırı aşan kısmı sessizce kırpar).
- [ ] **Deterministik kanıt, LLM yorum:** Telemetri, git diff ve boyut doğrulaması kod tarafından üretilir; model bunları yorumlar.
- [ ] **Hafıza zehirlenmesine karşı:** Taslak + onay, `status` alanı, `source` etiketi, sabit şema, link allowlist, silme yok / geçersiz kılma var.
- [ ] **Alıntı önce, cevap sonra** ve kod tarafında atıf doğrulama.
- [ ] **Dar başla:** Tek alt konu, 30-50 makale.

### VRAM bütçesi (yaklaşık, mutlaka ölç)

| Bileşen | Tahmini VRAM | Not |
|---|---|---|
| 3B Q4_K_M LLM | ~2.0 GB | Ağırlıklar |
| KV cache | ~0.15 GB (4096 ctx) / ~0.3 GB (8192 ctx) | Qwen2.5-3B için kabaca ~36 KB/token; modele göre değişir |
| CUDA overhead | ~0.4 GB | |
| **LLM toplamı** | **~2.6-2.9 GB** | 3.5 GB sınırına sığar |
| Embedding + reranker | 0 GB | CPU'da çalışır |

Ölçüm: `nvidia-smi` ve `ollama ps`.

---

## 2. Önerilen Proje Yapısı

```text
DL_Lab_Assistant/
├── config.py                  # Yollar, model adları, token bütçesi, eşikler
├── llm.py                     # generate() arayüzü, Ollama HTTP backend (+ opsiyonel llama-cpp)
├── budget.py                  # Token sayacı ve bağlam paketleyici
├── vault.py                   # Obsidian okuma/yazma, frontmatter, link allowlist
├── lang_guard.py              # Türkçe soru algılama uyarısı, /tr yardımcıları
├── ingest/
│   ├── arxiv_fetch.py         # arXiv LaTeX kaynağı / PDF indirme
│   ├── paper_parser.py        # LaTeX/PDF -> bölümlere ayrılmış yapı
│   ├── chunker.py             # Makale chunk'ları + kod (AST) chunk'ları
│   ├── cards.py               # Makale kartı üretimi (offline batch)
│   ├── repo_link.py           # Makale -> resmi kod deposu eşleştirmesi
│   ├── log_parser.py          # results.csv / args.yaml / train.log otopsisi
│   └── indexer.py             # Hash kontrolü, Chroma ve BM25 indeksleme
├── retrieval/
│   ├── embed.py               # bge-base-en-v1.5 (CPU)
│   ├── search.py              # Hibrit arama + RRF (+ reverse-RAG modu)
│   ├── rerank.py              # CPU reranker
│   └── git_context.py         # git diff / status bağlamı
├── answer/
│   ├── pipeline.py            # Soru -> retrieval -> alıntı -> cevap -> doğrulama
│   ├── prompts.py             # Sistem prompt'ları
│   ├── schemas.py             # JSON şemaları
│   ├── verify.py              # Alıntı/atıf doğrulama
│   ├── shape_check.py         # Tensör dry-run (sandbox'lı)
│   └── challenge.py           # /challenge akışı
├── memory/
│   ├── errors.py              # Hata kaydı, taslak/onay akışı
│   └── profile.py             # Stil ve bilgi durumu
├── eval/
│   ├── questions.yaml
│   └── run_eval.py
├── app.py                     # CLI (Rich), sonra kendi web arayüzün
├── data/
│   ├── papers/
│   ├── repos/                 # Resmi kod depolarından alınan çekirdek dosyalar
│   └── chroma_db/
└── Vault/
    ├── 00_Profile/            # Style.md, Knowledge_State.md
    ├── 01_Paper_Cards/
    ├── 02_Theory_Notes/
    ├── 03_Experiment_Logs/
    ├── 04_Code_Reviews/
    └── Drafts/                # Onay bekleyen kayıtlar
```

---

## FAZ 0: İngilizce Kabul Testi (1-2 saat)

**Amaç:** Kod yazmadan önce modelin (ve senin İngilizce yazma rahatlığının) işe yarayıp yaramayacağını görmek.

### Adımlar
- [ ] Ollama'yı kur ve modelleri indir: `ollama pull qwen2.5-coder:3b`, ardından `ollama pull qwen3:4b` (denemek için; qwen3'te düşünme/thinking modunu kapat).
- [ ] Geçmişte yaşadığın **5 gerçek hatayı** seç: ilgili kod parçası, hata mesajı/log özeti ve gerçek çözüm.
- [ ] Her biri için soruyu **İngilizce** yaz (kısa, teknik). İlgili kodu ve kısa bir kaynak notunu prompt'a yapıştır.
- [ ] Cevabı gerçek çözümle karşılaştır ve aşağıdaki şablonu doldur.
- [ ] Aynı 5 hatayı iki modelle dene; her çağrıda aynı `num_ctx` kullan (değişirse model yeniden yüklenebilir).
- [ ] `ollama ps` ve `nvidia-smi` ile VRAM ve ısıyı not et.

### Kabul testi şablonu

| # | Soru (EN) | Gerçek neden/çözüm | Model cevabı özeti | Doğru mu? (0/1/2) | Atıf/uydurma var mı? | VRAM | Süre |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |

Puan: 0 = yanlış, 1 = kısmen doğru, 2 = işe yarar.

### Bitti kriteri
- Toplam puan en az 6/10 ise devam. Değilse: daha büyük model dene ya da asistanı "arama + telemetri + not" aracına indirge (Faz 2-5 yine değerli).
- İngilizce soru yazarken zorlandığın yerleri not et; bunlar Faz 1'deki yardımcılara (örn. `/tr`) girdi olur.

---

## FAZ 1: İskelet ve Temel Katman

### 1.1 `config.py`
- [ ] Yollar: `VAULT_DIR`, `CHROMA_DIR`, `PAPERS_DIR`, `REPOS_DIR`.
- [ ] Model: `MODEL = "qwen2.5-coder:3b"`, `NUM_CTX` (tek sabit değer: 4096 ya da ölçüp uygunsa 8192), `TEMPERATURE = 0.2`, `KEEP_ALIVE = "5m"`.
- [ ] Token bütçesi payları (NUM_CTX'e göre): sistem+stil, telemetri, git diff, kartlar, chunk'lar, hata geçmişi, soru, cevap payı. Toplamı `NUM_CTX`'i aşmasın.
- [ ] Eşikler: `top_k`, retrieval skor eşiği, telemetri eşikleri.

### 1.2 `llm.py`
- [ ] `generate(messages, schema=None, max_tokens=800) -> str` arayüzü.
- [ ] Backend 1 (varsayılan): Ollama HTTP API (`/api/chat`); `options` içinde `num_ctx`, `temperature`; `schema` verilince `format` alanına JSON şeması.
- [ ] Her çağrıda **aynı** `num_ctx` ve `keep_alive` gönder.
- [ ] `unload()`: `keep_alive: 0` ile modeli boşalt (boşta ısı için).
- [ ] Backend 2 (ileride, opsiyonel): `llama-cpp-python`. Arayüz aynı kalır.

### 1.3 `budget.py`
- [ ] Token sayımı (yaklaşık sayım yeterli; gerekirse modelin tokenizer'ı).
- [ ] `pack_context(parts, limits)`: bileşenleri payına göre kırp, taşmayı logla ve uyarı bas.
- [ ] Test: bilerek aşırı uzun girdi ver; kırpılma kontrollü mü?

### 1.4 `vault.py`
- [ ] Klasör yapısını otomatik oluştur.
- [ ] Not yaz/oku: YAML frontmatter (`date`, `source`, `status`, `tags`) + gövde.
- [ ] **Link allowlist:** Vault'taki mevcut not adlarını tara; takma ad tablosu (`overfit` -> `Overfitting`); `[[...]]` linklerini **kod** üretsin, LLM değil.
- [ ] Vault için `git init`; `/approve` sonrası commit.

### 1.5 `lang_guard.py`
- [ ] Türkçe algılama: `çğıİöşü` karakterleri ya da yaygın Türkçe kelimeler. Bulunursa uyarı ver, çalışmayı durdurma.
- [ ] `/tr` yardımcısı: son cevabı Türkçeye çeviren tek çağrı (formüller ve teknik terimler aynen kalsın).

### Bitti kriteri
- `llm.generate("Hello")` çalışıyor, JSON şemalı çağrı geçerli JSON döndürüyor, bütçe testi geçiyor, vault'a örnek not yazılıp okunuyor.

---

## FAZ 2: Eğitim Logları ve Telemetri Otopsisi (`ingest/log_parser.py`)

**Neden erken:** Ucuz, LLM istemez, RAG'e bağımlı değil, hemen değer üretir.

### Adımlar
- [ ] Girdi: bir Ultralytics/PyTorch eğitim klasörü (`results.csv`, `args.yaml`, varsa `train.log`).
- [ ] `results.csv` sütun adlarında boşluk olur: okurken `strip` yap (`pandas`).
- [ ] Çıkarılacak sinyaller (hepsi eşikli ve ayarlanabilir, hareketli ortalamayla):
  - Train loss düşerken val loss'un ardışık N epoch artması
  - `NaN`/`inf` değerleri ve ilk göründüğü epoch
  - `mAP50` / `mAP50-95` düşüşü (en iyi epoch'a göre fark)
  - Loss'ların plato yapması (öğrenmenin durması)
  - Öğrenme oranı programı ve beklenmedik sıçramalar
- [ ] `args.yaml`'dan hiperparametreler: `lr0`, `batch`, `epochs`, `imgsz`, `model` vb.
- [ ] Özel katman adı (ör. `CoordAtt`) yalnızca model YAML'ı gerçekten parse edilebiliyorsa eklensin; aksi halde uydurma.
- [ ] **Hüküm değil kanıt yaz.** Örnek çıktı:
  `[AUTO-TELEMETRY] val/box_loss rising 6 consecutive epochs (29-34) while train/box_loss falling; best mAP50=0.62 @ epoch 27, last=0.58 (-4 pts); lr0=0.01, batch=16`
- [ ] Çıktıyı `budget.py`'daki telemetri payına koy; ayrıca hata kaydının `symptom` alanını önerisi olarak doldursun (Faz 8).
- [ ] Birim testleri: elle hazırladığın 3-4 sahte `results.csv` (overfitting, NaN, sağlıklı eğitim, plato) ile doğru sinyali üretiyor mu?

### Bitti kriteri
- Gerçek bir eğitim klasöründe sinyaller makul çıkıyor; sağlıklı bir eğitimde yanlış alarm vermiyor.

---

## FAZ 3: Makale İşleme Hattı

**Amaç:** Makaleleri temiz, bölüm yapısı korunmuş, formülleri bozulmamış veriye çevirmek.

### 3.1 Seçim
- [ ] Tek alt konu seç (örn. attention mekanizmaları, residual ağlar, normalizasyon, YOLO ailesi).
- [ ] 30-50 makalelik `papers.csv`: `arxiv_id, title, year, notes, repo_url` (repo_url boş bırakılabilir, Faz 11'de dolacak).

### 3.2 `arxiv_fetch.py`
- [ ] Her ID için önce **LaTeX kaynağını** (e-print) indir; olmazsa PDF'e düş.
- [ ] arXiv'e nazik davran: istekler arasında birkaç saniye bekle, yerelde önbellekle.
- [ ] Çıktıyı `data/papers/<arxiv_id>/` altına koy.

### 3.3 `paper_parser.py`
- [ ] LaTeX varsa: `\section`/`\subsection` yapısı; formülleri (`$...$`, `equation`, `align`) aynen koru; bibliyografiyi ayrı tut.
- [ ] PDF fallback: `pymupdf4llm` ile Markdown'a çevir; formül bozulması çoksa ağır araçları (örn. Marker) yalnızca gece toplu çalıştır.
- [ ] Çıktı: `{title, authors, year, arxiv_id, sections: [{name, text}], figure_captions, tables}`.
- [ ] Parse kalitesini gözle kontrol et: makale başına bölüm ve formül sayısını logla.

### 3.4 `chunker.py`
- [ ] **Bölüm bazlı + parent-child:** Küçük chunk'lar (~200-300 token) arama içindir; cevap için bağlı oldukları bölümden gerekli kısım çekilir.
- [ ] Her chunk'a başlık ekle: `[Title, year, section]`; bölüm adı **metadata**'da da dursun (Faz 12 `/challenge` buna ihtiyaç duyar).
- [ ] Formül bloklarını ortadan bölme.
- [ ] Kod için: `ast` ile `class`/`def` bazlı; büyük sınıflarda method bazında; chunk'a sınıf adı ve import'lar; `lineno`/`end_lineno` metadata. `.ipynb` için JSON'dan kod hücrelerini çıkar.

### 3.5 `indexer.py`
- [ ] Dosya hash'i (SHA256); değişmeyeni atla.
- [ ] Dosya değişince eski chunk'ları sil (`arxiv_id`/`filepath` metadata), yenilerini ekle.
- [ ] Koleksiyonlar: `paper_chunks`, `paper_cards`, `code`, `errors`, `experiments`.

### Bitti kriteri
- 5 makale uçtan uca işlendi; bölümler ve formüller doğru; tekrar çalıştırınca değişmeyenler atlanıyor.

---

## FAZ 4: Değerlendirme Seti (Retrieval'dan Önce Yaz)

**Amaç:** "Daha iyi" iddiasını ölçmek ve her değişikliği kıyaslamak.

### Adımlar
- [ ] `eval/questions.yaml`: ilk 15-20, sonra 30-50 soru. Alanlar: `question` (EN), `expected_source` (makale/bölüm), `key_points`, `type`.
- [ ] Soru türleri:
  - conceptual (kavram)
  - paper-detail (makale detayı)
  - my-error (kendi geçmiş hatan)
  - telemetry (bir `results.csv` + soru)
  - equation-to-code (Faz 11 sonrası)
  - hypothesis (Faz 12 için; **çalıştığı bilinen** hipotezler de olsun)
- [ ] Metrikler: `recall@5`, `MRR`, atıf doğruluğu (doğrulanan alıntı oranı), cevap puanı (1-5, sen ver ya da `key_points` sayımı).
- [ ] Aynı soruları ChatGPT / Claude / Gemini'ye de sor; hangi türde kazandığını/kaybettiğini kaydet.
- [ ] `run_eval.py` ile her büyük değişiklikten sonra çalıştır; sonuçları tarihli kaydet.

### Bitti kriteri
- Temel skor tablon var; her değişiklikte kıyaslayabiliyorsun.

---

## FAZ 5: Retrieval

### 5.1 `embed.py`
- [ ] Varsayılan: `BAAI/bge-base-en-v1.5` (CPU). `bge-small-en-v1.5` ile kıyasla; eval setinde hangisi daha iyi ise onu seç.
- [ ] Sorgu tarafında bge önekini kullan: `Represent this sentence for searching relevant passages: `. Dokümanlarda önek kullanma.
- [ ] Toplu işlemde küçük batch.

### 5.2 `search.py`
- [ ] **Hibrit arama:** vektör arama (Chroma) + BM25 (`rank_bm25`); model adları ve terimler için BM25 şart.
- [ ] **RRF** ile birleştir; ilk 20-30 sonucu al.
- [ ] Koleksiyonlar arası arama: önce `paper_cards`, sonra `paper_chunks`, gerekirse `code`, `errors`, `experiments`.

### 5.3 `rerank.py`
- [ ] CPU reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` ile başla (çok hafif); `BAAI/bge-reranker-base` ile kıyasla. (Not: bge-small/base embedding modelleridir, reranker yerine geçmez.)
- [ ] İlk 20'yi yeniden sırala, ilk 3-5'i tut.
- [ ] Skor eşiğinin altı = "yeterli kaynak yok" sinyali (Faz 7'de çekimser kalmak için).

### 5.4 Paketleme
- [ ] Önce kartlar, sonra gerekirse chunk'lar; parent-child ile bölümün ilgili kısmı.
- [ ] `budget.pack_context` ile paylara göre kırp.

### Bitti kriteri
- Eval setinde `recall@5` en az %70 (sonra yükselt).

---

## FAZ 6: Makale Kartları (Offline Toplu İşlem)

- [ ] Kart şeması: `problem`, `method`, `key_equations`, `datasets`, `results`, `limitations`, `summary`, `keywords`.
- [ ] İlgili bölümleri (Abstract, Method, Experiments, Conclusion) kısa parçalarla ver; **JSON şemalı** çıktı al.
- [ ] Formülleri model yazmasın; kod, parser çıktısından **kopyalasın**.
- [ ] Kartı `Vault/01_Paper_Cards/<arxiv_id>.md` olarak kaydet (`status: draft`).
- [ ] Sen gözden geçir; doğruysa `status: verified`. Retrieval öncelikle `verified` kartları kullanır.
- [ ] Toplu işi gece/şarjdayken, aralara soğuma payı bırakarak çalıştır.
- [ ] Sayısal sonuçları (mAP, doğruluk) ve formülleri mutlaka kaynakla karşılaştır; 3B model hata yapabilir.

### Bitti kriteri
- 10 makale kartı üretildi, en az 5'i doğrulandı.

---

## FAZ 7: Cevap Hattı

- [ ] `pipeline.py`: soru -> dil kontrolü -> retrieval -> (opsiyonel telemetri/git bağlamı) -> `pack_context` -> LLM -> doğrulama.
- [ ] **Alıntı önce:** Şema: `{ "quotes": [{"source": "...", "text": "..."}], "answer": "..." }`. Prompt: "Write the answer only from the quotes."
- [ ] `verify.py`: Her alıntı chunk'ta gerçekten geçiyor mu (normalize edilmiş karşılaştırma). Kod atfı için `file:line` gerçekten var mı. Doğrulanamayanı işaretle ya da at.
- [ ] **Çekimser kalma:** Retrieval skoru eşik altındaysa "No sufficient source found" de; genel bilgiyse açıkça belirt.
- [ ] Cevap formatı: kısa özet, gerekçe (formül/tensör boyutu), kaynak listesi.
- [ ] Sistem prompt'u: kısa, İngilizce, "cite sources; say when unsure".
- [ ] `/tr` ile son cevabı Türkçeye çevirebilme.

### Bitti kriteri
- Atıflar doğrulanıyor; kaynak yokken sistem uydurmak yerine çekimser kalıyor; eval skorları Faz 5'e göre kötüleşmiyor.

---

## FAZ 8: Hafıza ve Kişiselleştirme

### 8.1 Hata kaydı şeması
```json
{
  "date": "2026-10-22",
  "symptom": "val loss rising while train loss falling after epoch 30",
  "likely_cause": "dropout placed before attention output",
  "fix": "move dropout after the attention projection",
  "evidence": ["model.py:45-90", "AUTO-TELEMETRY epoch 29-34"],
  "source": "user | model | telemetry",
  "status": "suggestion | tried | verified | invalid"
}
```

### 8.2 Akış (`memory/errors.py`)
- [ ] `/save`: LLM yalnızca şemayı doldurur (tek çağrı); yazma, tarih, link işi kodda.
- [ ] Kayıt önce `Vault/Drafts/` altına düşer: `source: model`, `status: suggestion`.
- [ ] `/approve`: diff gösterilir; onaylarsan asıl günlüğe geçer ve commit atılır.
- [ ] `/verify <id>`: çözüm gerçekten işe yaradıysa `status: verified`.
- [ ] Yanlış çıkan kayıt silinmez, `status: invalid` olur; retrieval atlar.
- [ ] Kanıtı (`file:line` ya da telemetri) olmayan kayıt `suggestion` kalır.

### 8.3 Retrieval tarafı
- [ ] Her kaydı `errors` koleksiyonuna göm; soruya benzer kayıtları çek (günlüğü komple okuma).
- [ ] `verified` olanlara öncelik ver; `suggestion` olanları prompt'a "unverified" etiketiyle koy.
- [ ] `source: model` kayıtlar kanıt sayılmaz (modelin kendi çıktısını geri beslememesi için).

### 8.4 Kişiselleştirme
- [ ] `00_Profile/Style.md`: öğrenme tarzın (örn. "give equations and tensor shapes, minimal prose").
- [ ] `00_Profile/Knowledge_State.md`: iyi bildiğin ve takıldığın kavramlar; cevap derinliği buna göre ayarlanır. Güncellemeyi sen onaylarsın.

### Bitti kriteri
- Bir hata kaydet -> taslağa düştü -> onayla -> doğrula akışı çalışıyor; benzer soruda doğrulanmış kayıt çekiliyor.

---

## FAZ 9: CLI, Git Bağlamı ve Isı Yönetimi

### 9.1 `app.py` (Rich CLI)
- [ ] Komutlar: `/save`, `/approve`, `/verify <id>`, `/tr`, `/diff`, `/logs <dir>`, `/challenge <hypothesis>` (Faz 12), `/sources`, `/clear`, `/exit`.
- [ ] LLM'li hafıza işlemleri yalnızca komutla çalışsın (konuşma ortasında değil).
- [ ] Boşta kalınca (örn. 5 dk) `unload()`.
- [ ] İsteğe bağlı durum satırı: VRAM/sıcaklık (`pynvml`).

### 9.2 `retrieval/git_context.py`
- [ ] **`/diff` bayrağıyla** (ya da net anahtar kelimelerle) çalışsın; sorunun kodla ilgili olduğunu tahmin etmeye çalışma.
- [ ] Önce `git diff --stat`, sonra yalnızca ilgili dosyaların hunk'ları (`git diff HEAD -U3`).
- [ ] `.ipynb`, veri, çıktı, `runs/` gibi dosyaları hariç tut.
- [ ] Diff payı `budget.py`'da sabit; taşarsa kırp ve "diff truncated" notu ekle.
- [ ] Kod/diff içeren sorularda bağlam yetmiyorsa `NUM_CTX`'i 8192'ye çıkarmayı dene (KV cache maliyeti küçük; ön işleme süresi artar). Ölç ve karar ver.

### 9.3 Telemetri bağlantısı
- [ ] `/logs <dir>` Faz 2'nin çıktısını bağlam paketine ekler.

### Bitti kriteri
- Günlük kullanımda sorunsuz; "Son değişiklikten sonra gradyanlar neden sıfırlandı?" sorusu `/diff` ve `/logs` ile cevap alıyor.

---

## FAZ 10: Tensör Dry-Run Doğrulaması (`answer/shape_check.py`)

**Önemli:** Bu, LLM'in ürettiği kodu **çalıştırmak** demektir. Sandbox zorunludur.

### Adımlar
- [ ] Model çıktısına şema ile şunları iste: `module_code`, `init_args`, `example_input_shape`, `expected_output_shape`.
- [ ] **AST ön kontrolü:** Yalnızca `torch`, `torch.nn`, `torch.nn.functional`, `math`, `einops` import'larına izin ver; `os`, `sys`, `subprocess`, `open`, `eval`, `exec`, `__import__`, ağ erişimi yasak. Kural ihlalinde çalıştırma.
- [ ] **İzole çalıştırma:** Ayrı alt süreç, zaman aşımı, bellek sınırı, `CUDA_VISIBLE_DEVICES=""` (yalnızca CPU), dosya yazma yok.
- [ ] `import torch` yavaş olduğu için **kalıcı bir çalışan süreç** kullan (her kontrolde yeniden başlatma yok).
- [ ] Kontroller: modül kurulur mu, `forward` çalışır mı, çıktı boyutu beklenenle eşleşir mi, **backward** çalışır mı, gradyanlar `None`/NaN mı, parametre sayısı.
- [ ] Hata olursa cevabı **hatayla birlikte göster** (`RuntimeError: shape mismatch ...` gibi); gizleme. İstersen yalnızca hata durumunda en fazla 1 otomatik yeniden deneme (modele hata mesajını ver).
- [ ] Çıktıda açıkça yaz: "Shape check passed: this verifies tensor shapes and gradient flow, not that the math is correct."
- [ ] Test seti: biliniz doğru bir attention bloğu, kasıtlı boyut hatalı bir blok, sonsuz döngü, yasaklı import içeren bir kod.

### Bitti kriteri
- Doğru blok geçiyor, hatalı blok net mesajla yakalanıyor, kötü niyetli/yasaklı kod çalıştırılmadan reddediliyor.

---

## FAZ 11: Makale + Resmi Kod Eşleştirmesi (`ingest/repo_link.py`)

**Not:** Papers with Code Temmuz 2025'te kapandı ve eski API canlı değil; ona bağımlı kalma.

### Adımlar
- [ ] Repo URL kaynakları (güvenilirlik sırasıyla):
  1. `papers.csv` içindeki `repo_url` (elle girilen; 30-50 makale için en sağlamı)
  2. Makalenin LaTeX/abstract metninde regex ile yakalanan `github.com/...` linkleri
  3. Hugging Face Papers sayfasındaki bağlı kod linki
  4. Eski Papers with Code verisinin GitHub arşivi / Hugging Face yansısı (tek seferlik arama tablosu; eski veri, Temmuz 2025'e kadar)
- [ ] Depoyu tamamen alma; yalnızca çekirdek dosyalar (`model.py`, `modules.py`, `loss.py` gibi). **Commit hash'ini kaydet.**
- [ ] Lisans ve atıf bilgisini metadata'ya yaz.
- [ ] Alınan kodu `code` koleksiyonuna `arxiv_id` etiketiyle ekle (AST chunker ile).
- [ ] Retrieval: "Equation N" sorularında hem LaTeX formül chunk'ı hem aynı `arxiv_id`'li `class` chunk'ı birlikte gelsin.
- [ ] Formül ile kodu eşleştirmeyi model yapar; bu zor, eval setine `equation-to-code` soruları ekle ve ölç.

### Bitti kriteri
- En az 5 makale için resmi kod bağlandı ve "Equation N nasıl kodlanmış?" sorusu iki kaynağı birden getiriyor.

---

## FAZ 12: `/challenge` Modu (Hipotez Çürütme)

**Amaç:** Modelin (sycophancy) sana katılmasını değil, fikri kanıtla zorlamasını sağlamak.

### Adımlar
- [ ] `/challenge <hypothesis>` komutu `answer/challenge.py`'ı çalıştırır.
- [ ] **Reverse-RAG:** Hipotezin bileşenlerinden sorgular üret (ör. "ViT block", "small object detection", "6GB VRAM") ve aramayı yalnızca şu kaynaklarla sınırla: makalelerin `Limitations`, `Ablation`, `Discussion` bölümleri (chunk metadata'sı ile) ve senin `status: invalid` / başarısız deney kayıtların.
- [ ] **Hesaplanan maliyet (kod):** Hipotezdeki modül/katman elde varsa Faz 10'daki sandbox ile CPU'da kurup parametre sayısını ve yaklaşık bellek maliyetini hesapla; "6 GB'a sığmaz" gibi cümleler hesaba dayansın.
- [ ] Çıktı formatı sabit:
  1. Evidence against (her madde bir chunk'a atıflı)
  2. Computed cost (parametre, yaklaşık bellek)
  3. What experiment would confirm or refute it
- [ ] Atıfsız itiraz yasak: kanıt bulunamazsa "No evidence against found in indexed sources" yazsın; itiraz uydurmasın.
- [ ] Eval setine çalıştığı bilinen hipotezler ekle; model her şeyi çürütüyorsa fark edersin.

### Bitti kriteri
- Bilinen kötü bir fikirde atıflı itirazlar, bilinen iyi bir fikirde dengeli bir cevap üretiyor.

---

## FAZ 13: Genişleme (Skorlar Darboğazı Gösterirse)

- [ ] Korpusu büyüt (100+ makale); her genişlemeden sonra `run_eval.py`.
- [ ] Atıf grafiği: makaleler arası referanslar metadata olarak (Semantic Scholar gibi kaynaklardan).
- [ ] Daha iyi model: `llm.generate` arayüzü sayesinde 4B-7B dene; VRAM sınırına dikkat.
- [ ] `llama-cpp-python` backend'i: gramer/logprob gibi ince kontroller gerekiyorsa.
- [ ] QLoRA ince ayar: bilgi eklemek için değil, **üslup/çıktı formatı** için; ancak RAG olgunlaştıktan sonra.
- [ ] Kendi arayüzün: önce çekirdek (`pipeline.py`) arayüzden bağımsız kalsın; sonra hafif bir web arayüzü (FastAPI + basit HTML ya da Streamlit).

---

## Çalışma Sırası (Özet)

| Sıra | Faz | Ana çıktı |
|---|---|---|
| 1 | Faz 0 | Kabul testi, model seçimi |
| 2 | Faz 1 | İskelet, `llm.py`, `budget.py`, `vault.py` |
| 3 | Faz 2 | `log_parser.py` (hemen değer) |
| 4 | Faz 3 | 5 makale uçtan uca |
| 5 | Faz 4 | İlk 15-20 soruluk eval seti |
| 6 | Faz 5 | Hibrit retrieval + reranker |
| 7 | Faz 6 | Makale kartları |
| 8 | Faz 7 | Cevap hattı |
| 9 | Faz 8 | Hafıza, taslak/onay |
| 10 | Faz 9 | CLI, `/diff`, `/logs`, ısı yönetimi |
| 11 | Faz 10 | Tensör dry-run (sandbox) |
| 12 | Faz 11 | Resmi kod eşleştirmesi |
| 13 | Faz 12 | `/challenge` |
| 14 | Faz 13 | Genişleme |

Her fazı bitirmeden sonrakine geçme; "bitti kriteri" sağlanınca ilerle. Her büyük değişiklikten sonra `run_eval.py` çalıştır.

---

## Riskler ve Önlemler

| Risk | Önlem |
|---|---|
| Bağlam sessizce kırpılır | `budget.py`, taşma uyarısı |
| Türkçe soru İngilizce embedding'de kötü sonuç verir | `lang_guard.py` uyarısı, `/tr` |
| İngilizce yazma zorluğu | Kısa teknik sorular, `/tr`, Faz 0'da erken tespit |
| 3B model uydurur | Alıntı önce, `verify.py`, çekimser kalma |
| Telemetri yanlış alarm verir | Eşikli ve ayarlanabilir kurallar, hareketli ortalama, hüküm yerine kanıt |
| Üretilen kodun çalıştırılması | AST whitelist, alt süreç, zaman aşımı, CPU-only, dosya/ağ yasağı |
| Dry-run "doğru" demek zanneder | Çıktıda "yalnızca boyut/gradyan kontrolü" notu |
| Repo kodu makaleden farklı | Commit hash kaydı, eval'de equation-to-code soruları |
| Hafıza zehirlenmesi | Taslak/onay, `status`, `source`, sabit şema, geçersiz kılma |
| Tutarsız `[[link]]` | Allowlist + takma ad tablosu, linki kod üretir |
| PDF formülleri bozulur | LaTeX kaynağı öncelikli, bölüm bazlı kontrol |
| Hayalet chunk'lar | Hash değişince eski chunk'ları sil |
| Ollama `num_ctx` değişince yeniden yükler | Tek sabit `NUM_CTX`, her çağrıda aynı seçenekler |
| Laptop ısınır | Embedding/reranker CPU'da, toplu işler geceye, boşta `unload()` |
| `/challenge` itiraz uydurur | Atıf zorunlu, kanıt yoksa "bulunamadı" |
| "Daha iyi" iddiası ölçülmez | Faz 4 eval seti, düzenli çalıştırma |

---

## Ek: `requirements.txt` Taslağı

```text
requests              # Ollama HTTP API
chromadb
sentence-transformers # bge-base-en-v1.5, cross-encoder reranker (CPU)
rank-bm25
pymupdf4llm
pandas
pyyaml
rich
torch                 # Faz 10 dry-run (CPU çalıştırılacak)
pynvml                # isteğe bağlı, GPU izleme
# llama-cpp-python    # isteğe bağlı, ileride
```

> Not: Sürümleri ve kurulum bayraklarını kurarken resmi dokümantasyondan kontrol et; hızlı değişen kütüphaneler.
