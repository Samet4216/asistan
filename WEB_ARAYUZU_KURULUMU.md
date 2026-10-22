# 🌐 Local Deep Learning Lab Assistant — Modern Web UI Kurulum Kılavuzu

Bu belge, siyah terminal ekranı yerine **ChatGPT / Ollama Web UI** benzeri, ferah satır aralıklı, formülleri matematiksel olarak render eden, kod bloklarını renklendiren ve kopyalama butonu sunan modern bir arayüz kurma rehberidir.

---

## 🎯 Hedef Arayüz Özellikleri
- **Ferah Tipografi:** Geniş satır aralıkları, net başlıklar ve göz yormayan modern karanlık tema (Dark Mode).
- **Kusursuz Matematik (KaTeX/LaTeX):** $z_c^h(h), \sum, \times$ gibi denklemlerin bozulmadan, basılı ders kitabı kalitesinde görünmesi.
- **Renkli Kod Blokları:** Python ve PyTorch kodlarının sözdizimi renklendirmesi (Syntax Highlighting) ve tek tıkla `Copy` butonu.
- **Tam Yerel Mimari:** RTX 4050 GPU'yu yormayan, CPU üzerinde hibrit arama (ChromaDB + BM25) yapan ve Obsidian hafızasıyla konuşan motor.

---

## 🚀 YÖNTEM 1: Projeye Özel Entegre Web Arayüzü (Önerilen - En Kolay ve Hızlı)

Bu yöntem mevcut Python ortamımızı kullanır. Ekstra hiçbir Docker veya karmaşık sunucu kurulumu gerektirmez.

### 1. Adım: Arayüz Paketini Yükleme
Terminalinde proje klasöründeyken (`C:\Users\24406601057\Desktop\Asistan`) şu komutu çalıştır:

```powershell
.\.venv\Scripts\pip.exe install streamlit
```

### 2. Adım: Web Arayüzünü Başlatma
Projenin içine eklediğimiz `web_app.py` dosyasını şu komutla çalıştır:

```powershell
.\.venv\Scripts\streamlit.exe run web_app.py
```

Tarayıcında otomatik olarak `http://localhost:8501` adresi açılacaktır. 
Burada:
- Sol kenar çubuğunda anlık **RTX 4050 VRAM ve Sıcaklık Durumu** görünür.
- `/explain`, `/challenge`, `/logs` ve `/diff` butonları yer alır.
- Yazdığın tüm sorular ChatGPT tarzında ferah baloncuklar, renkli kodlar ve temiz matematiksel formüllerle dökülür.

---

## 🐳 YÖNTEM 2: Resmi Open WebUI (Fotoğraftaki Birebir Arayüz)

Fotoğrafta paylaştığın sol menülü arayüzün adı **Open WebUI**'dır. Doğrudan yerel Ollama motoruna bağlanır.

### Kurulum Adımları:

1. **Python ile Kurulum:**
   ```powershell
   pip install open-webui
   open-webui serve
   ```

2. **Veya Docker ile Kurulum (Eğer Docker Desktop yüklüyse):**
   ```powershell
   docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
   ```

3. Tarayıcından `http://localhost:8080` (veya port 8080) adresine gir.
4. Model seçme kutusundan `qwen2.5:7b-instruct-q4_K_M` modelini seç.
5. ChatGPT gibi sol tarafta sohbet geçmişi tutarak kullanmaya başla!

---

## ⚖️ İki Yöntemin Karşılaştırması

| Özellik | Yöntem 1 (`web_app.py` - Streamlit) | Yöntem 2 (Open WebUI) |
| :--- | :---: | :---: |
| **Kurulum Zorluğu** | Çok Kolay (Tek `pip install`) | Orta (Docker veya pip) |
| **Lab Asistanı RAG Sistemi** | ✅ Tam Entegre (ChromaDB + Makaleler + Kodlar) | ⚠️ Sadece Saf LLM (Özel pipeline entegrasyonu gerekir) |
| **Obsidian Bellek & Taslak Akışı** | ✅ Var (`/save`, `/approve`) | ❌ Yok |
| **RTX 4050 Canlı Isı/VRAM Takibi** | ✅ Sol panelde canlı gösterge | ❌ Yok |
| **Görünüm** | Modern ChatGPT stili | Birebir ChatGPT klonu |

> 💡 **Tavsiye:** Bizim kurduğumuz makale hafızasını, %100 alıntı doğrulamasını ve Obsidian sistemini doğrudan tarayıcında ChatGPT şıklığında görmek için **Yöntem 1 (`web_app.py`)** biçilmiş kaftandır.
