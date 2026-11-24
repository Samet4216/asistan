# AI Coding Agent Görev Tanımı

## Kişisel Uzman Yapay Zekâ Asistanı — Uygulama ve Geliştirme Spesifikasyonu

**Belge amacı:** Bu dosya, projenin adım adım tasarlanması, kodlanması, test edilmesi ve belgelendirilmesi için temel şartnamedir.

**Proje sahibi:** Samet\
**Proje türü:** Yerel çalışan, uzmanlaşabilir, çok ajanlı mühendislik asistanı ve model araştırma laboratuvarı\
**Hedef yerel donanım:** NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM; 16 GB RAM\
**Eğitim yaklaşımı:** Yerel küçük deneyler + gerektiğinde bulut GPU eğitimi\
**Ana geliştirme dili:** Python\
**Açıklama dili:** Türkçe\
**Kod, değişken, fonksiyon, sınıf, dosya ve Git commit dili:** İngilizce

---

### Uzmanlık Alanları
1. **Deep Learning / Neural Networks:** ANN, CNN, Transformer, optimizasyon, backpropagation, model eğitimi, mimari araştırması.
2. **Computer Vision / UAV:** Nesne tespiti, nesne takibi, veri kümeleri, görüntü işleme, gerçek zamanlı çıkarım, edge deployment, İHA sistemleri.
3. **Mathematics / Engineering:** Lineer cebir, kalkülüs, diferansiyel denklemler, olasılık, istatistik, sayısal yöntemler, fizik ve mühendislik problemleri.
4. **Software Engineering:** Python, C/C++, algoritmalar, debugging, test, yazılım mimarisi, otomasyon ve araç geliştirme.

---

### Temel Prensipler
- **Önce çalışan küçük sistem, sonra karmaşık sistem.**
- **Başarıyı iddiayla değil testle göster.** (Pytest ve benchmark).
- **RAG bilgi erişimidir; fine-tuning model davranışını değiştirir.**
- **Ollama inference aracıdır, eğitim aracı değildir.**
- **Donanım sınırlarını dikkate al:** 6 GB VRAM ve 16 GB RAM sınırlarını zorlamadan, OOM önlemleriyle çalış.
- **Güvenlik ve gizlilik:** API anahtarları `.env` içinde tutulacak, kullanıcı onaysız dış servise veri gönderilmeyecek.
