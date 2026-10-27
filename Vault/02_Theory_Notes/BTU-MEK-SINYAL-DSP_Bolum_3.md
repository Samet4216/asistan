# Sinyaller, Sistemler ve Sayısal Sinyal İşleme (DSP) - Sayısal Filtre Tasarımı: FIR ve IIR Filtreler

**Kitap/Ders:** Sinyaller, Sistemler ve Sayısal Sinyal İşleme (DSP)
**Yazarlar:** Alan V. Oppenheim & Alan S. Willsky

### Sayısal Filtre Mimarileri: FIR vs. IIR

**1. Sonlu Darbe Cevaplı (FIR - Finite Impulse Response) Filtreler:**
$$y[n] = \sum_{k=0}^{M} b_k x[n - k]$$
- Yalnızca mevcut ve geçmiş girişlere bağlıdır, geri besleme yoktur (kutupları yalnızca $z=0$'dadır).
- **Avantajları:** Her zaman koşulsuz kararlıdır. Simetrik katsayılar seçilerek kusursuz **Doğrusal Faz (Linear Phase)** elde edilebilir (tüm frekans bileşenleri eşit süre gecikir, sinyal şekli bozulmaz).
- **Dezavantajları:** Keskin frekans geçiş bandı elde etmek için yüksek filtre derecesi ($M > 50-100$) gerekir, bu da gömülü işlemcide hesaplama yükünü artırır.

**2. Sonsuz Darbe Cevaplı (IIR - Infinite Impulse Response) Filtreler:**
$$y[n] = \frac{1}{a_0} \left( \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k] \right)$$
- Geçmiş çıkışların geri beslemesini içerir ($z$-domeninde sıfırlar ve kutuplar vardır).
- **Avantajları:** Çok daha düşük dereceyle ($N = 2$ ila $6$) çok keskin kesim frekansları sağlar, işlemci yükü azdır.
- **Dezavantajları:** Kutuplar birim çemberin dışına çıkarsa kararsızlaşabilir. Faz cevabı doğrusal değildir.
