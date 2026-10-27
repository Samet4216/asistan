# Sayısal Mantık Devreleri ve Sayısal Sistem Tasarımı - Karnaugh Haritaları (K-Map) ve Mantıksal Fonksiyon İndirgeme

**Kitap/Ders:** Sayısal Mantık Devreleri ve Sayısal Sistem Tasarımı
**Yazarlar:** M. Morris Mano & Michael D. Ciletti, Thomas L. Floyd

### Karnaugh Haritaları (K-Map) ile Sadeleştirme

Doğruluk tablosundaki çıktıyı en az sayıda kapı ve giriş ile gerçekleştirmek için grafiksel K-Map yöntemi kullanılır. K-Map hücreleri Gray Kodu sırasıyla dizilir ($00, 01, 11, 10$); böylece komşu hücreler arasında yalnızca tek 1 bit değişir.

**K-Map Gruplama Kuralları:**
1. Gruplar daima $2^n$ ($1, 2, 4, 8, 16$) adet komşu '1' hücresinden oluşmalıdır.
2. Gruplar mümkün olduğunca büyük seçilmelidir (büyük grup daha fazla değişkeni eler).
3. Haritanın karşıt kenarları ve dört köşesi bitişik kabul edilir (küresel topoloji / toroidal wrap-around).
4. **Fark Etmez Durumları (Don't Care - $d$ veya $X$):** Pratikte ortaya çıkmayacak veya çıktısı önemsiz durumlar, eğer bir grubu $2^n$ büyüklüğe tamamlayıp fonksiyonu basitleştiriyorsa '1' kabul edilir; aksi halde '0' sayılıp dışarıda bırakılır.

**Çarpımlar Toplamı (SOP - Sum of Products) ve Toplamlar Çarpımı (POS - Product of Sums):**
- **Minterm ($\Sigma m$):** Çıktıyı '1' yapan giriş kombinasyonlarının AND çarpımları.
- **Maxterm ($\Pi M$):** Çıktıyı '0' yapan giriş kombinasyonlarının OR toplamları.
- Asal Kapsayıcı (Prime Implicant) ve Zorunlu Asal Kapsayıcı (Essential Prime Implicant) bulunarak minimum donanımsal mantık devresi elde edilir.
