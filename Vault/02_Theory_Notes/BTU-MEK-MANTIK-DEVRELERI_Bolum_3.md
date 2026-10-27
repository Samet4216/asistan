# Sayısal Mantık Devreleri ve Sayısal Sistem Tasarımı - Kombinezonsal Mantık Devreleri: Toplayıcılar, MUX, DEMUX, Kod Çözücüler

**Kitap/Ders:** Sayısal Mantık Devreleri ve Sayısal Sistem Tasarımı
**Yazarlar:** M. Morris Mano & Michael D. Ciletti, Thomas L. Floyd

### Kombinezonsal (Bileşimsel) Mantık Devreleri

Çıkışların sadece o anki giriş durumlarına bağlı olduğu, bellek içermeyen devrelerdir.

**1. İkili Toplayıcılar:**
- **Yarım Toplayıcı (Half Adder):** İki 1-bitlik girişi ($A, B$) toplar:
  $$\text{Toplam (Sum)} = A \oplus B, \quad \text{Elde (Carry)} = A \cdot B$$
- **Tam Toplayıcı (Full Adder):** Önceki basamaktan gelen eldeyi ($C_{in}$) de hesaba katar:
  $$S = A \oplus B \oplus C_{in}$$
  $$C_{out} = AB + C_{in}(A \oplus B)$$
- Hızlı toplama için Ripple Carry Adder (zincirleme gecikmeli) yerine Carry Lookahead Adder (CLA - İleri Taşımalı Elde) kullanılır: Üretim $G_i = A_i B_i$ ve yayılım $P_i = A_i \oplus B_i$ bağıntılarıyla elde bitleri kapı gecikmesi beklemeden paralel türetilir.

**2. Multiplexer (Veri Seçici - MUX):**
$2^n$ veri girişinden birini $n$ adet seçme ucuyla tekil çıkışa aktarır. Bir $2^n \times 1$ MUX, herhangi bir $n$ değişkenli Boolean fonksiyonunu harici kapı kullanmadan doğrudan gerçekleştirebilir.

**3. Demultiplexer (DEMUX) ve Kod Çözücü (Decoder):**
$n$-bitlik ikili kodu $2^n$ adet tekil aktif çıkışa dönüştürür (örneğin 3-to-8 decoder). Aktif düşük (active-low) yetkilendirme (Enable) girişleri ile kaskat bağlanarak bellek adresleme blokları oluşturulur.
