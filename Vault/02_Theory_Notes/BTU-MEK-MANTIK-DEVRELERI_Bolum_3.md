# Sayısal Mantık Devreleri ve Sayısal Sistem Tasarımı - Kombinezonsal Mantık Devreleri: Toplayıcılar, MUX, DEMUX, Kod Çözücüler

**Kitap/Ders:** Sayısal Mantık Devreleri ve Sayısal Sistem Tasarımı
**Yazarlar:** M. Morris Mano & Michael D. Ciletti, Thomas L. Floyd

### Kombinezonsal (Bileşimsel) Mantık Devreleri

Çıkışların sadece o anki giriş durumlarına bağlı olduğu, geçmiş durumları saklayan bellek (flip-flop/latch) içermeyen sayısal devrelerdir.

**1. İkili Toplayıcılar (Binary Adders):**
- **Yarım Toplayıcı (Half Adder):** İki 1-bitlik girişi ($A, B$) toplar:
  $$\text{Toplam (Sum)} = A \oplus B, \quad \text{Elde (Carry)} = A \cdot B$$
- **Tam Toplayıcı (Full Adder):** Önceki basamaktan gelen eldeyi ($C_{in}$) de hesaba katar:
  $$S = A \oplus B \oplus C_{in}$$
  $$C_{out} = AB + C_{in}(A \oplus B) = AB + BC_{in} + AC_{in}$$
- **Hızlı Toplayıcılar:** Ripple Carry Adder (zincirleme gecikmeli) yerine Carry Lookahead Adder (CLA - İleri Taşımalı Elde) kullanılır: Üretim $G_i = A_i B_i$ ve yayılım $P_i = A_i \oplus B_i$ bağıntılarıyla elde bitleri kapı gecikmesi beklemeden paralel türetilir.

**2. Multiplexer (Veri Seçici / Çoklayıcı - MUX):**
Multiplexer (MUX), $2^n$ adet veri girişinden ($I_0, I_1, \dots, I_{2^n-1}$) yalnızca birini, $n$ adet kontrol/seçme hattı ($S_0, S_1, \dots, S_{n-1}$) aracılığıyla seçerek tek bir ortak çıkış hattına ($Y$) yönlendiren kombinezonsal mantık devresidir. Sayısal anahtar (rotary switch) görevi görür.

**2:1 MUX Mimarisi:**
- Girişler: $I_0, I_1$, Seçme ucu: $S$, Çıkış: $Y$.
- Boolean İfadesi: $$Y = \overline{S} \cdot I_0 + S \cdot I_1$$
- Doğruluk Tablosu:
  | $S$ | Çıkış ($Y$) |
  | :---: | :---: |
  | 0 | $I_0$ |
  | 1 | $I_1$ |

**4:1 MUX Mimarisi:**
- Girişler: $I_0, I_1, I_2, I_3$, Seçme uçları: $S_1, S_0$, Çıkış: $Y$.
- Boolean İfadesi:
  $$Y = \overline{S_1}\,\overline{S_0}\,I_0 + \overline{S_1}\,S_0\,I_1 + S_1\,\overline{S_0}\,I_2 + S_1\,S_0\,I_3$$
- Doğruluk Tablosu:
  | $S_1$ | $S_0$ | Aktif Giriş | Çıkış ($Y$) |
  | :---: | :---: | :---: | :---: |
  | 0 | 0 | $I_0$ | $I_0$ |
  | 0 | 1 | $I_1$ | $I_1$ |
  | 1 | 0 | $I_2$ | $I_2$ |
  | 1 | 1 | $I_3$ | $I_3$ |

**İç Kapı Yapısı (Gate-Level Implementation):**
Her veri girişi, ilgili seçme kodunu çözen bir AND kapısına bağlanır. Tüm AND kapılarının çıkışları tek bir OR kapısında toplanarak $Y$ elde edilir. Çoğu entegrede devreyi pasif kılan aktif-düşük (active-low) yetkilendirme (Enable - $\overline{E}$) ucu bulunur.

**Mekatronik ve Sayısal Sistemlerdeki Uygulamaları:**
1. **Mikrodenetleyici & İşlemci GPIO Çoklama:** STM32 gibi mikrodenetleyicilerde tek bir fiziksel pinin MUX aracılığıyla USART_TX, SPI_MOSI, PWM veya Standart GPIO olarak seçilmesi (Alternate Function Muxing).
2. **Aritmetik Mantık Birimi (ALU) Veri Yolu:** ALU içinde toplama, çıkarma, AND, OR işlemlerinin sonuçlarından birini op-code seçme bitlerine göre yazmaca aktarmak.
3. **Paralelden Seri Veri Dönüşümü:** Paralel 8-bitlik veri girişlerini bir sayıcı ($S$) ile tarayarak tek hattan seri iletmek.
4. **FPGA Mantık Blokları (LUT - Look-Up Table):** Modern FPGA'ler (Xilinx, Altera) tüm mantık fonksiyonlarını SRAM tabanlı MUX ağları ile gerçekleştirir. $2^n \times 1$ MUX, harici kapı gerektirmeden herhangi bir $n$ değişkenli Boolean fonksiyonunu doğrudan sentezler.

**3. Demultiplexer (DEMUX) ve Kod Çözücü (Decoder):**
DEMUX, MUX'ın tam tersi olarak tek bir veri girişini ($D$), $n$ adet seçme ucuyla $2^n$ adet çıkıştan birine aktarır. Bir $n$-to-$2^n$ Decoder (örneğin 3-to-8 decoder 74HC138), ikili adres kodunu aktif hatta dönüştürür; mikroişlemci bellek adres çözme (chip select) mimarisinin kalbidir.
