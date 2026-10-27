# Mekatronik Sensörler, Aktüatörler ve Sinyal Şartlandırma - Enkoderler, LVDT ve Gerinim Pulları (Wheatstone Köprüsü)

**Kitap/Ders:** Mekatronik Sensörler, Aktüatörler ve Sinyal Şartlandırma
**Yazarlar:** David G. Alciatore & Michael B. Histand

### Konum ve Kuvvet Sensörleri

**1. Optik ve Manyetik Enkoderler:**
- **Artımsal (Incremental) Enkoder:** Kanal A ve Kanal B sinyalleri $90^\circ$ faz farkıyla (quadrature) üretilir. A'nın B'den önde veya geride olması dönüş yönünü belirler. 4x kuadratür kod çözme (quadrature decoding) ile her iki kanalın yükselen ve düşen kenarları sayılarak çözünürlük 4 katına ($4 \times CPR$) çıkarılır. Z (indeks) kanalı her turda 1 pals vererek sıfırlama referansı sağlar.
- **Mutlak (Absolute) Enkoder:** Gray kodu diski kullanılarak güç kesilip açılsa dahi mutlak açısal konum anında okunur. Gray kodunda ardışık durumlar arasında yalnızca tek 1 bit değişir; böylece geçiş anlarındaki belirsizlikler (glitch) engellenir.

**2. Gerinim Pulu (Strain Gauge) ve Wheatstone Köprüsü:**
Mekanik deformasyon direnç değişimine dönüşür: $\frac{\Delta R}{R} = GF \cdot \epsilon$ ($GF$: Gauge Factor, $\epsilon$: Birim uzama).
Küçük direnç değişimlerini hassas gerilime dönüştürmek için Wheatstone köprüsü kullanılır:
$$V_{out} = V_s \left( \frac{R_1}{R_1 + R_2} - \frac{R_4}{R_3 + R_4} \right)$$
Tam köprü (Full-bridge) bağlantıda 4 gerinim pulu kullanılarak sıcaklık kaymaları dengelenir ve duyarlılık 4 katına çıkar.
