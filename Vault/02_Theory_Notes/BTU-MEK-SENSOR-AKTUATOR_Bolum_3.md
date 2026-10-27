# Mekatronik Sensörler, Aktüatörler ve Sinyal Şartlandırma - Fırçasız DC (BLDC) Motorlar, Hall Sensörleri ve 6-Adımlı Komütasyon

**Kitap/Ders:** Mekatronik Sensörler, Aktüatörler ve Sinyal Şartlandırma
**Yazarlar:** David G. Alciatore & Michael B. Histand

### Fırçasız DC (BLDC) Motor Komütasyonu ve Sürüşü

BLDC motorlarda rotor kalıcı mıknatıs, stator ise 3 fazlı (U, V, W) sargılardan oluşur. Mekanik fırça/kolektör yerine elektronik anahtarlama yapılır.

**1. Hall Etkisi Sensörleri ve Rotor Pozisyonu:**
120 elektriksel derece aralıkla yerleştirilmiş 3 Hall sensörü ($H_1, H_2, H_3$) rotor kutuplarını algılayarak 6 farklı mantıksal durum (sektör: 1'den 6'ya) üretir.

**2. Üç Fazlı H-Köprüsü (Six-Step / Trapezoidal Commutation):**
6 adet MOSFET/IGBT anahtarı (3 üst - High Side, 3 alt - Low Side) kullanılır:
- Her 60 elektriksel derecelik adımda 1 faz pozitif beslemeye ($+V_{dc}$), 1 faz toprağa (GND) çekilir, 3. faz ise boşta (floating) bırakılır.
- Boşta kalan fazda rotasyon sonucu Zıt EMF (Back-EMF) endüklenir. Bu gerilimin sıfır geçiş noktası (Zero-Crossing Detection - ZCD) algılanarak sensörsüz (sensorless) sürüş de yapılabilir.

**3. Alan Yönelimli Kontrol (Field Oriented Control - FOC):**
Clarke ve Park dönüşümleri ile 3 fazlı AC akımlar ($i_a, i_b, i_c$), dönen rotor referans çerçevesinde DC büyüklüklere dönüştürülür:
- $i_d$ (Doğrudan eksen akımı): Manyetik akıyı oluşturur, genellikle sıfıra regüle edilir ($i_d^* = 0$).
- $i_q$ (Karesel eksen akımı): Tork üreten bileşendir ($\tau = \frac{3}{2} p \lambda_m i_q$). Böylece motor maksimum verim ve minimum tork dalgalanması ile sürülür.
