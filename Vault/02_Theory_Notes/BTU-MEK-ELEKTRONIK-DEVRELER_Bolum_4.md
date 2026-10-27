# Analog Elektronik, Yarıiletken Fiziği ve Devre Teorisi - İşlemsel Yükselteçler (Op-Amp): Doğrusal ve Doğrusal Olmayan Devreler

**Kitap/Ders:** Analog Elektronik, Yarıiletken Fiziği ve Devre Teorisi
**Yazarlar:** Robert L. Boylestad, Louis Nashelsky & Adel S. Sedra, Kenneth C. Smith

### İşlemsel Yükselteçler (Op-Amp) ve Aktif Devreler

İdeal Op-Amp özellikleri: Sonsuz açık çevrim kazancı ($A_{OL} \to \infty$), sonsuz giriş empedansı ($R_{in} \to \infty \implies I_+ = I_- = 0$), sıfır çıkış empedansı ($R_{out} = 0$) ve sonsuz bant genişliği.
Negatif geri besleme altında sanal kısa devre (Virtual Ground / Virtual Short) prensibi geçerlidir: $V_+ = V_-$.

**1. Doğrusal Op-Amp Konfigürasyonları:**
- **Evirmeyen (Non-Inverting) Yükselteç:** $V_{out} = \left( 1 + \frac{R_f}{R_1} \right) V_{in}$
- **Eviren (Inverting) Yükselteç:** $V_{out} = -\frac{R_f}{R_1} V_{in}$
- **İdeal İntegratör:** $V_{out}(t) = -\frac{1}{R C} \int_0^t V_{in}(\tau) d\tau + V_{out}(0)$
- **İdeal Türev Alıcı:** $V_{out}(t) = -R C \frac{dV_{in}(t)}{dt}$ (Pratikte yüksek frekans gürültüsünü engellemek için seri giriş direnci eklenir).

**2. Schmitt Trigger (Histerezisli Karşılaştırıcı):**
Gürültülü sensör sinyallerinde çıkışın titreşmesini (chatter) önlemek için pozitif geri besleme kullanılır. İki farklı eşik gerilimi üretilir:
$$V_{TH} = \frac{R_1}{R_1 + R_2} (+V_{sat}), \quad V_{TL} = \frac{R_1}{R_1 + R_2} (-V_{sat})$$
Histerezis bandı $V_H = V_{TH} - V_{TL}$ gürültü bağışıklığı sağlar.

**3. Aktif Sallen-Key 2. Derece Alçak Geçiren Filtre:**
İki direnç ve iki kondansatör ile $40\text{ dB/decade}$ düşüş sağlayan transfer fonksiyonu:
$$H(s) = \frac{K \omega_0^2}{s^2 + \frac{\omega_0}{Q} s + \omega_0^2}, \quad f_c = \frac{1}{2\pi \sqrt{R_1 R_2 C_1 C_2}}$$
