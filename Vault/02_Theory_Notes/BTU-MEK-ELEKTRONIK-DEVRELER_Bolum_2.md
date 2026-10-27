# Analog Elektronik, Yarıiletken Fiziği ve Devre Teorisi - Bipolar Jonksiyon Transistörler (BJT): Kutuplama ve Küçük Sinyal Analizi

**Kitap/Ders:** Analog Elektronik, Yarıiletken Fiziği ve Devre Teorisi
**Yazarlar:** Robert L. Boylestad, Louis Nashelsky & Adel S. Sedra, Kenneth C. Smith

### BJT Transistörler ve Küçük Sinyal Modelleri

BJT (NPN veya PNP), akım kontrollü bir yarıiletken elemandır. Üç temel çalışma bölgesi vardır:
1. **Kesim (Cut-off):** $V_{BE} < 0.7\text{ V}$, $I_B = 0, I_C \approx 0$. Transistör açık bir anahtar gibidir.
2. **İleri Aktif (Forward Active - Yükselteç Bölgesi):** Baz-Emiter ileri kutuplu ($V_{BE} \approx 0.7\text{ V}$), Kollektör-Baz ters kutupludur. Akım kazancı $\beta$ (veya $h_{FE}$):
   $$I_C = \beta I_B, \quad I_E = I_B + I_C = (\beta + 1) I_B$$
3. **Doyum (Saturation):** Her iki jonksiyon da ileri kutupludur. Kollektör gerilimi $V_{CE,sat} \approx 0.2\text{ V}$ seviyesine düşer. Transistör kapalı bir anahtar gibi davranır ($I_C < \beta I_B$).

**DC Kutuplama (Voltaj Bölücülü Kutuplama - Voltage Divider Bias):**
Sıcaklık ve $\beta$ değişimlerinden etkilenmeyen kararlı çalışma noktası (Q-noktası) elde etmek için kullanılır:
$$V_{TH} = V_{CC} \frac{R_2}{R_1 + R_2}, \quad R_{TH} = R_1 \parallel R_2$$
$$I_B = \frac{V_{TH} - V_{BE}}{R_{TH} + (\beta + 1) R_E}$$

**Küçük Sinyal Hibrid-$\pi$ Modeli:**
AC sinyaller için transistör lineerleştirilir:
- Eşdeğer Transkondüktans: $g_m = \frac{I_{CQ}}{V_T} = \frac{I_{CQ}}{26\text{ mV}}$
- Baz Giriş Direnci: $r_\pi = \frac{\beta}{g_m} = \frac{\beta V_T}{I_{CQ}}$
- Early Direnci (Çıkış Direnci): $r_o = \frac{V_A + V_{CEQ}}{I_{CQ}}$ ($V_A$: Early Gerilimi)
- Ortak Emiterli (Common-Emitter) Gerilim Kazancı: $A_v = -g_m (R_C \parallel R_L \parallel r_o)$.
