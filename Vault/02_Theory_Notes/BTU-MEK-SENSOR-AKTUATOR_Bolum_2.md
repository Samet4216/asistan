# Mekatronik Sensörler, Aktüatörler ve Sinyal Şartlandırma - MEMS IMU: İvmeölçer, Jiroskop ve Tamamlayıcı Filtre (Complementary Filter)

**Kitap/Ders:** Mekatronik Sensörler, Aktüatörler ve Sinyal Şartlandırma
**Yazarlar:** David G. Alciatore & Michael B. Histand

### 6-DoF Ataletsel Ölçüm Birimi (IMU) ve Sensör Füzyonu

**1. MEMS İvmeölçer (Accelerometer):** Kütle-yay sistemi prensibiyle yerçekimi ivmesi dahil çizgisel ivmeyi ölçer. Statik durumda eğim açısını verir ($ roll = \arctan2(a_y, a_z) $). Ancak dinamik hareketlerdeki titreşim ve ivmelerden yüksek frekanslı gürültü içerir.
**2. MEMS Jiroskop (Gyroscope):** Coriolis ivmesi prensibiyle açısal hızları ($\omega$) ölçer. Açısal konum için integral alınır: $\theta(t) = \int \omega(t) dt$. Kısa vadede çok hassastır ancak integrasyon nedeniyle sabit bir sapma (drift/bias) birikerek uzun vadede kaymaya neden olur.

**3. Tamamlayıcı Filtre (Complementary Filter):**
Düşük frekanslı ivmeölçer bilgisi ile yüksek frekanslı jiroskop integralini birleştirir:
$$
\theta_{t} = \alpha \left( \theta_{t-1} + \omega_t \Delta t \right) + (1 - \alpha) \theta_{acc}
$$
Burada $\alpha = \frac{\tau}{\tau + \Delta t} \approx 0.98$ olarak seçilir. Jiroskopun drift yapması engellenirken ivmeölçerin titreşim gürültüsü süzülür.
