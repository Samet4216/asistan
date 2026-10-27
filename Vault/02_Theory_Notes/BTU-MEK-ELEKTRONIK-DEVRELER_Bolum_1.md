# Analog Elektronik, Yarıiletken Fiziği ve Devre Teorisi - Yarıiletken Fiziği, p-n Jonksiyonu ve Diyot Devreleri

**Kitap/Ders:** Analog Elektronik, Yarıiletken Fiziği ve Devre Teorisi
**Yazarlar:** Robert L. Boylestad, Louis Nashelsky & Adel S. Sedra, Kenneth C. Smith

### Yarıiletkenler, p-n Jonksiyonu ve Diyot Uygulamaları

Silisyum (Si) tabanlı yarıiletkenlerde p-tipi (delik çoğunluk taşıyıcı) ve n-tipi (elektron çoğunluk taşıyıcı) bölgelerin birleşimiyle p-n jonksiyonu oluşur. Birleşim yüzeyinde taşıyıcıların difüzyonu sonucu boşaltılmış bölge (depletion region) ve bir iç potansiyel bariyeri ($V_{bi} \approx 0.7\text{ V}$) meydana gelir.

**Shockley Diyot İdeal Akım Denklemi:**
$$
I_D = I_S \left( e^{\frac{V_D}{n V_T}} - 1 \right)
$$
Burada $I_S$ ters doyma akımı ($10^{-15} - 10^{-12}\text{ A}$), $n$ emisyon katsayısı ($1 \le n \le 2$), $V_T = \frac{k T}{q}$ termal gerilimdir ($T = 300\text{ K}$ oda sıcaklığında $V_T \approx 26\text{ mV}$).

**Temel Diyot Devreleri:**
1. **Köprü Doğrultucu (Full-Wave Bridge Rectifier):** 4 diyot ile AC girişin her iki yarım dalgasını da DC'ye dönüştürür. Çıkış tepe gerilimi $V_{peak,out} = V_{peak,in} - 2 V_D$ olur.
2. **Filtre Kondansatörü ve Dalgalanma Gerilimi (Ripple Voltage):**
   $$V_{ripple} = \frac{I_{load}}{2 f C}$$
3. **Zener Diyot ve Gerilim Regülasyonu:** Ters kırılma bölgesinde ($V_Z$) çalıştırılarak giriş gerilimi veya yük akımı değişse dahi yük üzerinde sabit gerilim sağlar. Seri direnç $R_s = \frac{V_{in,min} - V_Z}{I_{L,max} + I_{Z,min}}$ ile boyutlandırılır.
