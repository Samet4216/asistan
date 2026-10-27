# Analog Elektronik, Yarıiletken Fiziği ve Devre Teorisi - Alan Etkili Transistörler (MOSFET): Karakteristik Denklemler ve CMOS

**Kitap/Ders:** Analog Elektronik, Yarıiletken Fiziği ve Devre Teorisi
**Yazarlar:** Robert L. Boylestad, Louis Nashelsky & Adel S. Sedra, Kenneth C. Smith

### MOSFET (Metal-Oxide-Semiconductor Field-Effect Transistor)

MOSFET gerilim kontrollü bir elemandır; Gate (kapı) ucu silikon dioksit ($SiO_2$) yalıtkan katmanla ayrıldığı için DC kapı akımı sıfırdır ($I_G = 0$). Giriş empedansı teorik olarak sonsuzdur.

**Zenginleştirilmiş NMOS Çalışma Bölgeleri:**
1. **Kesim Bölgesi ($V_{GS} < V_{th}$):** İletim kanalı oluşmamıştır. $I_D = 0$.
2. **Lineer / Triyot Bölgesi ($V_{GS} \ge V_{th}$ ve $V_{DS} < V_{GS} - V_{th}$):**
   Kanal boydan boya iletkendir, gerilim kontrollü direnç gibi davranır:
   $$I_D = \mu_n C_{ox} \frac{W}{L} \left[ (V_{GS} - V_{th}) V_{DS} - \frac{1}{2} V_{DS}^2 \right]$$
3. **Doyum (Saturation / Pinch-off) Bölgesi ($V_{GS} \ge V_{th}$ ve $V_{DS} \ge V_{GS} - V_{th}$):**
   Kanal Drain ucunda boğulur (pinch-off). Akım $V_{DS}$'den bağımsız hale gelerek sabitlenir (yükselteç çalışma bölgesi):
   $$I_D = \frac{1}{2} \mu_n C_{ox} \frac{W}{L} (V_{GS} - V_{th})^2 (1 + \lambda V_{DS})$$
   Burada $\lambda$ kanal boyu modülasyon parametresidir (Early etkisinin MOSFET karşılığı).

**CMOS Teknolojisi (Complementary MOS):**
Bir pMOS ve bir nMOS transistörün seri bağlanmasıyla oluşur. Mantık 0 veya 1 durumlarında transistörlerden biri daima kesimde olduğundan statik güç tüketimi sıfıra yakındır; güç yalnızca anahtarlama anında (dinamik: $P = C f V_{DD}^2$) tüketilir.
