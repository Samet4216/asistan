# Sinyaller, Sistemler ve Sayısal Sinyal İşleme (DSP) - Örnekleme Teoremi (Nyquist-Shannon) ve Aliasing

**Kitap/Ders:** Sinyaller, Sistemler ve Sayısal Sinyal İşleme (DSP)
**Yazarlar:** Alan V. Oppenheim & Alan S. Willsky

### Örnekleme Teoremi ve Örtüşme (Aliasing)

Sürekli bir analog sinyalin $x(t)$ bilgi kaybı olmaksızın ayrık zamanlı diziye $x[n] = x(n T_s)$ dönüştürülmesi ve kusursuz şekilde geri çatılabilmesi (reconstruction) için örnekleme frekansı $f_s$, sinyalin içerdiği en yüksek frekans bileşeninin ($f_{max}$) en az iki katı olmalıdır:
$$f_s \ge 2 f_{max} \quad (\text{Nyquist Oranı})$$

Eğer $f_s < 2 f_{max}$ ise, yüksek frekanslı bileşenler alçak frekans bölgesine katlanarak sahte (hayali) alçak frekanslı sinyaller oluşturur (Aliasing / Örtüşme). Bunu önlemek için ADC (Analog-Digital Converter) girişinden önce kesim frekansı $f_c \le f_s / 2$ olan analog donanımsal bir **Anti-Aliasing Filtresi** (Alçak Geçiren Filtre) konulması zorunludur.
