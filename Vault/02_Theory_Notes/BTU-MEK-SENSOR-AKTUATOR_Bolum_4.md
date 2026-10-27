# Mekatronik Sensörler, Aktüatörler ve Sinyal Şartlandırma - Güç Elektroniği Anahtarlama ve Op-Amp Sinyal Şartlandırma

**Kitap/Ders:** Mekatronik Sensörler, Aktüatörler ve Sinyal Şartlandırma
**Yazarlar:** David G. Alciatore & Michael B. Histand

### Güç Elektroniği ve Analog Sinyal Şartlandırma

**1. MOSFET/IGBT Sürüşü ve Dead-Time (Ölü Zaman):**
Aynı köprü kolundaki üst ve alt MOSFET'lerin aynı anda iletime geçmesi kısa devreye (shoot-through) neden olarak transistörleri patlatır. Bunu önlemek için donanımsal veya yazılımsal 'Dead-Time' gecikmesi (tipik 500ns - 2µs) uygulanır.
İndüktif yüklerin (motor sargıları) enerjisini boşaltmak için ters bağlı hızlı geri besleme diyotları (freewheeling diodes) gereklidir.

**2. Enstrümantasyon Yükselteci (Instrumentation Amplifier):**
Düşük seviyeli sensör sinyallerini (örneğin mV mertebesindeki gerinim pulu) yükseltmek ve ortak mod gürültüsünü (CMRR) yok etmek için 3 Op-Amp'lı enstrümantasyon yükselteci kullanılır:
$$V_{out} = \left( 1 + \frac{2 R_1}{R_G} \right) \frac{R_3}{R_2} (V_2 - V_1)$$
Tek bir $R_G$ direnci ayarlanarak sistem kazancı hassasça belirlenir.
