# Gömülü Sistemler, ARM Cortex-M ve CAN Bus Mimarisi - CAN Bus (CAN 2.0B ve CAN-FD) Protokolü ve Bit Arbitrasyonu

**Kitap/Ders:** Gömülü Sistemler, ARM Cortex-M ve CAN Bus Mimarisi
**Yazarlar:** Joseph Yiu & STMicroelectronics

### Endüstriyel Haberleşme: Controller Area Network (CAN Bus)

Otomotiv ve robotik sistemlerde elektromanyetik parazitlere (EMI) dayanıklı, çok efendili (multi-master) ve mesaj öncelikli diferansiyel seri haberleşme protokolüdür.

**1. Diferansiyel Fiziksel Katman (Physical Layer):**
İki hatlı bükümlü kablo ($CAN\_H$ ve $CAN\_L$) ve hat sonlarında yansımaları sönümleyen $120\,\Omega$ hat sonlandırma dirençleri kullanılır:
- **Resesif (Çekinik - Mantık 1):** Her iki hat da $2.5\text{ V}$ seviyesindedir ($V_{diff} = CAN\_H - CAN\_L = 0\text{ V}$).
- **Dominant (Baskın - Mantık 0):** $CAN\_H = 3.5\text{ V}$, $CAN\_L = 1.5\text{ V}$ ($V_{diff} = 2.0\text{ V}$). Dominant bit her zaman çekinik biti ezer.

**2. Kayıpsız Çarpışma Çözümü (CSMA/CD + Non-Destructive Bitwise Arbitration):**
Aynı anda birden fazla düğüm hatta mesaj göndermeye başladığında, her düğüm kendi gönderdiği biti hattan geri okur:
- Mesaj ID'si düşük olan mesaj (örneğin acil fren mesajı: `0x001`), ID bitlerinde daha erken '0' (dominant) basar.
- Çekinik '1' basarken hattan dominant '0' okuyan düğüm arbitrasyonu kaybettiğini anlar ve mesajını hemen keserek dinleme moduna geçer.
- Böylece en yüksek öncelikli mesaj sıfır gecikmeyle ve veri bozulması olmadan iletilir.

**3. CAN 2.0B vs. CAN-FD:**
- Standart CAN 2.0B: Maksimum $1\text{ Mbps}$ hız ve çerçeve başına maksimum 8 bayt veri taşır.
- CAN-FD (Flexible Data-Rate): Veri alanında bit hızını $5-8\text{ Mbps}$ seviyesine çıkarır ve yük kapasitesini çerçeve başına 64 bayta yükseltir.
