# Gömülü Sistemler, ARM Cortex-M ve CAN Bus Mimarisi - NVIC Kesme Yönetimi ve Doğrudan Bellek Erişimi (DMA)

**Kitap/Ders:** Gömülü Sistemler, ARM Cortex-M ve CAN Bus Mimarisi
**Yazarlar:** Joseph Yiu & STMicroelectronics

### İç İçe Vektörlü Kesme Kontrolcüsü (NVIC) ve DMA Mimarisi

**1. NVIC (Nested Vectored Interrupt Controller):**
- Donanımsal önceliklendirme ve düşük kesme gecikmesi (12 çevrim) sağlar.
- **Preemption Priority (Öncelikli Kesme):** Yüksek öncelikli bir kesme, o anda çalışan düşük öncelikli bir kesmeyi bölerek (preempt) hemen devreye girer.
- **Subpriority (Alt Öncelik):** Aynı anda iki kesme tetiklendiğinde hangisinin önce çalışacağını belirler, ancak çalışan kesmeyi bölemez.
- Kesme tetiklendiğinde donanım otomatik olarak $R_0-R_3, R_{12}, LR, PC, xPSR$ registerlarını yığına (Stack) iter (Hardware Stacking).

**2. DMA (Direct Memory Access) Kontrolcüsü:**
CPU'yu hiçbir döngüde meşgul etmeden çevre birimleri (ADC, SPI, UART, Timer) ile SRAM/Flash arasında yüksek hızlı blok veri transferi sağlar.
- **Dairesel Mod (Circular Mode):** Özellikle sensör okumalarında veya ADC çok kanallı taramada tampon dolduğunda başa dönerek kesintisiz akış sağlar.
- **Çift Tampon Modu (Double Buffer):** Bir tampon DMA tarafından doldurulurken diğeri CPU tarafından işlenir; böylece yarış durumu (race condition) ve veri kaybı önlenir.
