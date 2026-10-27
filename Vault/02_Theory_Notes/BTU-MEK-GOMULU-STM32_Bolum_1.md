# Gömülü Sistemler, ARM Cortex-M ve CAN Bus Mimarisi - ARM Cortex-M4 Mimarisi, Bellek Haritası ve Registerlar

**Kitap/Ders:** Gömülü Sistemler, ARM Cortex-M ve CAN Bus Mimarisi
**Yazarlar:** Joseph Yiu & STMicroelectronics

### ARM Cortex-M4 Çekirdek Mimarisi ve Bellek Organizasyonu

ARM Cortex-M4, Harvard mimarisine dayalı 32-bit RISC işlemcidir. Thumb-2 komut seti, donanımsal tek-çevrim FPU (Floating Point Unit) ve DSP komut uzantıları içerir.

**1. Çekirdek Register Yapısı:**
- $R_0 - R_{12}$: Genel amaçlı veri saklayıcılar (General Purpose Registers).
- $R_{13}$ (SP - Stack Pointer): Main Stack Pointer (MSP) ve Process Stack Pointer (PSP) olarak iki ayrı yığın işaretçisi içerir.
- $R_{14}$ (LR - Link Register): Fonksiyon çağrılarında veya kesmelerde geri dönüş adresini saklar.
- $R_{15}$ (PC - Program Counter): Çalıştırılacak bir sonraki komutun adresini tutar.
- xPSR (Program Status Register): APSR (Aritmetik bayraklar: N, Z, C, V), IPSR (Aktif kesme numarası) ve EPSR içerir.

**2. Bellek Haritası (Memory Map):**
Toplam 4 GB doğrusal adresleme alanı standart ARM segmentlerine bölünmüştür:
- `0x0000_0000 - 0x1FFF_FFFF`: Flash / Boot ROM bellek alanı (Kod ve Vektör Tablosu).
- `0x2000_0000 - 0x3FFF_FFFF`: Dahili SRAM bellek alanı (Değişkenler ve yığın).
- `0x4000_0000 - 0x5FFF_FFFF`: Periferaller (APB1, APB2, AHB veri yolu çevre birimleri: Timers, ADC, USART, CAN, SPI).
- `0xE000_0000 - 0xE00F_FFFF`: Sistem Kontrol Alanı (NVIC, SysTick, MPU, SCB).
