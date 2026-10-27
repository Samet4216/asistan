# Sayısal Mantık Devreleri ve Sayısal Sistem Tasarımı - Ardışıl Mantık: Flip-Floplar, Saat Parametreleri ve Sayıcılar

**Kitap/Ders:** Sayısal Mantık Devreleri ve Sayısal Sistem Tasarımı
**Yazarlar:** M. Morris Mano & Michael D. Ciletti, Thomas L. Floyd

### Ardışıl Mantık Devreleri ve Flip-Floplar

Çıkışın hem mevcut girişlere hem de devrenin geçmiş durumuna (bellek) bağlı olduğu sistemlerdir. Saat darbesiyle (Clock) senkronize edilirler.

**1. Temel Flip-Flop Tipleri ve Karakteristik Denklemleri:**
- **SR Flip-Flop:** $Q_{next} = S + \overline{R}Q$ ($S=1, R=1$ durumu belirsiz ve yasaktır).
- **D (Data) Flip-Flop:** $Q_{next} = D$. Sayısal kaydırmalı yazmaçlar ve pipeline mimarisinin temel yapı taşıdır.
- **JK Flip-Flop:** SR'ın yasak durumunu toggle (durum değiştirme) yaparak çözer:
  $$Q_{next} = J\overline{Q} + \overline{K}Q$$
  $J=1, K=1$ olduğunda çıkış terslenir ($Q_{next} = \overline{Q}$).
- **T (Toggle) Flip-Flop:** $Q_{next} = T \oplus Q$. $T=1$ ise her saat darbesinde durum değişir (frekans bölücü ve sayıcı temeli).

**2. Zamanlama Parametreleri (Timing Constraints):**
- **Setup Time ($t_{su}$):** Saat kenarı gelmeden önce verinin giriş ucunda kararlı kalması gereken minimum süre.
- **Hold Time ($t_h$):** Saat kenarı geldikten sonra verinin bozulmadan tutulması gereken minimum süre.
Bu kurallar ihlal edilirse flip-flop kararsızlık (metastability) durumuna düşer ve çıkış mantık 0 ile 1 arasında asılı kalarak devreyi çökertir.

**3. Sayıcılar (Counters):**
- Asenkron (Ripple) Sayıcı: Bir flip-flop'un çıkışı diğerinin saat girişini besler; gecikme toplamı artar.
- Senkron Sayıcı: Tüm flip-flop'lar aynı saat darbesiyle tetiklenir, yüksek frekanslarda kararlıdır.
