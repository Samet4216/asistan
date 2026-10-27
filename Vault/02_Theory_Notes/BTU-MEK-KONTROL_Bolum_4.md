# Modern Otomatik Kontrol Sistemleri ve Durum-Uzay Analizi - Klasik PID Tasarımı, Anti-Windup ve Ziegler-Nichols Ayar Yöntemleri

**Kitap/Ders:** Modern Otomatik Kontrol Sistemleri ve Durum-Uzay Analizi
**Yazarlar:** Katsuhiko Ogata & Gene F. Franklin

### PID Kontrolör Tasarımı ve Ayrıklaştırma

Standart PID kontrolör transfer fonksiyonu:
$$u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}$$
$$C(s) = K_p \left( 1 + \frac{1}{T_i s} + T_d s \right)$$

- **$K_p$ (Oransal Kazanç):** Yükselme zamanını ($t_r$) düşürür, hatayı azaltır ancak aşımı (overshoot) artırabilir.
- **$K_i$ (İntegral Kazanç):** Kalıcı durum hatasını (steady-state error) tamamen sıfırlar ancak sistem fazını $90^\circ$ geciktirerek salınıma ve kararsızlığa yol açabilir.
- **$K_d$ (Türevsel Kazanç):** Sistem sönümünü artırır, aşımı düşürür ve gelecekteki eğilimi öngörür. Ancak yüksek frekanslı ölçüm gürültüsünü aşırı yükselttiği için pratikte alçak geçiren filtreyle ($N / (1 + N/s)$) birlikte kullanılır.

**İntegral Windup ve Anti-Windup Stratejisi:**
Aktüatör (motor gerilimi vb.) doyuma ulaştığında (saturation: $u_{sat} \ne u$), hata integrali şişmeye devam eder (windup). Hata işaret değiştirdiğinde sistem hemen toparlanamaz ve aşırı aşım yapar. Anti-windup clamping (integratör durdurma) veya back-calculation (geri hesaplama) ile integratör deşarj edilir.
