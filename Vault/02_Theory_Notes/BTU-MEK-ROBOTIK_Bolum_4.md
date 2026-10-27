# Modern Robotik: Kinematik, Dinamik ve Yörünge Kontrolü - Robot Dinamiği: Euler-Lagrange ve Newton-Euler Formülasyonu

**Kitap/Ders:** Modern Robotik: Kinematik, Dinamik ve Yörünge Kontrolü
**Yazarlar:** John J. Craig & Kevin M. Lynch, Frank C. Park

### Robot Dinamiği ve Hareket Denklemleri

Eklem torkları $\tau$ ile ivmeler $\ddot{q}$ arasındaki ilişki dinamik denklemlerle modellenir. **Euler-Lagrange Formülasyonu:**
Lagrange fonksiyonu $L = K - P$ (Kinetik Enerji - Potansiyel Enerji):
$$\frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q}} \right) - \frac{\partial L}{\partial q} = \tau$$

Matrisel formda standart manipülatör dinamik denklemi:
$$
M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) + F(\dot{q}) = \tau
$$
Burada:
- $M(q) \in \mathbb{R}^{n \times n}$: Simetrik ve pozitif tanımlı kütle/eylemsizlik matrisi ($M(q) = M^T(q) > 0$).
- $C(q, \dot{q})\dot{q} \in \mathbb{R}^n$: Coriolis ve merkezkaç (centrifugal) kuvvetler vektörü. Matris türevi $\dot{M}(q) - 2C(q, \dot{q})$ ters-simetrik (skew-symmetric) olup enerjinin korunduğunu gösterir:
$$s^T (\dot{M} - 2C) s = 0, \quad \forall s \in \mathbb{R}^n$$
- $G(q) \in \mathbb{R}^n$: Yerçekimi torkları vektörü ($G(q) = \frac{\partial P}{\partial q}$).
- $F(\dot{q})$: Viskoz ve Coulomb sürtünme terimleri ($F_v \dot{q} + F_c \text{sgn}(\dot{q})$).

**Hesaplanmış Tork Kontrolü (Computed Torque Control / Inverse Dynamics):**
Doğrusal olmayan dinamikleri telafi etmek için geri beslemeli doğrusallaştırma uygulanır:
$$\tau = M(q) \left( \ddot{q}_d + K_v (\dot{q}_d - \dot{q}) + K_p (q_d - q) \right) + C(q, \dot{q})\dot{q} + G(q)$$
Bu kontrol kuralı altında hata dinamiği $\ddot{e} + K_v \dot{e} + K_p e = 0$ olur ve kritik sönümlü olarak sıfıra yakınsar.
