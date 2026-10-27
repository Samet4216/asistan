# Modern Otomatik Kontrol Sistemleri ve Durum-Uzay Analizi - Kontrol Edilebilirlik, Gözlenebilirlik ve Kutup Yerleştirme

**Kitap/Ders:** Modern Otomatik Kontrol Sistemleri ve Durum-Uzay Analizi
**Yazarlar:** Katsuhiko Ogata & Gene F. Franklin

### Kontrol Edilebilirlik (Controllability) ve Gözlenebilirlik (Observability)

**1. Kalman Kontrol Edilebilirlik Matrisi $(\mathcal{C})$:**
Sistemin durumlarının sonlu bir sürede istenen herhangi bir değere sürülebilmesi için kontrol edilebilirlik matrisinin tam ranka sahip olması gerekir:
$$\mathcal{C} = \begin{bmatrix} B & AB & A^2 B & \dots & A^{n-1} B \end{bmatrix} \in \mathbb{R}^{n \times nm}$$
$$\text{rank}(\mathcal{C}) = n \iff \text{Sistem tamamen kontrol edilebilirdir.}$$

**2. Kalman Gözlenebilirlik Matrisi $(\mathcal{O})$:**
Sistem çıkışlarından $y(t)$ durumların tamamının $x(t)$ kestirilebilmesi için:
$$\mathcal{O} = \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{n-1} \end{bmatrix} \in \mathbb{R}^{np \times n}$$
$$\text{rank}(\mathcal{O}) = n \iff \text{Sistem tamamen gözlenebilirdir.}$$

**3. Durum Geri Beslemesi ve Kutup Yerleştirme (Ackermann Formülü):**
Kontrol edilebilir bir sistemde durum geri besleme kazancı $u = -Kx + r$ tasarlanarak kapalı çevrim sistemi $\dot{x} = (A - BK)x + Br$ elde edilir. İstenen karakteristik polinom $\alpha_c(s) = s^n + \alpha_1 s^{n-1} + \dots + \alpha_n$ olmak üzere:
$$K = \begin{bmatrix} 0 & 0 & \dots & 1 \end{bmatrix} \mathcal{C}^{-1} \alpha_c(A)$$

**4. Luenberger Gözlemcisi (State Observer):**
Durumların doğrudan ölçülemediği durumlarda durum kestiricisi $\dot{\hat{x}} = A\hat{x} + Bu + L(y - C\hat{x})$ tasarlanır. Kestirim hatası $\dot{e} = (A - LC)e$ olup $L$ kazancı özdeğerleri sol yarı düzlemde hızla sıfıra gidecek şekilde seçilir.
