# Modern Otomatik Kontrol Sistemleri ve Durum-Uzay Analizi - Durum-Uzay Modellemesi ve Transfer Fonksiyonu Dönüşümü

**Kitap/Ders:** Modern Otomatik Kontrol Sistemleri ve Durum-Uzay Analizi
**Yazarlar:** Katsuhiko Ogata & Gene F. Franklin

### Durum-Uzay (State-Space) Temsili ve LTI Sistemler

Çok girişli çok çıkışlı (MIMO) veya doğrusal zamanla değişmeyen (LTI) mekatronik sistemler 1. derece diferansiyel denklem takımı olarak ifade edilir:
$$
\dot{x}(t) = A x(t) + B u(t)
$$
$$
y(t) = C x(t) + D u(t)
$$
Burada $x(t) \in \mathbb{R}^n$ durum vektörü (örneğin konum ve hız), $u(t) \in \mathbb{R}^m$ kontrol giriş vektörü (örneğin motor gerilimi), $y(t) \in \mathbb{R}^p$ ölçülen çıkış vektörüdür.

**Durum-Uzayından Transfer Fonksiyonuna Geçiş:**
Laplace dönüşümü alınarak sıfır başlangıç şartlarında:
$$s X(s) = A X(s) + B U(s) \implies X(s) = (sI - A)^{-1} B U(s)$$
$$Y(s) = C X(s) + D U(s) = \left[ C(sI - A)^{-1} B + D \right] U(s)$$
$$G(s) = \frac{Y(s)}{U(s)} = C(sI - A)^{-1} B + D = \frac{C \text{adj}(sI - A) B}{\det(sI - A)} + D$$
Sistemin karakteristik denklemi ve özdeğerleri (kutupları) $\det(sI - A) = 0$ kökleridir.
