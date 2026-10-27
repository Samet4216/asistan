# Modern Robotik: Kinematik, Dinamik ve Yörünge Kontrolü - Manipülatör Jacobian Matrisi, Hız Kinematiği ve Tekillik Analizi

**Kitap/Ders:** Modern Robotik: Kinematik, Dinamik ve Yörünge Kontrolü
**Yazarlar:** John J. Craig & Kevin M. Lynch, Frank C. Park

### Diferansiyel Kinematik, Geometrik Jacobian ve Tekillikler

Manipülatör Jacobian matrisi $J(q) \in \mathbb{R}^{6 \times n}$, eklem hız vektörü $\dot{q}$ ile uç işlevcinin çizgisel ($v$) ve açısal ($\omega$) hız vektörü arasındaki doğrusal bağıntıyı kurar:
$$
\xi = \begin{bmatrix} v \\ \omega \end{bmatrix} = J(q) \dot{q}
$$
Her $i$'nci döner eklem için Jacobian sütunu:
$$J_i(q) = \begin{bmatrix} z_{i-1} \times (p_e - p_{i-1}) \\ z_{i-1} \end{bmatrix}$$
Prizmatik eklemler için ise:
$$J_i(q) = \begin{bmatrix} z_{i-1} \\ 0 \end{bmatrix}$$

**Kinematik Tekillikler (Kinematic Singularities):**
Eğer $\det(J(q)) = 0$ (kare Jacobian için) veya rankı düşerse ($	ext{rank}(J) < 6$):
1. Manipülatör belirli yönlerde hareket serbestliğini kaybeder (o yönde sıfır hız üretilebilir).
2. İstenen son hız için gereken eklem hızları sonsuza yaklaşır ($\dot{q} = J^{-1} \xi \to \infty$).
3. Kartezyen kuvvet kontrolünde robot sonsuz mekanik avantaj kazanarak çevreye aşırı kontrolsüz kuvvet uygulayabilir.

Tekillik analizi için **Manipülabilirlik Elipsoidi (Yoshikawa):**
$$w = \sqrt{\det(J(q) J^T(q))}$$
$w \to 0$ olduğunda robot tekillik sınırındadır. Çözüm olarak Damped Least Squares (Levenberg-Marquardt sönümlendirmesi) kullanılır: $J^* = J^T (J J^T + \lambda^2 I)^{-1}$.
