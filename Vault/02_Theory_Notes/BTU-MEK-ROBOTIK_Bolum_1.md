# Modern Robotik: Kinematik, Dinamik ve Yörünge Kontrolü - Denavit-Hartenberg (DH) Parametreleri ve Homojen Dönüşüm Matrisleri

**Kitap/Ders:** Modern Robotik: Kinematik, Dinamik ve Yörünge Kontrolü
**Yazarlar:** John J. Craig & Kevin M. Lynch, Frank C. Park

### Denavit-Hartenberg (DH) Parametreleri ve Çerçeve Yerleşim Kuralları

Robot manipülatörlerinin ardışık uzuvları arasındaki kinematik bağıntıyı kurmak için standart ve modifiye Denavit-Hartenberg (DH) konvansiyonu kullanılır. Her eklem ekseni için bir koordinat çerçevesi $\{i\}$ atanır:
1. **$z_i$ ekseni:** $i+1$'inci eklemin dönme veya ötelenme ekseni doğrultusunda seçilir.
2. **$x_i$ ekseni:** $z_{i-1}$ ile $z_i$ eksenlerinin ortak dikmesi (common normal) boyunca ve $z_{i-1}$'den $z_i$'ye doğru yönlendirilir. Kesişen eksenlerde $x_i = z_{i-1} \times z_i$ yönünde seçilir.
3. **$y_i$ ekseni:** Sağ el kuralı tamamlanacak şekilde $y_i = z_i \times x_i$ olarak tanımlanır.

Dört temel DH parametresi şunlardır:
- **$a_i$ (Uzuv Uzunluğu - Link Length):** $z_{i-1}$ ile $z_i$ arasındaki mesafe ($x_i$ ekseni boyunca ölçülür).
- **$\alpha_i$ (Uzuv Eğimi - Link Twist):** $z_{i-1}$ ekseninden $z_i$ eksenine olan açı ($x_i$ etrafında sağ el kuralı ile ölçülür).
- **$d_i$ (Uzuv Ötelemesi - Link Offset):** $x_{i-1}$ ile $x_i$ arasındaki mesafe ($z_{i-1}$ ekseni boyunca ölçülür, prizmatik eklemlerde değişkendir).
- **$\theta_i$ (Eklem Açısı - Joint Angle):** $x_{i-1}$ ekseninden $x_i$ eksenine olan açı ($z_{i-1}$ etrafında sağ el kuralı ile ölçülür, döner eklemlerde değişkendir).

Komşu çerçeveler arasındaki $4 \times 4$ homojen dönüşüm matrisi:
$$
^{i-1}T_i = \begin{bmatrix}
\cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i \cos\theta_i \\
\sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i \sin\theta_i \\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}
$$
Taban koordinat sisteminden uç işlevciye (end-effector) kadar olan toplam ileri kinematik matrisi ardışık çarpımla elde edilir: $$^0T_n = \prod_{i=1}^{n} {^{i-1}T_i}$$

