# Modern Robotik: Kinematik, Dinamik ve Yörünge Kontrolü - İleri ve Ters Kinematik Analizi (FK / IK)

**Kitap/Ders:** Modern Robotik: Kinematik, Dinamik ve Yörünge Kontrolü
**Yazarlar:** John J. Craig & Kevin M. Lynch, Frank C. Park

### İleri ve Ters Kinematik Analizi (FK ve IK)

**İleri Kinematik (Forward Kinematics - FK):**
Eklem değişkenleri $q = [\theta_1, \theta_2, \dots, \theta_n]^T$ bilindiğinde, uç işlevcinin Kartezyen uzaydaki konumu $P = [x, y, z]^T$ ve yönelimi $R \in SO(3)$ homojen dönüşüm zinciriyle doğrudan ve tekil bir çözüm olarak bulunur:
$$X = f(q)$$

**Ters Kinematik (Inverse Kinematics - IK):**
İstenen hedef pozisyon ve yönelim $X_{hedef}$ verildiğinde, bunu sağlayacak eklem açıları $q = f^{-1}(X)$ çözülür. Ters kinematik doğrusal olmayan trigonometrik denklemler içerir; birden fazla geçerli çözüm (dirsek yukarı / dirsek aşağı), sonsuz çözüm veya tekillik durumunda çözümsüzlük görülebilir.
- **Analitik (Kapalı Form) Çözümler:** Geometrik yaklaşım ve Pieper Kriteri. Eğer robotun 3 ardışık döner eklem ekseni tek bir noktada kesişiyorsa (örneğin küresel bilek - spherical wrist yapısı), ters kinematik analitik olarak kesin biçimde çözülebilir. Bilek merkez noktası konumu önce bulunur ($P_w = P_{e} - d_6 R_{hedef} [0, 0, 1]^T$), ilk 3 eklem konumu, son 3 eklem ise Euler açıları ile yönelimi çözer.
- **Sayısal Çözümler (Newton-Raphson & Levenberg-Marquardt):** Jacobian tersi veya devriği kullanılarak iteratif güncellenir:
$$q_{k+1} = q_k + J^\dagger(q_k) (X_{hedef} - f(q_k))$$

