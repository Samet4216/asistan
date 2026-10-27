# Sinyaller, Sistemler ve Sayısal Sinyal İşleme (DSP) - Durum Kestirimi ve Ayrık Zamanlı Kalman Filtresi

**Kitap/Ders:** Sinyaller, Sistemler ve Sayısal Sinyal İşleme (DSP)
**Yazarlar:** Alan V. Oppenheim & Alan S. Willsky

### Ayrık Zamanlı Doğrusal Kalman Filtresi (Linear Kalman Filter)

Sensör gürültüsü ($R$) ve sistem dinamik belirsizliği ($Q$) içeren doğrusal bir sistemin optimum durum kestirimidir. İki adımdan oluşur:

**1. Kestirim (Prediction / Zaman Güncellemesi):**
Fiziksel model kullanılarak durum ve hata kovaryansı ileriye ötelenir:
$$\hat{x}_{k|k-1} = A \hat{x}_{k-1|k-1} + B u_k$$
$$P_{k|k-1} = A P_{k-1|k-1} A^T + Q$$

**2. Güncelleme (Correction / Ölçüm Güncellemesi):**
Sensörden gelen ölçüm $z_k$ kullanılarak kestirim düzeltilir:
İnovasyon (ölçüm artığı): $\tilde{y}_k = z_k - H \hat{x}_{k|k-1}$
İnovasyon kovaryansı: $S_k = H P_{k|k-1} H^T + R$
Optimum Kalman Kazancı: $K_k = P_{k|k-1} H^T S_k^{-1}$
Güncellenmiş durum: $\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k \tilde{y}_k$
Güncellenmiş kovaryans: $P_{k|k} = (I - K_k H) P_{k|k-1}$

Sensör çok gürültülü olduğunda ($R \to \infty$), $K_k \to 0$ olur ve filtre sadece modele güvenir. Model çok belirsiz olduğunda ($Q \to \infty$), $K_k \to H^{-1}$ olur ve filtre sensör ölçümünü kabul eder.
