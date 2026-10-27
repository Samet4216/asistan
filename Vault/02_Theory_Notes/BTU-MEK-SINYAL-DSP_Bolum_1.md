# Sinyaller, Sistemler ve Sayısal Sinyal İşleme (DSP) - Doğrusal Zamanla Değişmeyen (LTI) Sistemler ve Konvolüsyon

**Kitap/Ders:** Sinyaller, Sistemler ve Sayısal Sinyal İşleme (DSP)
**Yazarlar:** Alan V. Oppenheim & Alan S. Willsky

### LTI Sistemler ve Konvolüsyon İşlemi

Bir sistem doğrusallık (süperpozisyon) ve zamanla değişmezlik özelliklerini sağlıyorsa LTI (Linear Time-Invariant) sistemdir. LTI bir sistem birim darbe cevabı (impulse response) $h(t)$ veya $h[n]$ ile tamamen karakterize edilir.

**Sürekli Zaman Konvolüsyon İntegrali:**
$$y(t) = x(t) * h(t) = \int_{-\infty}^{\infty} x(\tau) h(t - \tau) d\tau$$

**Ayrık Zaman Konvolüsyon Toplamı:**
$$y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k] h[n - k]$$

Frekans domeninde konvolüsyon basit bir çarpmaya dönüşür: $\mathcal{F}\{x * h\} = X(j\omega) H(j\omega)$. Bu özellik sayısal filtrelerin verimli hesaplanmasında FFT (Hızlı Fourier Dönüşümü) tabanlı hızlı konvolüsyonun temelidir.
