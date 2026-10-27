# Modern Otomatik Kontrol Sistemleri ve Durum-Uzay Analizi - Frekans Cevabı: Bode Diyagramları, Faz/Kazanç Payı ve Nyquist Kararlılığı

**Kitap/Ders:** Modern Otomatik Kontrol Sistemleri ve Durum-Uzay Analizi
**Yazarlar:** Katsuhiko Ogata & Gene F. Franklin

### Frekans Domeni Analizi ve Kararlılık Kriterleri

Sisteme harmonik giriş $u(t) = U_0 \sin(\omega t)$ uygulandığında kalıcı durum çıkışı genlik ve faz farkıyla $y_{ss}(t) = Y_0 \sin(\omega t + \phi)$ olur:
$$G(j\omega) = |G(j\omega)| e^{j\angle G(j\omega)}$$
- Genlik (dB): $M_{dB} = 20 \log_{10} |G(j\omega)|$
- Faz (derece): $\phi = \arg(G(j\omega))$

**Kazanç Payı (Gain Margin - GM) ve Faz Payı (Phase Margin - PM):**
- **Kazanç Geçiş Frekansı (Gain Crossover Frequency - $\omega_{gc}$):** Genliğin $0\text{ dB}$ olduğu frekans ($|G(j\omega_{gc})| = 1$).
- **Faz Geçiş Frekansı (Phase Crossover Frequency - $\omega_{pc}$):** Fazın $-180^\circ$ olduğu frekans ($\angle G(j\omega_{pc}) = -180^\circ$).
- **Faz Payı:** $\text{PM} = 180^\circ + \angle G(j\omega_{gc})$ (Kararlı ve iyi sönümlü bir sistem için tipik olarak $45^\circ \le \text{PM} \le 65^\circ$ istenir).
- **Kazanç Payı:** $\text{GM} = -20 \log_{10} |G(j\omega_{pc})| \text{ dB}$.

**Nyquist Kararlılık Kriteri:**
Kapalı çevrim kararsız kutup sayısı $Z = N + P$. Burada $P$ açık çevrim kararsız kutup sayısı, $N$ ise açık çevrim Nyquist eğrisinin $(-1, j0)$ kritik noktasını saat yönünün tersine dolanma (encirclement) sayısıdır. Sistemin kapalı çevrimde kararlı olması için $Z = 0$, dolayısıyla $N = -P$ olmalıdır.
