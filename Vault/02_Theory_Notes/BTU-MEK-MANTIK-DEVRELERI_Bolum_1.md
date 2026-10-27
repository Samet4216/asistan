# Sayısal Mantık Devreleri ve Sayısal Sistem Tasarımı - Boolean Cebri, Temel Kapılar ve De Morgan Teoremleri

**Kitap/Ders:** Sayısal Mantık Devreleri ve Sayısal Sistem Tasarımı
**Yazarlar:** M. Morris Mano & Michael D. Ciletti, Thomas L. Floyd

### Boolean Cebri, Temel Mantık Kapıları ve De Morgan Kuralları

Sayısal elektronik sistemler iki değerli (0 ve 1, Low ve High) Boolean cebri üzerine kuruludur. Temel mantık kapıları AND, OR, NOT, NAND, NOR, XOR ve XNOR'dur. NAND ve NOR kapıları 'evrensel kapılar' (universal gates) olup sadece bu kapılar kullanılarak tüm mantıksal fonksiyonlar türetilebilir.

**De Morgan Teoremleri:**
Mantıksal çarpımın değili, değillerin toplamına; mantıksal toplamın değili ise değillerin çarpımına eşittir:
$$
\overline{A \cdot B} = \overline{A} + \overline{B}
$$
$$
\overline{A + B} = \overline{A} \cdot \overline{B}
$$
Genelleştirilmiş De Morgan kuralı ile $n$ değişkenli fonksiyonlar tek adımda dönüştürülebilir: $\overline{f(x_1, x_2, \dots, x_n, \cdot, +)} = f(\overline{x}_1, \overline{x}_2, \dots, \overline{x}_n, +, \cdot)$.

**Boolean Eşitlikleri ve İndirgeme Kuralları:**
- Yutma Kuralı (Absorption): $A + A \cdot B = A$ ve $A \cdot (A + B) = A$
- Dağılma Kuralı (Distributive): $A + (B \cdot C) = (A + B) \cdot (A + C)$
- Sadeleştirme Kuralı: $A + \overline{A} B = A + B$
- XOR Fonksiyonu: $A \oplus B = A\overline{B} + \overline{A}B$
- XNOR (Eşdeğerlik) Fonksiyonu: $\overline{A \oplus B} = AB + \overline{A}\,\overline{B}$
