"""
BTÜ Mechatronics Engineering Core Books & Compendiums Ingestion Script.
Ingests comprehensive textbook knowledge for:
1. Modern Robotik (Craig, Lynch & Park) - Kinematics, Dynamics, Jacobians
2. Modern Kontrol Sistemleri (Ogata, Franklin) - Transfer Functions, State-Space, Bode, Root-Locus
3. Mekatronik Sensörler & Aktüatörler (Alciatore) - Encoders, BLDC, IMU, Op-Amps, H-Bridge
4. Sinyaller & Sayısal Sinyal İşleme (Oppenheim) - LTI, Fourier, Z-Transform, Kalman Filter
5. Gömülü Sistemler & ARM STM32 / CAN Bus (Yiu, ST) - Cortex-M4, NVIC, DMA, CAN-FD
6. Robot İşletim Sistemi (ROS2 Humble & Nav2) - Nodes, Topics, Actions, TF2, URDF, Nav2

Indexes into ChromaDB collection 'paper_chunks', creates Obsidian Cards,
and adds entries into 'data/corpus_catalog.json'.
"""

import sys
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from ingest.indexer import Indexer

BTU_BOOKS = [
    {
        "id": "BTU-MEK-ROBOTIK",
        "title": "Modern Robotik: Kinematik, Dinamik ve Yörünge Kontrolü",
        "authors": "John J. Craig & Kevin M. Lynch, Frank C. Park",
        "year": 2024,
        "domain": "BTÜ Mekatronik / Robotik",
        "color": "#3b82f6",
        "desc": "BTÜ Robotik dersi ana kaynakları: Denavit-Hartenberg (DH) parametreleri, homojen dönüşümler, ileri/ters kinematik, Jacobian tekillikleri ve Euler-Lagrange dinamik denklemleri.",
        "prompt": "BTÜ Mekatronik Robotik dersindeki Denavit-Hartenberg (DH) kural adımlarını, homojen dönüşüm matrisini ve Jacobian tekillik analizi denklemlerini detaylandır.",
        "sections": [
            {
                "section_name": "Denavit-Hartenberg (DH) Parametreleri ve Homojen Dönüşüm Matrisleri",
                "content": (
                    "### Denavit-Hartenberg (DH) Parametreleri ve Çerçeve Yerleşim Kuralları\n\n"
                    "Robot manipülatörlerinin ardışık uzuvları arasındaki kinematik bağıntıyı kurmak için standart ve modifiye Denavit-Hartenberg (DH) konvansiyonu kullanılır. "
                    "Her eklem ekseni için bir koordinat çerçevesi $\{i\}$ atanır:\n"
                    "1. **$z_i$ ekseni:** $i+1$'inci eklemin dönme veya ötelenme ekseni doğrultusunda seçilir.\n"
                    "2. **$x_i$ ekseni:** $z_{i-1}$ ile $z_i$ eksenlerinin ortak dikmesi (common normal) boyunca ve $z_{i-1}$'den $z_i$'ye doğru yönlendirilir. Kesişen eksenlerde $x_i = z_{i-1} \\times z_i$ yönünde seçilir.\n"
                    "3. **$y_i$ ekseni:** Sağ el kuralı tamamlanacak şekilde $y_i = z_i \\times x_i$ olarak tanımlanır.\n\n"
                    "Dört temel DH parametresi şunlardır:\n"
                    "- **$a_i$ (Uzuv Uzunluğu - Link Length):** $z_{i-1}$ ile $z_i$ arasındaki mesafe ($x_i$ ekseni boyunca ölçülür).\n"
                    "- **$\\alpha_i$ (Uzuv Eğimi - Link Twist):** $z_{i-1}$ ekseninden $z_i$ eksenine olan açı ($x_i$ etrafında sağ el kuralı ile ölçülür).\n"
                    "- **$d_i$ (Uzuv Ötelemesi - Link Offset):** $x_{i-1}$ ile $x_i$ arasındaki mesafe ($z_{i-1}$ ekseni boyunca ölçülür, prizmatik eklemlerde değişkendir).\n"
                    "- **$\\theta_i$ (Eklem Açısı - Joint Angle):** $x_{i-1}$ ekseninden $x_i$ eksenine olan açı ($z_{i-1}$ etrafında sağ el kuralı ile ölçülür, döner eklemlerde değişkendir).\n\n"
                    "Komşu çerçeveler arasındaki $4 \\times 4$ homojen dönüşüm matrisi:\n"
                    "$$\n"
                    "^{i-1}T_i = \\begin{bmatrix}\n"
                    "\\cos\\theta_i & -\\sin\\theta_i\\cos\\alpha_i & \\sin\\theta_i\\sin\\alpha_i & a_i \\cos\\theta_i \\\\\n"
                    "\\sin\\theta_i & \\cos\\theta_i\\cos\\alpha_i & -\\cos\\theta_i\\sin\\alpha_i & a_i \\sin\\theta_i \\\\\n"
                    "0 & \\sin\\alpha_i & \\cos\\alpha_i & d_i \\\\\n"
                    "0 & 0 & 0 & 1\n"
                    "\\end{bmatrix}\n"
                    "$$\n"
                    "Taban koordinat sisteminden uç işlevciye (end-effector) kadar olan toplam ileri kinematik matrisi ardışık çarpımla elde edilir: "
                    "$$^0T_n = \\prod_{i=1}^{n} {^{i-1}T_i}$$\n"
                )
            },
            {
                "section_name": "İleri ve Ters Kinematik Analizi (FK / IK)",
                "content": (
                    "### İleri ve Ters Kinematik Analizi (FK ve IK)\n\n"
                    "**İleri Kinematik (Forward Kinematics - FK):**\n"
                    "Eklem değişkenleri $q = [\\theta_1, \\theta_2, \\dots, \\theta_n]^T$ bilindiğinde, uç işlevcinin Kartezyen uzaydaki konumu $P = [x, y, z]^T$ ve yönelimi $R \\in SO(3)$ homojen dönüşüm zinciriyle doğrudan ve tekil bir çözüm olarak bulunur:\n"
                    "$$X = f(q)$$\n\n"
                    "**Ters Kinematik (Inverse Kinematics - IK):**\n"
                    "İstenen hedef pozisyon ve yönelim $X_{hedef}$ verildiğinde, bunu sağlayacak eklem açıları $q = f^{-1}(X)$ çözülür. Ters kinematik doğrusal olmayan trigonometrik denklemler içerir; birden fazla geçerli çözüm (dirsek yukarı / dirsek aşağı), sonsuz çözüm veya tekillik durumunda çözümsüzlük görülebilir.\n"
                    "- **Analitik (Kapalı Form) Çözümler:** Geometrik yaklaşım ve Pieper Kriteri. Eğer robotun 3 ardışık döner eklem ekseni tek bir noktada kesişiyorsa (örneğin küresel bilek - spherical wrist yapısı), ters kinematik analitik olarak kesin biçimde çözülebilir. Bilek merkez noktası konumu önce bulunur ($P_w = P_{e} - d_6 R_{hedef} [0, 0, 1]^T$), ilk 3 eklem konumu, son 3 eklem ise Euler açıları ile yönelimi çözer.\n"
                    "- **Sayısal Çözümler (Newton-Raphson & Levenberg-Marquardt):** Jacobian tersi veya devriği kullanılarak iteratif güncellenir:\n"
                    "$$q_{k+1} = q_k + J^\\dagger(q_k) (X_{hedef} - f(q_k))$$\n"
                )
            },
            {
                "section_name": "Manipülatör Jacobian Matrisi, Hız Kinematiği ve Tekillik Analizi",
                "content": (
                    "### Diferansiyel Kinematik, Geometrik Jacobian ve Tekillikler\n\n"
                    "Manipülatör Jacobian matrisi $J(q) \\in \\mathbb{R}^{6 \\times n}$, eklem hız vektörü $\\dot{q}$ ile uç işlevcinin çizgisel ($v$) ve açısal ($\\omega$) hız vektörü arasındaki doğrusal bağıntıyı kurar:\n"
                    "$$\n"
                    "\\xi = \\begin{bmatrix} v \\\\ \\omega \\end{bmatrix} = J(q) \\dot{q}\n"
                    "$$\n"
                    "Her $i$'nci döner eklem için Jacobian sütunu:\n"
                    "$$J_i(q) = \\begin{bmatrix} z_{i-1} \\times (p_e - p_{i-1}) \\\\ z_{i-1} \\end{bmatrix}$$\n"
                    "Prizmatik eklemler için ise:\n"
                    "$$J_i(q) = \\begin{bmatrix} z_{i-1} \\\\ 0 \\end{bmatrix}$$\n\n"
                    "**Kinematik Tekillikler (Kinematic Singularities):**\n"
                    "Eğer $\\det(J(q)) = 0$ (kare Jacobian için) veya rankı düşerse ($\text{rank}(J) < 6$):\n"
                    "1. Manipülatör belirli yönlerde hareket serbestliğini kaybeder (o yönde sıfır hız üretilebilir).\n"
                    "2. İstenen son hız için gereken eklem hızları sonsuza yaklaşır ($\\dot{q} = J^{-1} \\xi \\to \\infty$).\n"
                    "3. Kartezyen kuvvet kontrolünde robot sonsuz mekanik avantaj kazanarak çevreye aşırı kontrolsüz kuvvet uygulayabilir.\n\n"
                    "Tekillik analizi için **Manipülabilirlik Elipsoidi (Yoshikawa):**\n"
                    "$$w = \\sqrt{\\det(J(q) J^T(q))}$$\n"
                    "$w \\to 0$ olduğunda robot tekillik sınırındadır. Çözüm olarak Damped Least Squares (Levenberg-Marquardt sönümlendirmesi) kullanılır: $J^* = J^T (J J^T + \\lambda^2 I)^{-1}$."
                )
            },
            {
                "section_name": "Robot Dinamiği: Euler-Lagrange ve Newton-Euler Formülasyonu",
                "content": (
                    "### Robot Dinamiği ve Hareket Denklemleri\n\n"
                    "Eklem torkları $\\tau$ ile ivmeler $\\ddot{q}$ arasındaki ilişki dinamik denklemlerle modellenir. "
                    "**Euler-Lagrange Formülasyonu:**\n"
                    "Lagrange fonksiyonu $L = K - P$ (Kinetik Enerji - Potansiyel Enerji):\n"
                    "$$\\frac{d}{dt} \\left( \\frac{\\partial L}{\\partial \\dot{q}} \\right) - \\frac{\\partial L}{\\partial q} = \\tau$$\n\n"
                    "Matrisel formda standart manipülatör dinamik denklemi:\n"
                    "$$\n"
                    "M(q)\\ddot{q} + C(q, \\dot{q})\\dot{q} + G(q) + F(\\dot{q}) = \\tau\n"
                    "$$\n"
                    "Burada:\n"
                    "- $M(q) \\in \\mathbb{R}^{n \\times n}$: Simetrik ve pozitif tanımlı kütle/eylemsizlik matrisi ($M(q) = M^T(q) > 0$).\n"
                    "- $C(q, \\dot{q})\\dot{q} \\in \\mathbb{R}^n$: Coriolis ve merkezkaç (centrifugal) kuvvetler vektörü. Matris türevi $\\dot{M}(q) - 2C(q, \\dot{q})$ ters-simetrik (skew-symmetric) olup enerjinin korunduğunu gösterir:\n"
                    "$$s^T (\\dot{M} - 2C) s = 0, \\quad \\forall s \\in \\mathbb{R}^n$$\n"
                    "- $G(q) \\in \\mathbb{R}^n$: Yerçekimi torkları vektörü ($G(q) = \\frac{\\partial P}{\\partial q}$).\n"
                    "- $F(\\dot{q})$: Viskoz ve Coulomb sürtünme terimleri ($F_v \\dot{q} + F_c \\text{sgn}(\\dot{q})$).\n\n"
                    "**Hesaplanmış Tork Kontrolü (Computed Torque Control / Inverse Dynamics):**\n"
                    "Doğrusal olmayan dinamikleri telafi etmek için geri beslemeli doğrusallaştırma uygulanır:\n"
                    "$$\\tau = M(q) \\left( \\ddot{q}_d + K_v (\\dot{q}_d - \\dot{q}) + K_p (q_d - q) \\right) + C(q, \\dot{q})\\dot{q} + G(q)$$\n"
                    "Bu kontrol kuralı altında hata dinamiği $\\ddot{e} + K_v \\dot{e} + K_p e = 0$ olur ve kritik sönümlü olarak sıfıra yakınsar."
                )
            }
        ]
    },
    {
        "id": "BTU-MEK-KONTROL",
        "title": "Modern Otomatik Kontrol Sistemleri ve Durum-Uzay Analizi",
        "authors": "Katsuhiko Ogata & Gene F. Franklin",
        "year": 2024,
        "domain": "BTÜ Mekatronik / Kontrol Teorisi",
        "color": "#10b981",
        "desc": "BTÜ Kontrol dersi ana kaynakları: Transfer fonksiyonları, Durum-Uzay modelleme, Kök-Yer Eğrisi, Bode/Nyquist frekans analizi, PID tasarımı ve Durum Geri Beslemesi (State Feedback / Luenberger Observer).",
        "prompt": "BTÜ Kontrol dersi kapsamında Durum-Uzay (State-Space) kutup yerleştirme tasarımı ve Kalman Rank ile kontrol edilebilirlik şartını matematiksel olarak açıkla.",
        "sections": [
            {
                "section_name": "Durum-Uzay Modellemesi ve Transfer Fonksiyonu Dönüşümü",
                "content": (
                    "### Durum-Uzay (State-Space) Temsili ve LTI Sistemler\n\n"
                    "Çok girişli çok çıkışlı (MIMO) veya doğrusal zamanla değişmeyen (LTI) mekatronik sistemler 1. derece diferansiyel denklem takımı olarak ifade edilir:\n"
                    "$$\n"
                    "\\dot{x}(t) = A x(t) + B u(t)\n"
                    "$$\n"
                    "$$\n"
                    "y(t) = C x(t) + D u(t)\n"
                    "$$\n"
                    "Burada $x(t) \\in \\mathbb{R}^n$ durum vektörü (örneğin konum ve hız), $u(t) \\in \\mathbb{R}^m$ kontrol giriş vektörü (örneğin motor gerilimi), $y(t) \\in \\mathbb{R}^p$ ölçülen çıkış vektörüdür.\n\n"
                    "**Durum-Uzayından Transfer Fonksiyonuna Geçiş:**\n"
                    "Laplace dönüşümü alınarak sıfır başlangıç şartlarında:\n"
                    "$$s X(s) = A X(s) + B U(s) \\implies X(s) = (sI - A)^{-1} B U(s)$$\n"
                    "$$Y(s) = C X(s) + D U(s) = \\left[ C(sI - A)^{-1} B + D \\right] U(s)$$\n"
                    "$$G(s) = \\frac{Y(s)}{U(s)} = C(sI - A)^{-1} B + D = \\frac{C \\text{adj}(sI - A) B}{\\det(sI - A)} + D$$\n"
                    "Sistemin karakteristik denklemi ve özdeğerleri (kutupları) $\\det(sI - A) = 0$ kökleridir."
                )
            },
            {
                "section_name": "Kontrol Edilebilirlik, Gözlenebilirlik ve Kutup Yerleştirme",
                "content": (
                    "### Kontrol Edilebilirlik (Controllability) ve Gözlenebilirlik (Observability)\n\n"
                    "**1. Kalman Kontrol Edilebilirlik Matrisi $(\\mathcal{C})$:**\n"
                    "Sistemin durumlarının sonlu bir sürede istenen herhangi bir değere sürülebilmesi için kontrol edilebilirlik matrisinin tam ranka sahip olması gerekir:\n"
                    "$$\\mathcal{C} = \\begin{bmatrix} B & AB & A^2 B & \\dots & A^{n-1} B \\end{bmatrix} \\in \\mathbb{R}^{n \\times nm}$$\n"
                    "$$\\text{rank}(\\mathcal{C}) = n \\iff \\text{Sistem tamamen kontrol edilebilirdir.}$$\n\n"
                    "**2. Kalman Gözlenebilirlik Matrisi $(\\mathcal{O})$:**\n"
                    "Sistem çıkışlarından $y(t)$ durumların tamamının $x(t)$ kestirilebilmesi için:\n"
                    "$$\\mathcal{O} = \\begin{bmatrix} C \\\\ CA \\\\ CA^2 \\\\ \\vdots \\\\ CA^{n-1} \\end{bmatrix} \\in \\mathbb{R}^{np \\times n}$$\n"
                    "$$\\text{rank}(\\mathcal{O}) = n \\iff \\text{Sistem tamamen gözlenebilirdir.}$$\n\n"
                    "**3. Durum Geri Beslemesi ve Kutup Yerleştirme (Ackermann Formülü):**\n"
                    "Kontrol edilebilir bir sistemde durum geri besleme kazancı $u = -Kx + r$ tasarlanarak kapalı çevrim sistemi $\\dot{x} = (A - BK)x + Br$ elde edilir. İstenen karakteristik polinom $\\alpha_c(s) = s^n + \\alpha_1 s^{n-1} + \\dots + \\alpha_n$ olmak üzere:\n"
                    "$$K = \\begin{bmatrix} 0 & 0 & \\dots & 1 \\end{bmatrix} \\mathcal{C}^{-1} \\alpha_c(A)$$\n\n"
                    "**4. Luenberger Gözlemcisi (State Observer):**\n"
                    "Durumların doğrudan ölçülemediği durumlarda durum kestiricisi $\\dot{\\hat{x}} = A\\hat{x} + Bu + L(y - C\\hat{x})$ tasarlanır. Kestirim hatası $\\dot{e} = (A - LC)e$ olup $L$ kazancı özdeğerleri sol yarı düzlemde hızla sıfıra gidecek şekilde seçilir."
                )
            },
            {
                "section_name": "Frekans Cevabı: Bode Diyagramları, Faz/Kazanç Payı ve Nyquist Kararlılığı",
                "content": (
                    "### Frekans Domeni Analizi ve Kararlılık Kriterleri\n\n"
                    "Sisteme harmonik giriş $u(t) = U_0 \\sin(\\omega t)$ uygulandığında kalıcı durum çıkışı genlik ve faz farkıyla $y_{ss}(t) = Y_0 \\sin(\\omega t + \\phi)$ olur:\n"
                    "$$G(j\\omega) = |G(j\\omega)| e^{j\\angle G(j\\omega)}$$\n"
                    "- Genlik (dB): $M_{dB} = 20 \\log_{10} |G(j\\omega)|$\n"
                    "- Faz (derece): $\\phi = \\arg(G(j\\omega))$\n\n"
                    "**Kazanç Payı (Gain Margin - GM) ve Faz Payı (Phase Margin - PM):**\n"
                    "- **Kazanç Geçiş Frekansı (Gain Crossover Frequency - $\\omega_{gc}$):** Genliğin $0\\text{ dB}$ olduğu frekans ($|G(j\\omega_{gc})| = 1$).\n"
                    "- **Faz Geçiş Frekansı (Phase Crossover Frequency - $\\omega_{pc}$):** Fazın $-180^\\circ$ olduğu frekans ($\\angle G(j\\omega_{pc}) = -180^\\circ$).\n"
                    "- **Faz Payı:** $\\text{PM} = 180^\\circ + \\angle G(j\\omega_{gc})$ (Kararlı ve iyi sönümlü bir sistem için tipik olarak $45^\\circ \\le \\text{PM} \\le 65^\\circ$ istenir).\n"
                    "- **Kazanç Payı:** $\\text{GM} = -20 \\log_{10} |G(j\\omega_{pc})| \\text{ dB}$.\n\n"
                    "**Nyquist Kararlılık Kriteri:**\n"
                    "Kapalı çevrim kararsız kutup sayısı $Z = N + P$. Burada $P$ açık çevrim kararsız kutup sayısı, $N$ ise açık çevrim Nyquist eğrisinin $(-1, j0)$ kritik noktasını saat yönünün tersine dolanma (encirclement) sayısıdır. Sistemin kapalı çevrimde kararlı olması için $Z = 0$, dolayısıyla $N = -P$ olmalıdır."
                )
            },
            {
                "section_name": "Klasik PID Tasarımı, Anti-Windup ve Ziegler-Nichols Ayar Yöntemleri",
                "content": (
                    "### PID Kontrolör Tasarımı ve Ayrıklaştırma\n\n"
                    "Standart PID kontrolör transfer fonksiyonu:\n"
                    "$$u(t) = K_p e(t) + K_i \\int_0^t e(\\tau) d\\tau + K_d \\frac{de(t)}{dt}$$\n"
                    "$$C(s) = K_p \\left( 1 + \\frac{1}{T_i s} + T_d s \\right)$$\n\n"
                    "- **$K_p$ (Oransal Kazanç):** Yükselme zamanını ($t_r$) düşürür, hatayı azaltır ancak aşımı (overshoot) artırabilir.\n"
                    "- **$K_i$ (İntegral Kazanç):** Kalıcı durum hatasını (steady-state error) tamamen sıfırlar ancak sistem fazını $90^\\circ$ geciktirerek salınıma ve kararsızlığa yol açabilir.\n"
                    "- **$K_d$ (Türevsel Kazanç):** Sistem sönümünü artırır, aşımı düşürür ve gelecekteki eğilimi öngörür. Ancak yüksek frekanslı ölçüm gürültüsünü aşırı yükselttiği için pratikte alçak geçiren filtreyle ($N / (1 + N/s)$) birlikte kullanılır.\n\n"
                    "**İntegral Windup ve Anti-Windup Stratejisi:**\n"
                    "Aktüatör (motor gerilimi vb.) doyuma ulaştığında (saturation: $u_{sat} \\ne u$), hata integrali şişmeye devam eder (windup). Hata işaret değiştirdiğinde sistem hemen toparlanamaz ve aşırı aşım yapar. Anti-windup clamping (integratör durdurma) veya back-calculation (geri hesaplama) ile integratör deşarj edilir."
                )
            }
        ]
    },
    {
        "id": "BTU-MEK-SENSOR-AKTUATOR",
        "title": "Mekatronik Sensörler, Aktüatörler ve Sinyal Şartlandırma",
        "authors": "David G. Alciatore & Michael B. Histand",
        "year": 2024,
        "domain": "BTÜ Mekatronik / Sensör & Aktüatör",
        "color": "#f59e0b",
        "desc": "BTÜ Sensör & Aktüatör dersi ana kaynakları: Artımsal/mutlak enkoderler, IMU sensör füzyonu, BLDC motor 6-adımlı komütasyonu, H-Köprüsü PWM sürüşü ve Op-Amp sinyal şartlandırma.",
        "prompt": "Fırçasız DC (BLDC) motorlarda Hall sensörleri ile 6 adımlı komütasyon mantığını ve H-köprüsü PWM anahtarlama mimarisini açıkla.",
        "sections": [
            {
                "section_name": "Enkoderler, LVDT ve Gerinim Pulları (Wheatstone Köprüsü)",
                "content": (
                    "### Konum ve Kuvvet Sensörleri\n\n"
                    "**1. Optik ve Manyetik Enkoderler:**\n"
                    "- **Artımsal (Incremental) Enkoder:** Kanal A ve Kanal B sinyalleri $90^\\circ$ faz farkıyla (quadrature) üretilir. A'nın B'den önde veya geride olması dönüş yönünü belirler. 4x kuadratür kod çözme (quadrature decoding) ile her iki kanalın yükselen ve düşen kenarları sayılarak çözünürlük 4 katına ($4 \\times CPR$) çıkarılır. Z (indeks) kanalı her turda 1 pals vererek sıfırlama referansı sağlar.\n"
                    "- **Mutlak (Absolute) Enkoder:** Gray kodu diski kullanılarak güç kesilip açılsa dahi mutlak açısal konum anında okunur. Gray kodunda ardışık durumlar arasında yalnızca tek 1 bit değişir; böylece geçiş anlarındaki belirsizlikler (glitch) engellenir.\n\n"
                    "**2. Gerinim Pulu (Strain Gauge) ve Wheatstone Köprüsü:**\n"
                    "Mekanik deformasyon direnç değişimine dönüşür: $\\frac{\\Delta R}{R} = GF \\cdot \\epsilon$ ($GF$: Gauge Factor, $\\epsilon$: Birim uzama).\n"
                    "Küçük direnç değişimlerini hassas gerilime dönüştürmek için Wheatstone köprüsü kullanılır:\n"
                    "$$V_{out} = V_s \\left( \\frac{R_1}{R_1 + R_2} - \\frac{R_4}{R_3 + R_4} \\right)$$\n"
                    "Tam köprü (Full-bridge) bağlantıda 4 gerinim pulu kullanılarak sıcaklık kaymaları dengelenir ve duyarlılık 4 katına çıkar."
                )
            },
            {
                "section_name": "MEMS IMU: İvmeölçer, Jiroskop ve Tamamlayıcı Filtre (Complementary Filter)",
                "content": (
                    "### 6-DoF Ataletsel Ölçüm Birimi (IMU) ve Sensör Füzyonu\n\n"
                    "**1. MEMS İvmeölçer (Accelerometer):** Kütle-yay sistemi prensibiyle yerçekimi ivmesi dahil çizgisel ivmeyi ölçer. Statik durumda eğim açısını verir ($ roll = \\arctan2(a_y, a_z) $). Ancak dinamik hareketlerdeki titreşim ve ivmelerden yüksek frekanslı gürültü içerir.\n"
                    "**2. MEMS Jiroskop (Gyroscope):** Coriolis ivmesi prensibiyle açısal hızları ($\\omega$) ölçer. Açısal konum için integral alınır: $\\theta(t) = \\int \\omega(t) dt$. Kısa vadede çok hassastır ancak integrasyon nedeniyle sabit bir sapma (drift/bias) birikerek uzun vadede kaymaya neden olur.\n\n"
                    "**3. Tamamlayıcı Filtre (Complementary Filter):**\n"
                    "Düşük frekanslı ivmeölçer bilgisi ile yüksek frekanslı jiroskop integralini birleştirir:\n"
                    "$$\n"
                    "\\theta_{t} = \\alpha \\left( \\theta_{t-1} + \\omega_t \\Delta t \\right) + (1 - \\alpha) \\theta_{acc}\n"
                    "$$\n"
                    "Burada $\\alpha = \\frac{\\tau}{\\tau + \\Delta t} \\approx 0.98$ olarak seçilir. Jiroskopun drift yapması engellenirken ivmeölçerin titreşim gürültüsü süzülür."
                )
            },
            {
                "section_name": "Fırçasız DC (BLDC) Motorlar, Hall Sensörleri ve 6-Adımlı Komütasyon",
                "content": (
                    "### Fırçasız DC (BLDC) Motor Komütasyonu ve Sürüşü\n\n"
                    "BLDC motorlarda rotor kalıcı mıknatıs, stator ise 3 fazlı (U, V, W) sargılardan oluşur. Mekanik fırça/kolektör yerine elektronik anahtarlama yapılır.\n\n"
                    "**1. Hall Etkisi Sensörleri ve Rotor Pozisyonu:**\n"
                    "120 elektriksel derece aralıkla yerleştirilmiş 3 Hall sensörü ($H_1, H_2, H_3$) rotor kutuplarını algılayarak 6 farklı mantıksal durum (sektör: 1'den 6'ya) üretir.\n\n"
                    "**2. Üç Fazlı H-Köprüsü (Six-Step / Trapezoidal Commutation):**\n"
                    "6 adet MOSFET/IGBT anahtarı (3 üst - High Side, 3 alt - Low Side) kullanılır:\n"
                    "- Her 60 elektriksel derecelik adımda 1 faz pozitif beslemeye ($+V_{dc}$), 1 faz toprağa (GND) çekilir, 3. faz ise boşta (floating) bırakılır.\n"
                    "- Boşta kalan fazda rotasyon sonucu Zıt EMF (Back-EMF) endüklenir. Bu gerilimin sıfır geçiş noktası (Zero-Crossing Detection - ZCD) algılanarak sensörsüz (sensorless) sürüş de yapılabilir.\n\n"
                    "**3. Alan Yönelimli Kontrol (Field Oriented Control - FOC):**\n"
                    "Clarke ve Park dönüşümleri ile 3 fazlı AC akımlar ($i_a, i_b, i_c$), dönen rotor referans çerçevesinde DC büyüklüklere dönüştürülür:\n"
                    "- $i_d$ (Doğrudan eksen akımı): Manyetik akıyı oluşturur, genellikle sıfıra regüle edilir ($i_d^* = 0$).\n"
                    "- $i_q$ (Karesel eksen akımı): Tork üreten bileşendir ($\\tau = \\frac{3}{2} p \\lambda_m i_q$). Böylece motor maksimum verim ve minimum tork dalgalanması ile sürülür."
                )
            },
            {
                "section_name": "Güç Elektroniği Anahtarlama ve Op-Amp Sinyal Şartlandırma",
                "content": (
                    "### Güç Elektroniği ve Analog Sinyal Şartlandırma\n\n"
                    "**1. MOSFET/IGBT Sürüşü ve Dead-Time (Ölü Zaman):**\n"
                    "Aynı köprü kolundaki üst ve alt MOSFET'lerin aynı anda iletime geçmesi kısa devreye (shoot-through) neden olarak transistörleri patlatır. Bunu önlemek için donanımsal veya yazılımsal 'Dead-Time' gecikmesi (tipik 500ns - 2µs) uygulanır.\n"
                    "İndüktif yüklerin (motor sargıları) enerjisini boşaltmak için ters bağlı hızlı geri besleme diyotları (freewheeling diodes) gereklidir.\n\n"
                    "**2. Enstrümantasyon Yükselteci (Instrumentation Amplifier):**\n"
                    "Düşük seviyeli sensör sinyallerini (örneğin mV mertebesindeki gerinim pulu) yükseltmek ve ortak mod gürültüsünü (CMRR) yok etmek için 3 Op-Amp'lı enstrümantasyon yükselteci kullanılır:\n"
                    "$$V_{out} = \\left( 1 + \\frac{2 R_1}{R_G} \\right) \\frac{R_3}{R_2} (V_2 - V_1)$$\n"
                    "Tek bir $R_G$ direnci ayarlanarak sistem kazancı hassasça belirlenir."
                )
            }
        ]
    },
    {
        "id": "BTU-MEK-SINYAL-DSP",
        "title": "Sinyaller, Sistemler ve Sayısal Sinyal İşleme (DSP)",
        "authors": "Alan V. Oppenheim & Alan S. Willsky",
        "year": 2024,
        "domain": "BTÜ Mekatronik / Sinyal & DSP",
        "color": "#8b5cf6",
        "desc": "BTÜ Sinyaller & DSP dersi ana kaynakları: LTI sistem konvolüsyonu, Fourier ve Z-dönüşümü, Nyquist örnekleme teoremi, FIR/IIR sayısal filtreler ve robotik sensörler için 1D Kalman Filtresi.",
        "prompt": "Ayrık zamanlı Kalman filtresinin kestirim (predict) ve güncelleme (update) matris denklemlerini bir robotik odometri örneğiyle adım adım göster.",
        "sections": [
            {
                "section_name": "Doğrusal Zamanla Değişmeyen (LTI) Sistemler ve Konvolüsyon",
                "content": (
                    "### LTI Sistemler ve Konvolüsyon İşlemi\n\n"
                    "Bir sistem doğrusallık (süperpozisyon) ve zamanla değişmezlik özelliklerini sağlıyorsa LTI (Linear Time-Invariant) sistemdir. LTI bir sistem birim darbe cevabı (impulse response) $h(t)$ veya $h[n]$ ile tamamen karakterize edilir.\n\n"
                    "**Sürekli Zaman Konvolüsyon İntegrali:**\n"
                    "$$y(t) = x(t) * h(t) = \\int_{-\\infty}^{\\infty} x(\\tau) h(t - \\tau) d\\tau$$\n\n"
                    "**Ayrık Zaman Konvolüsyon Toplamı:**\n"
                    "$$y[n] = x[n] * h[n] = \\sum_{k=-\\infty}^{\\infty} x[k] h[n - k]$$\n\n"
                    "Frekans domeninde konvolüsyon basit bir çarpmaya dönüşür: $\\mathcal{F}\\{x * h\\} = X(j\\omega) H(j\\omega)$. Bu özellik sayısal filtrelerin verimli hesaplanmasında FFT (Hızlı Fourier Dönüşümü) tabanlı hızlı konvolüsyonun temelidir."
                )
            },
            {
                "section_name": "Örnekleme Teoremi (Nyquist-Shannon) ve Aliasing",
                "content": (
                    "### Örnekleme Teoremi ve Örtüşme (Aliasing)\n\n"
                    "Sürekli bir analog sinyalin $x(t)$ bilgi kaybı olmaksızın ayrık zamanlı diziye $x[n] = x(n T_s)$ dönüştürülmesi ve kusursuz şekilde geri çatılabilmesi (reconstruction) için örnekleme frekansı $f_s$, sinyalin içerdiği en yüksek frekans bileşeninin ($f_{max}$) en az iki katı olmalıdır:\n"
                    "$$f_s \\ge 2 f_{max} \\quad (\\text{Nyquist Oranı})$$\n\n"
                    "Eğer $f_s < 2 f_{max}$ ise, yüksek frekanslı bileşenler alçak frekans bölgesine katlanarak sahte (hayali) alçak frekanslı sinyaller oluşturur (Aliasing / Örtüşme). "
                    "Bunu önlemek için ADC (Analog-Digital Converter) girişinden önce kesim frekansı $f_c \\le f_s / 2$ olan analog donanımsal bir **Anti-Aliasing Filtresi** (Alçak Geçiren Filtre) konulması zorunludur."
                )
            },
            {
                "section_name": "Sayısal Filtre Tasarımı: FIR ve IIR Filtreler",
                "content": (
                    "### Sayısal Filtre Mimarileri: FIR vs. IIR\n\n"
                    "**1. Sonlu Darbe Cevaplı (FIR - Finite Impulse Response) Filtreler:**\n"
                    "$$y[n] = \\sum_{k=0}^{M} b_k x[n - k]$$\n"
                    "- Yalnızca mevcut ve geçmiş girişlere bağlıdır, geri besleme yoktur (kutupları yalnızca $z=0$'dadır).\n"
                    "- **Avantajları:** Her zaman koşulsuz kararlıdır. Simetrik katsayılar seçilerek kusursuz **Doğrusal Faz (Linear Phase)** elde edilebilir (tüm frekans bileşenleri eşit süre gecikir, sinyal şekli bozulmaz).\n"
                    "- **Dezavantajları:** Keskin frekans geçiş bandı elde etmek için yüksek filtre derecesi ($M > 50-100$) gerekir, bu da gömülü işlemcide hesaplama yükünü artırır.\n\n"
                    "**2. Sonsuz Darbe Cevaplı (IIR - Infinite Impulse Response) Filtreler:**\n"
                    "$$y[n] = \\frac{1}{a_0} \\left( \\sum_{k=0}^{M} b_k x[n-k] - \\sum_{k=1}^{N} a_k y[n-k] \\right)$$\n"
                    "- Geçmiş çıkışların geri beslemesini içerir ($z$-domeninde sıfırlar ve kutuplar vardır).\n"
                    "- **Avantajları:** Çok daha düşük dereceyle ($N = 2$ ila $6$) çok keskin kesim frekansları sağlar, işlemci yükü azdır.\n"
                    "- **Dezavantajları:** Kutuplar birim çemberin dışına çıkarsa kararsızlaşabilir. Faz cevabı doğrusal değildir."
                )
            },
            {
                "section_name": "Durum Kestirimi ve Ayrık Zamanlı Kalman Filtresi",
                "content": (
                    "### Ayrık Zamanlı Doğrusal Kalman Filtresi (Linear Kalman Filter)\n\n"
                    "Sensör gürültüsü ($R$) ve sistem dinamik belirsizliği ($Q$) içeren doğrusal bir sistemin optimum durum kestirimidir. İki adımdan oluşur:\n\n"
                    "**1. Kestirim (Prediction / Zaman Güncellemesi):**\n"
                    "Fiziksel model kullanılarak durum ve hata kovaryansı ileriye ötelenir:\n"
                    "$$\\hat{x}_{k|k-1} = A \\hat{x}_{k-1|k-1} + B u_k$$\n"
                    "$$P_{k|k-1} = A P_{k-1|k-1} A^T + Q$$\n\n"
                    "**2. Güncelleme (Correction / Ölçüm Güncellemesi):**\n"
                    "Sensörden gelen ölçüm $z_k$ kullanılarak kestirim düzeltilir:\n"
                    "İnovasyon (ölçüm artığı): $\\tilde{y}_k = z_k - H \\hat{x}_{k|k-1}$\n"
                    "İnovasyon kovaryansı: $S_k = H P_{k|k-1} H^T + R$\n"
                    "Optimum Kalman Kazancı: $K_k = P_{k|k-1} H^T S_k^{-1}$\n"
                    "Güncellenmiş durum: $\\hat{x}_{k|k} = \\hat{x}_{k|k-1} + K_k \\tilde{y}_k$\n"
                    "Güncellenmiş kovaryans: $P_{k|k} = (I - K_k H) P_{k|k-1}$\n\n"
                    "Sensör çok gürültülü olduğunda ($R \\to \\infty$), $K_k \\to 0$ olur ve filtre sadece modele güvenir. Model çok belirsiz olduğunda ($Q \\to \\infty$), $K_k \\to H^{-1}$ olur ve filtre sensör ölçümünü kabul eder."
                )
            }
        ]
    },
    {
        "id": "BTU-MEK-GOMULU-STM32",
        "title": "Gömülü Sistemler, ARM Cortex-M ve CAN Bus Mimarisi",
        "authors": "Joseph Yiu & STMicroelectronics",
        "year": 2024,
        "domain": "BTÜ Mekatronik / Gömülü Sistemler",
        "color": "#ec4899",
        "desc": "BTÜ Gömülü Sistemler dersi ana kaynakları: ARM Cortex-M4 çekirdeği, NVIC kesme kontrolcüsü, DMA bellek aktarımı, Timer PWM, SPI/I2C/UART ve CAN 2.0B / CAN-FD endüstriyel haberleşmesi.",
        "prompt": "STM32 mikrodenetleyicilerde CAN Bus 2.0B bit arbitrasyonunu, diferansiyel sinyalleşmeyi ve NVIC kesme önceliklendirmesini teknik olarak açıkla.",
        "sections": [
            {
                "section_name": "ARM Cortex-M4 Mimarisi, Bellek Haritası ve Registerlar",
                "content": (
                    "### ARM Cortex-M4 Çekirdek Mimarisi ve Bellek Organizasyonu\n\n"
                    "ARM Cortex-M4, Harvard mimarisine dayalı 32-bit RISC işlemcidir. Thumb-2 komut seti, donanımsal tek-çevrim FPU (Floating Point Unit) ve DSP komut uzantıları içerir.\n\n"
                    "**1. Çekirdek Register Yapısı:**\n"
                    "- $R_0 - R_{12}$: Genel amaçlı veri saklayıcılar (General Purpose Registers).\n"
                    "- $R_{13}$ (SP - Stack Pointer): Main Stack Pointer (MSP) ve Process Stack Pointer (PSP) olarak iki ayrı yığın işaretçisi içerir.\n"
                    "- $R_{14}$ (LR - Link Register): Fonksiyon çağrılarında veya kesmelerde geri dönüş adresini saklar.\n"
                    "- $R_{15}$ (PC - Program Counter): Çalıştırılacak bir sonraki komutun adresini tutar.\n"
                    "- xPSR (Program Status Register): APSR (Aritmetik bayraklar: N, Z, C, V), IPSR (Aktif kesme numarası) ve EPSR içerir.\n\n"
                    "**2. Bellek Haritası (Memory Map):**\n"
                    "Toplam 4 GB doğrusal adresleme alanı standart ARM segmentlerine bölünmüştür:\n"
                    "- `0x0000_0000 - 0x1FFF_FFFF`: Flash / Boot ROM bellek alanı (Kod ve Vektör Tablosu).\n"
                    "- `0x2000_0000 - 0x3FFF_FFFF`: Dahili SRAM bellek alanı (Değişkenler ve yığın).\n"
                    "- `0x4000_0000 - 0x5FFF_FFFF`: Periferaller (APB1, APB2, AHB veri yolu çevre birimleri: Timers, ADC, USART, CAN, SPI).\n"
                    "- `0xE000_0000 - 0xE00F_FFFF`: Sistem Kontrol Alanı (NVIC, SysTick, MPU, SCB)."
                )
            },
            {
                "section_name": "NVIC Kesme Yönetimi ve Doğrudan Bellek Erişimi (DMA)",
                "content": (
                    "### İç İçe Vektörlü Kesme Kontrolcüsü (NVIC) ve DMA Mimarisi\n\n"
                    "**1. NVIC (Nested Vectored Interrupt Controller):**\n"
                    "- Donanımsal önceliklendirme ve düşük kesme gecikmesi (12 çevrim) sağlar.\n"
                    "- **Preemption Priority (Öncelikli Kesme):** Yüksek öncelikli bir kesme, o anda çalışan düşük öncelikli bir kesmeyi bölerek (preempt) hemen devreye girer.\n"
                    "- **Subpriority (Alt Öncelik):** Aynı anda iki kesme tetiklendiğinde hangisinin önce çalışacağını belirler, ancak çalışan kesmeyi bölemez.\n"
                    "- Kesme tetiklendiğinde donanım otomatik olarak $R_0-R_3, R_{12}, LR, PC, xPSR$ registerlarını yığına (Stack) iter (Hardware Stacking).\n\n"
                    "**2. DMA (Direct Memory Access) Kontrolcüsü:**\n"
                    "CPU'yu hiçbir döngüde meşgul etmeden çevre birimleri (ADC, SPI, UART, Timer) ile SRAM/Flash arasında yüksek hızlı blok veri transferi sağlar.\n"
                    "- **Dairesel Mod (Circular Mode):** Özellikle sensör okumalarında veya ADC çok kanallı taramada tampon dolduğunda başa dönerek kesintisiz akış sağlar.\n"
                    "- **Çift Tampon Modu (Double Buffer):** Bir tampon DMA tarafından doldurulurken diğeri CPU tarafından işlenir; böylece yarış durumu (race condition) ve veri kaybı önlenir."
                )
            },
            {
                "section_name": "CAN Bus (CAN 2.0B ve CAN-FD) Protokolü ve Bit Arbitrasyonu",
                "content": (
                    "### Endüstriyel Haberleşme: Controller Area Network (CAN Bus)\n\n"
                    "Otomotiv ve robotik sistemlerde elektromanyetik parazitlere (EMI) dayanıklı, çok efendili (multi-master) ve mesaj öncelikli diferansiyel seri haberleşme protokolüdür.\n\n"
                    "**1. Diferansiyel Fiziksel Katman (Physical Layer):**\n"
                    "İki hatlı bükümlü kablo ($CAN\\_H$ ve $CAN\\_L$) ve hat sonlarında yansımaları sönümleyen $120\\,\\Omega$ hat sonlandırma dirençleri kullanılır:\n"
                    "- **Resesif (Çekinik - Mantık 1):** Her iki hat da $2.5\\text{ V}$ seviyesindedir ($V_{diff} = CAN\\_H - CAN\\_L = 0\\text{ V}$).\n"
                    "- **Dominant (Baskın - Mantık 0):** $CAN\\_H = 3.5\\text{ V}$, $CAN\\_L = 1.5\\text{ V}$ ($V_{diff} = 2.0\\text{ V}$). Dominant bit her zaman çekinik biti ezer.\n\n"
                    "**2. Kayıpsız Çarpışma Çözümü (CSMA/CD + Non-Destructive Bitwise Arbitration):**\n"
                    "Aynı anda birden fazla düğüm hatta mesaj göndermeye başladığında, her düğüm kendi gönderdiği biti hattan geri okur:\n"
                    "- Mesaj ID'si düşük olan mesaj (örneğin acil fren mesajı: `0x001`), ID bitlerinde daha erken '0' (dominant) basar.\n"
                    "- Çekinik '1' basarken hattan dominant '0' okuyan düğüm arbitrasyonu kaybettiğini anlar ve mesajını hemen keserek dinleme moduna geçer.\n"
                    "- Böylece en yüksek öncelikli mesaj sıfır gecikmeyle ve veri bozulması olmadan iletilir.\n\n"
                    "**3. CAN 2.0B vs. CAN-FD:**\n"
                    "- Standart CAN 2.0B: Maksimum $1\\text{ Mbps}$ hız ve çerçeve başına maksimum 8 bayt veri taşır.\n"
                    "- CAN-FD (Flexible Data-Rate): Veri alanında bit hızını $5-8\\text{ Mbps}$ seviyesine çıkarır ve yük kapasitesini çerçeve başına 64 bayta yükseltir."
                )
            }
        ]
    },
    {
        "id": "BTU-MEK-ROS2",
        "title": "Robot İşletim Sistemi (ROS2 Humble) ve Otonom Mobil Robotlar",
        "authors": "Steven Macenski, Tully Foote & Open Robotics",
        "year": 2024,
        "domain": "BTÜ Mekatronik / Robotik Yazılım & ROS2",
        "color": "#06b6d4",
        "desc": "BTÜ Robotik Yazılım dersi ana kaynakları: DDS haberleşme omurgası, ROS2 Nodes/Topics/Actions, URDF/Xacro robot modelleme, TF2 koordinat dönüşümleri ve Nav2 otonom navigasyon yığını.",
        "prompt": "ROS2 mimarisinde TF2 koordinat dönüşüm kütüphanesi, URDF robot modeli ve Nav2 Costmap entegrasyonu otonom sürüşte nasıl birlikte çalışır?",
        "sections": [
            {
                "section_name": "ROS2 Mimarisi, DDS Omurgası ve QoS Politikaları",
                "content": (
                    "### ROS2 Haberleşme Altyapısı ve Data Distribution Service (DDS)\n\n"
                    "ROS 1'in tekil merkezi master (`roscore`) bağımlılığı kaldırılarak endüstriyel standart OMG DDS (Data Distribution Service) üzerine inşa edilmiştir. "
                    "Bu sayede sistem dağıtık, sıfır tek nokta arızalı (no single point of failure) ve gerçek zamanlı çalışabilir.\n\n"
                    "**Haberleşme Primitifleri:**\n"
                    "1. **Topics (Konular):** Çoktan-çoğa tek yönlü veri akışı (Publish / Subscribe). Sensör akışları ve telemetri için kullanılır.\n"
                    "2. **Services (Servisler):** Bire-bir eşzamanlı istek/cevap (Request / Response). Parametre yapılandırma veya anlık hesaplamalar için uygundur.\n"
                    "3. **Actions (Eylemler):** İstemci-sunucu tabanlı, uzun süren, iptal edilebilir ve ara geri bildirim (feedback) sağlayan hedefler (örneğin otonom navigasyonda hedef noktaya gitme).\n\n"
                    "**Quality of Service (QoS) Profilleri:**\n"
                    "- **Reliability (Güvenilirlik):** `BEST_EFFORT` (Kayıp paketleri umursamaz, düşük gecikme; kamera/LiDAR akışları için ideal) veya `RELIABLE` (TCP benzeri onay mekanizması, kritik komutlar için).\n"
                    "- **Durability (Kalıcılık):** `VOLATILE` (Sadece o anki abonelere gider) veya `TRANSIENT_LOCAL` (Son mesaj saklanır, yeni abone bağlandığında ona hemen iletilir; örneğin `/map` harita verisi)."
                )
            },
            {
                "section_name": "URDF, Xacro ve TF2 Koordinat Dönüşüm Kütüphanesi",
                "content": (
                    "### Robot Modelleme ve Koordinat Dönüşümleri (TF2)\n\n"
                    "**1. URDF (Unified Robot Description Format) ve Xacro:**\n"
                    "Robotun 3B kinematik ve dinamik ağacı XML tabanlı tanımlanır:\n"
                    "- `<link>`: Kütle ($m$), eylemsizlik tensörü ($I_{xx}, I_{yy}, \\dots$), görsel mesh (visual) ve çarpışma geometrisi (collision).\n"
                    "- `<joint>`: İki link arasındaki ilişki (revolute, continuous, prismatic, fixed). Eksen ve hareket limitleri tanımlanır.\n"
                    "- Xacro, makrolar ve matematiksel ifadeler ekleyerek kod tekrarını önler.\n\n"
                    "**2. TF2 Dönüşüm Ağacı (Transform Tree):**\n"
                    "Otonom robotlarda birden fazla sensör farklı konumlara monte edilmiştir. TF2, zaman damgalı dönüşümleri ($T_A^B(t)$) tamponlayarak koordinat dönüşümlerini yönetir:\n"
                    "Standart ROS koordinat hiyerarşisi:\n"
                    "`map` $\\longrightarrow$ `odom` $\\longrightarrow$ `base_link` $\\longrightarrow$ `laser_frame` / `camera_link`\n"
                    "- `map -> odom`: Global yerelleştirme (AMCL veya SLAM) tarafından yayınlanan, kaymasız ancak kesintili sıçramalar yapabilen dönüşüm.\n"
                    "- `odom -> base_link`: Tekerlek odometrisi ve IMU tarafından yüksek frekansta yayınlanan, sürekli ancak zamanla drift yapan yerel dönüşüm.\n"
                    "- `base_link -> sensor`: URDF tarafından robot gövdesine göre sabit tanımlanan statik dönüşüm."
                )
            },
            {
                "section_name": "Nav2 Navigasyon Yığını: Costmap2D, AMCL ve Yol Planlama",
                "content": (
                    "### ROS2 Nav2 Otonom Navigasyon Yığını\n\n"
                    "Mobil robotun bir harita üzerinde engellere çarpmadan hedef koordinata gitmesini sağlayan modüler çerçevedir.\n\n"
                    "**1. Costmap2D (Maliyet Haritaları):**\n"
                    "Robotun çevresini 2B ızgaralara bölerek maliyet değerleri (0: Serbest, 254: Engel, 255: Bilinmeyen) atar:\n"
                    "- **Static Layer:** Önceden çıkarılmış statik harita (`map`).\n"
                    "- **Obstacle Layer:** 2D LiDAR veya derinlik kamerasından anlık tespit edilen dinamik engeller.\n"
                    "- **Inflation Layer:** Robot yarıçapına bağlı olarak engellerin etrafına güvenlik çemberi şişirir.\n\n"
                    "**2. Yerelleştirme (AMCL - Adaptive Monte Carlo Localization):**\n"
                    "Parçacık filtresi (Particle Filter) kullanarak robotun olası pozisyonlarını binlerce parçacıkla temsil eder. Robot hareket ettikçe parçacıklar odometriye göre dağıtılır; sensör ölçümü geldikçe lazer ışınları haritayla kıyaslanarak parçacıkların ağırlıkları güncellenir ve robotun konumu santimetre mertebesinde kilitlenir.\n\n"
                    "**3. Global ve Local Planlayıcılar:**\n"
                    "- **Global Planner (NavFn / Smac Planner):** $A^*$ veya Dijkstra algoritmasıyla başlangıçtan hedefe statik en kısa yolu çizer.\n"
                    "- **Local Controller (DWB / TEB Local Planner):** Robotun kinematik ve dinamik limitlerini (maksimum hız, ivme, dönüş yarıçapı) dikkate alarak anlık engellerden kaçan hız komutlarını (`geometry_msgs/Twist - cmd_vel`) motor sürücüsüne gönderir."
                )
            }
        ]
    }
]

def main():
    print(f"[*] Starting ingestion of {len(BTU_BOOKS)} BTÜ Mechatronics textbooks/compendiums...")

    indexer = Indexer()
    all_chunks = []
    vault_paper_cards = ROOT / "Vault" / "01_Paper_Cards"
    vault_theory = ROOT / "Vault" / "02_Theory_Notes"
    vault_paper_cards.mkdir(parents=True, exist_ok=True)
    vault_theory.mkdir(parents=True, exist_ok=True)

    catalog_path = ROOT / "data" / "corpus_catalog.json"
    existing_catalog = []
    if catalog_path.exists():
        try:
            existing_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except Exception:
            existing_catalog = []

    existing_ids = {item["id"] for item in existing_catalog}
    new_catalog_items = []

    for book in BTU_BOOKS:
        b_id = book["id"]
        b_title = book["title"]
        b_authors = book["authors"]
        b_year = book["year"]
        b_domain = book["domain"]

        print(f"\nProcessing book: {b_id} - {b_title}")

        # 1. Create Obsidian Paper Card
        card_file = vault_paper_cards / f"{b_id}.md"
        card_content = (
            f"---\n"
            f"type: paper_card\n"
            f"arxiv_id: '{b_id}'\n"
            f"title: {b_title}\n"
            f"authors: {b_authors}\n"
            f"year: {b_year}\n"
            f"status: approved\n"
            f"source: btu_mechatronics_curriculum\n"
            f"tags:\n"
            f"- btu-mekatronik\n"
            f"- ders-kitabi\n"
            f"- {b_domain.lower().replace('/', ' ').replace('&', ' ').replace('-', ' ')}\n"
            f"---\n\n"
            f"# {b_title} ({b_year})\n\n"
            f"**Kaynak:** `{b_authors}` | **Kategori:** `{b_domain}`\n\n"
            f"## 🎯 Genel Bakış & Müfredat Uyumu\n"
            f"{book['desc']}\n\n"
            f"## 📚 İçerik Bölümleri\n"
        )
        for s in book["sections"]:
            card_content += f"- **{s['section_name']}**\n"
        
        card_file.write_text(card_content, encoding="utf-8")
        print(f"  [+] Saved Obsidian Paper Card: {card_file.name}")

        # 2. Create chunks for ChromaDB
        for idx, sec in enumerate(book["sections"], 1):
            chunk_id = f"{b_id}_sec{idx}"
            chunk_text = f"Title: {b_title}\nSubject: {b_domain}\nChapter: {sec['section_name']}\n\n{sec['content']}"
            all_chunks.append({
                "id": chunk_id,
                "content": chunk_text,
                "arxiv_id": b_id,
                "title": b_title,
                "year": b_year,
                "section_name": sec["section_name"],
                "part_no": idx,
                "type": "paper_chunk"
            })

            # Save detailed theory note
            theory_file = vault_theory / f"{b_id}_Bolum_{idx}.md"
            theory_file.write_text(
                f"# {b_title} - {sec['section_name']}\n\n"
                f"**Kitap/Ders:** {b_title}\n"
                f"**Yazarlar:** {b_authors}\n\n"
                f"{sec['content']}\n",
                encoding="utf-8"
            )

        # 3. Add to catalog
        catalog_entry = {
            "id": b_id,
            "title": b_title,
            "authors": b_authors,
            "domain": b_domain,
            "color": book["color"],
            "desc": book["desc"],
            "prompt": book["prompt"]
        }
        if b_id not in existing_ids:
            existing_catalog.insert(0, catalog_entry)
            existing_ids.add(b_id)
            new_catalog_items.append(catalog_entry)

    # 4. Index into ChromaDB
    print(f"\n[*] Indexing {len(all_chunks)} chunks into ChromaDB collection 'paper_chunks'...")
    indexer.index_paper_chunks(all_chunks)
    print(f"[+] Successfully indexed {len(all_chunks)} chunks into ChromaDB!")

    # 5. Save updated catalog
    catalog_path.write_text(json.dumps(existing_catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] Updated corpus_catalog.json (Total entries: {len(existing_catalog)})")

    # Invalidate search cache if file exists
    cache_file = ROOT / "data" / "search_cache.json"
    if cache_file.exists():
        cache_file.unlink()
        print("[+] Invalidated search cache.")

    print("\n[OK] BTU Mechatronics books ingestion finished successfully!")

if __name__ == "__main__":
    main()
