"""
BTÜ Mechatronics Digital Logic Design & Electronics Circuits Ingestion Script.
Ingests:
1. BTU-MEK-MANTIK-DEVRELERI:
   - Digital Logic Design & Switching Theory (Morris Mano & Floyd)
   - Boolean Algebra, De Morgan, K-Maps, Combinational Circuits (Adders, MUX, Decoders)
   - Sequential Circuits (SR, D, JK, T Flip-Flops, Setup/Hold time)
   - Synchronous/Asynchronous Counters & Shift Registers
   - Finite State Machines (FSM - Mealy & Moore Machines, State Minimization)
2. BTU-MEK-ELEKTRONIK-DEVRELER:
   - Analog Electronics & Semiconductor Devices (Boylestad & Sedra/Smith)
   - Diode Physics, Shockley Equation, Rectifiers, Zener Regulators
   - BJT Transistors (NPN/PNP, Bias networks, Small-Signal Hybrid-pi, Common Emitter/Collector)
   - MOSFET Transistors (Enhancement/Depletion NMOS/PMOS, Triode/Saturation equations, CMOS Inverter)
   - Operational Amplifiers (Op-Amps, Schmitt Trigger hysteresis, Integrator/Differentiator, Active Sallen-Key Filters)

Indexes chunks into ChromaDB collection 'paper_chunks', creates Obsidian Cards,
and updates 'data/corpus_catalog.json'.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from ingest.indexer import Indexer

ELECTRONICS_LOGIC_BOOKS = [
    {
        "id": "BTU-MEK-MANTIK-DEVRELERI",
        "title": "Sayısal Mantık Devreleri ve Sayısal Sistem Tasarımı",
        "authors": "M. Morris Mano & Michael D. Ciletti, Thomas L. Floyd",
        "year": 2024,
        "domain": "BTÜ Mekatronik / Mantık Devreleri",
        "color": "#e11d48",
        "desc": "BTÜ Mantık Devreleri dersi ana kaynakları: Boolean cebri, Karnaugh haritaları (K-Map indirgemeleri), Kombinezonsal mantık (Toplayıcılar, MUX/DEMUX, Kodlayıcılar), Ardışıl devreler (Flip-Floplar, Sayıcılar) ve Sonlu Durum Makineleri (FSM Mealy & Moore).",
        "prompt": "BTÜ Mantık Devreleri dersi kapsamında Karnaugh Haritası (K-Map) ile mantıksal indirgeme kurallarını ve Mealy/Moore sonlu durum makineleri (FSM) arasındaki temel farkları açıkla.",
        "sections": [
            {
                "section_name": "Boolean Cebri, Temel Kapılar ve De Morgan Teoremleri",
                "content": (
                    "### Boolean Cebri, Temel Mantık Kapıları ve De Morgan Kuralları\n\n"
                    "Sayısal elektronik sistemler iki değerli (0 ve 1, Low ve High) Boolean cebri üzerine kuruludur. "
                    "Temel mantık kapıları AND, OR, NOT, NAND, NOR, XOR ve XNOR'dur. NAND ve NOR kapıları 'evrensel kapılar' (universal gates) olup sadece bu kapılar kullanılarak tüm mantıksal fonksiyonlar türetilebilir.\n\n"
                    "**De Morgan Teoremleri:**\n"
                    "Mantıksal çarpımın değili, değillerin toplamına; mantıksal toplamın değili ise değillerin çarpımına eşittir:\n"
                    "$$\n"
                    "\\overline{A \\cdot B} = \\overline{A} + \\overline{B}\n"
                    "$$\n"
                    "$$\n"
                    "\\overline{A + B} = \\overline{A} \\cdot \\overline{B}\n"
                    "$$\n"
                    "Genelleştirilmiş De Morgan kuralı ile $n$ değişkenli fonksiyonlar tek adımda dönüştürülebilir: "
                    "$\\overline{f(x_1, x_2, \\dots, x_n, \\cdot, +)} = f(\\overline{x}_1, \\overline{x}_2, \\dots, \\overline{x}_n, +, \\cdot)$.\n\n"
                    "**Boolean Eşitlikleri ve İndirgeme Kuralları:**\n"
                    "- Yutma Kuralı (Absorption): $A + A \\cdot B = A$ ve $A \\cdot (A + B) = A$\n"
                    "- Dağılma Kuralı (Distributive): $A + (B \\cdot C) = (A + B) \\cdot (A + C)$\n"
                    "- Sadeleştirme Kuralı: $A + \\overline{A} B = A + B$\n"
                    "- XOR Fonksiyonu: $A \\oplus B = A\\overline{B} + \\overline{A}B$\n"
                    "- XNOR (Eşdeğerlik) Fonksiyonu: $\\overline{A \\oplus B} = AB + \\overline{A}\\,\\overline{B}$"
                )
            },
            {
                "section_name": "Karnaugh Haritaları (K-Map) ve Mantıksal Fonksiyon İndirgeme",
                "content": (
                    "### Karnaugh Haritaları (K-Map) ile Sadeleştirme\n\n"
                    "Doğruluk tablosundaki çıktıyı en az sayıda kapı ve giriş ile gerçekleştirmek için grafiksel K-Map yöntemi kullanılır. K-Map hücreleri Gray Kodu sırasıyla dizilir ($00, 01, 11, 10$); böylece komşu hücreler arasında yalnızca tek 1 bit değişir.\n\n"
                    "**K-Map Gruplama Kuralları:**\n"
                    "1. Gruplar daima $2^n$ ($1, 2, 4, 8, 16$) adet komşu '1' hücresinden oluşmalıdır.\n"
                    "2. Gruplar mümkün olduğunca büyük seçilmelidir (büyük grup daha fazla değişkeni eler).\n"
                    "3. Haritanın karşıt kenarları ve dört köşesi bitişik kabul edilir (küresel topoloji / toroidal wrap-around).\n"
                    "4. **Fark Etmez Durumları (Don't Care - $d$ veya $X$):** Pratikte ortaya çıkmayacak veya çıktısı önemsiz durumlar, eğer bir grubu $2^n$ büyüklüğe tamamlayıp fonksiyonu basitleştiriyorsa '1' kabul edilir; aksi halde '0' sayılıp dışarıda bırakılır.\n\n"
                    "**Çarpımlar Toplamı (SOP - Sum of Products) ve Toplamlar Çarpımı (POS - Product of Sums):**\n"
                    "- **Minterm ($\\Sigma m$):** Çıktıyı '1' yapan giriş kombinasyonlarının AND çarpımları.\n"
                    "- **Maxterm ($\\Pi M$):** Çıktıyı '0' yapan giriş kombinasyonlarının OR toplamları.\n"
                    "- Asal Kapsayıcı (Prime Implicant) ve Zorunlu Asal Kapsayıcı (Essential Prime Implicant) bulunarak minimum donanımsal mantık devresi elde edilir."
                )
            },
            {
                "section_name": "Kombinezonsal Mantık Devreleri: Toplayıcılar, MUX, DEMUX, Kod Çözücüler",
                "content": (
                    "### Kombinezonsal (Bileşimsel) Mantık Devreleri\n\n"
                    "Çıkışların sadece o anki giriş durumlarına bağlı olduğu, bellek içermeyen devrelerdir.\n\n"
                    "**1. İkili Toplayıcılar:**\n"
                    "- **Yarım Toplayıcı (Half Adder):** İki 1-bitlik girişi ($A, B$) toplar:\n"
                    "  $$\\text{Toplam (Sum)} = A \\oplus B, \\quad \\text{Elde (Carry)} = A \\cdot B$$\n"
                    "- **Tam Toplayıcı (Full Adder):** Önceki basamaktan gelen eldeyi ($C_{in}$) de hesaba katar:\n"
                    "  $$S = A \\oplus B \\oplus C_{in}$$\n"
                    "  $$C_{out} = AB + C_{in}(A \\oplus B)$$\n"
                    "- Hızlı toplama için Ripple Carry Adder (zincirleme gecikmeli) yerine Carry Lookahead Adder (CLA - İleri Taşımalı Elde) kullanılır: Üretim $G_i = A_i B_i$ ve yayılım $P_i = A_i \\oplus B_i$ bağıntılarıyla elde bitleri kapı gecikmesi beklemeden paralel türetilir.\n\n"
                    "**2. Multiplexer (Veri Seçici - MUX):**\n"
                    "$2^n$ veri girişinden birini $n$ adet seçme ucuyla tekil çıkışa aktarır. Bir $2^n \\times 1$ MUX, herhangi bir $n$ değişkenli Boolean fonksiyonunu harici kapı kullanmadan doğrudan gerçekleştirebilir.\n\n"
                    "**3. Demultiplexer (DEMUX) ve Kod Çözücü (Decoder):**\n"
                    "$n$-bitlik ikili kodu $2^n$ adet tekil aktif çıkışa dönüştürür (örneğin 3-to-8 decoder). Aktif düşük (active-low) yetkilendirme (Enable) girişleri ile kaskat bağlanarak bellek adresleme blokları oluşturulur."
                )
            },
            {
                "section_name": "Ardışıl Mantık: Flip-Floplar, Saat Parametreleri ve Sayıcılar",
                "content": (
                    "### Ardışıl Mantık Devreleri ve Flip-Floplar\n\n"
                    "Çıkışın hem mevcut girişlere hem de devrenin geçmiş durumuna (bellek) bağlı olduğu sistemlerdir. Saat darbesiyle (Clock) senkronize edilirler.\n\n"
                    "**1. Temel Flip-Flop Tipleri ve Karakteristik Denklemleri:**\n"
                    "- **SR Flip-Flop:** $Q_{next} = S + \\overline{R}Q$ ($S=1, R=1$ durumu belirsiz ve yasaktır).\n"
                    "- **D (Data) Flip-Flop:** $Q_{next} = D$. Sayısal kaydırmalı yazmaçlar ve pipeline mimarisinin temel yapı taşıdır.\n"
                    "- **JK Flip-Flop:** SR'ın yasak durumunu toggle (durum değiştirme) yaparak çözer:\n"
                    "  $$Q_{next} = J\\overline{Q} + \\overline{K}Q$$\n"
                    "  $J=1, K=1$ olduğunda çıkış terslenir ($Q_{next} = \\overline{Q}$).\n"
                    "- **T (Toggle) Flip-Flop:** $Q_{next} = T \\oplus Q$. $T=1$ ise her saat darbesinde durum değişir (frekans bölücü ve sayıcı temeli).\n\n"
                    "**2. Zamanlama Parametreleri (Timing Constraints):**\n"
                    "- **Setup Time ($t_{su}$):** Saat kenarı gelmeden önce verinin giriş ucunda kararlı kalması gereken minimum süre.\n"
                    "- **Hold Time ($t_h$):** Saat kenarı geldikten sonra verinin bozulmadan tutulması gereken minimum süre.\n"
                    "Bu kurallar ihlal edilirse flip-flop kararsızlık (metastability) durumuna düşer ve çıkış mantık 0 ile 1 arasında asılı kalarak devreyi çökertir.\n\n"
                    "**3. Sayıcılar (Counters):**\n"
                    "- Asenkron (Ripple) Sayıcı: Bir flip-flop'un çıkışı diğerinin saat girişini besler; gecikme toplamı artar.\n"
                    "- Senkron Sayıcı: Tüm flip-flop'lar aynı saat darbesiyle tetiklenir, yüksek frekanslarda kararlıdır."
                )
            },
            {
                "section_name": "Sonlu Durum Makineleri (FSM): Mealy ve Moore Modelleri",
                "content": (
                    "### Sonlu Durum Makineleri (Finite State Machines - FSM)\n\n"
                    "Robotik kontrol mantığı, sıralayıcılar ve iletişim protokolü kontrolcülerinde durum geçişlerini modellemek için iki temel FSM mimarisi kullanılır:\n\n"
                    "**1. Moore Durum Makinesi:**\n"
                    "- Çıkışlar **yalnızca mevcut duruma** bağlıdır ($Y = g(S)$).\n"
                    "- Giriş sinyallerindeki anlık gürültüler (glitch) doğrudan çıkışa yansımaz; çıkış saat kenarıyla senkronize değişir.\n"
                    "- Durum geçiş diyagramında çıkışlar durum dairesinin içine yazılır (örneğin $S_0 / 0$). Genellikle Mealy'ye göre daha fazla durum gerektirir.\n\n"
                    "**2. Mealy Durum Makinesi:**\n"
                    "- Çıkışlar **hem mevcut duruma hem de o anki girişlere** bağlıdır ($Y = g(S, X)$).\n"
                    "- Girişte meydana gelen bir değişiklik saat darbesini beklemeden aynı çevrim içinde çıkışta anında görülebilir (asenkron çıkış tepkisi).\n"
                    "- Genellikle Moore makinesine göre daha az durum ile aynı işlevi gerçekleştirir ancak giriş hatlarındaki parazitlerin çıkışa sızma riski vardır.\n\n"
                    "**FSM Tasarım Adımları:**\n"
                    "1. Problem tanımından Durum Diyagramı (State Diagram) çizilir.\n"
                    "2. Durum Tablosu (State Table) oluşturulur ve durum indirgemesi (State Reduction) yapılır.\n"
                    "3. Durum Kodlama (State Assignment: Binary, Gray veya FPGA için hızlı One-Hot kodlama) seçilir.\n"
                    "4. Flip-Flop geçiş tabloları ve K-Map kullanılarak bir sonraki durum mantığı ve çıkış mantığı denklemleri türetilir."
                )
            }
        ]
    },
    {
        "id": "BTU-MEK-ELEKTRONIK-DEVRELER",
        "title": "Analog Elektronik, Yarıiletken Fiziği ve Devre Teorisi",
        "authors": "Robert L. Boylestad, Louis Nashelsky & Adel S. Sedra, Kenneth C. Smith",
        "year": 2024,
        "domain": "BTÜ Mekatronik / Elektronik Devreleri",
        "color": "#0ea5e9",
        "desc": "BTÜ Elektronik Devreleri dersi ana kaynakları: Diyot fiziği (Shockley denklemi, doğrultucular, Zener), BJT transistörler (Kutuplama, küçük sinyal hibrid-pi modeli, yükselteçler), MOSFET fiziği (Karakteristik denklemler, CMOS anahtarlama) ve Op-Amp analog devreleri (Schmitt trigger, aktif filtreler).",
        "prompt": "BTÜ Elektronik dersi kapsamında BJT transistörün küçük sinyal hibrid-pi modelini ve MOSFET doyum bölgesi akım denklemini türeterek çalışma bölgelerini açıkla.",
        "sections": [
            {
                "section_name": "Yarıiletken Fiziği, p-n Jonksiyonu ve Diyot Devreleri",
                "content": (
                    "### Yarıiletkenler, p-n Jonksiyonu ve Diyot Uygulamaları\n\n"
                    "Silisyum (Si) tabanlı yarıiletkenlerde p-tipi (delik çoğunluk taşıyıcı) ve n-tipi (elektron çoğunluk taşıyıcı) bölgelerin birleşimiyle p-n jonksiyonu oluşur. Birleşim yüzeyinde taşıyıcıların difüzyonu sonucu boşaltılmış bölge (depletion region) ve bir iç potansiyel bariyeri ($V_{bi} \\approx 0.7\\text{ V}$) meydana gelir.\n\n"
                    "**Shockley Diyot İdeal Akım Denklemi:**\n"
                    "$$\n"
                    "I_D = I_S \\left( e^{\\frac{V_D}{n V_T}} - 1 \\right)\n"
                    "$$\n"
                    "Burada $I_S$ ters doyma akımı ($10^{-15} - 10^{-12}\\text{ A}$), $n$ emisyon katsayısı ($1 \\le n \\le 2$), $V_T = \\frac{k T}{q}$ termal gerilimdir ($T = 300\\text{ K}$ oda sıcaklığında $V_T \\approx 26\\text{ mV}$).\n\n"
                    "**Temel Diyot Devreleri:**\n"
                    "1. **Köprü Doğrultucu (Full-Wave Bridge Rectifier):** 4 diyot ile AC girişin her iki yarım dalgasını da DC'ye dönüştürür. Çıkış tepe gerilimi $V_{peak,out} = V_{peak,in} - 2 V_D$ olur.\n"
                    "2. **Filtre Kondansatörü ve Dalgalanma Gerilimi (Ripple Voltage):**\n"
                    "   $$V_{ripple} = \\frac{I_{load}}{2 f C}$$\n"
                    "3. **Zener Diyot ve Gerilim Regülasyonu:** Ters kırılma bölgesinde ($V_Z$) çalıştırılarak giriş gerilimi veya yük akımı değişse dahi yük üzerinde sabit gerilim sağlar. Seri direnç $R_s = \\frac{V_{in,min} - V_Z}{I_{L,max} + I_{Z,min}}$ ile boyutlandırılır."
                )
            },
            {
                "section_name": "Bipolar Jonksiyon Transistörler (BJT): Kutuplama ve Küçük Sinyal Analizi",
                "content": (
                    "### BJT Transistörler ve Küçük Sinyal Modelleri\n\n"
                    "BJT (NPN veya PNP), akım kontrollü bir yarıiletken elemandır. Üç temel çalışma bölgesi vardır:\n"
                    "1. **Kesim (Cut-off):** $V_{BE} < 0.7\\text{ V}$, $I_B = 0, I_C \\approx 0$. Transistör açık bir anahtar gibidir.\n"
                    "2. **İleri Aktif (Forward Active - Yükselteç Bölgesi):** Baz-Emiter ileri kutuplu ($V_{BE} \\approx 0.7\\text{ V}$), Kollektör-Baz ters kutupludur. Akım kazancı $\\beta$ (veya $h_{FE}$):\n"
                    "   $$I_C = \\beta I_B, \\quad I_E = I_B + I_C = (\\beta + 1) I_B$$\n"
                    "3. **Doyum (Saturation):** Her iki jonksiyon da ileri kutupludur. Kollektör gerilimi $V_{CE,sat} \\approx 0.2\\text{ V}$ seviyesine düşer. Transistör kapalı bir anahtar gibi davranır ($I_C < \\beta I_B$).\n\n"
                    "**DC Kutuplama (Voltaj Bölücülü Kutuplama - Voltage Divider Bias):**\n"
                    "Sıcaklık ve $\\beta$ değişimlerinden etkilenmeyen kararlı çalışma noktası (Q-noktası) elde etmek için kullanılır:\n"
                    "$$V_{TH} = V_{CC} \\frac{R_2}{R_1 + R_2}, \\quad R_{TH} = R_1 \\parallel R_2$$\n"
                    "$$I_B = \\frac{V_{TH} - V_{BE}}{R_{TH} + (\\beta + 1) R_E}$$\n\n"
                    "**Küçük Sinyal Hibrid-$\\pi$ Modeli:**\n"
                    "AC sinyaller için transistör lineerleştirilir:\n"
                    "- Eşdeğer Transkondüktans: $g_m = \\frac{I_{CQ}}{V_T} = \\frac{I_{CQ}}{26\\text{ mV}}$\n"
                    "- Baz Giriş Direnci: $r_\\pi = \\frac{\\beta}{g_m} = \\frac{\\beta V_T}{I_{CQ}}$\n"
                    "- Early Direnci (Çıkış Direnci): $r_o = \\frac{V_A + V_{CEQ}}{I_{CQ}}$ ($V_A$: Early Gerilimi)\n"
                    "- Ortak Emiterli (Common-Emitter) Gerilim Kazancı: $A_v = -g_m (R_C \\parallel R_L \\parallel r_o)$."
                )
            },
            {
                "section_name": "Alan Etkili Transistörler (MOSFET): Karakteristik Denklemler ve CMOS",
                "content": (
                    "### MOSFET (Metal-Oxide-Semiconductor Field-Effect Transistor)\n\n"
                    "MOSFET gerilim kontrollü bir elemandır; Gate (kapı) ucu silikon dioksit ($SiO_2$) yalıtkan katmanla ayrıldığı için DC kapı akımı sıfırdır ($I_G = 0$). Giriş empedansı teorik olarak sonsuzdur.\n\n"
                    "**Zenginleştirilmiş NMOS Çalışma Bölgeleri:**\n"
                    "1. **Kesim Bölgesi ($V_{GS} < V_{th}$):** İletim kanalı oluşmamıştır. $I_D = 0$.\n"
                    "2. **Lineer / Triyot Bölgesi ($V_{GS} \\ge V_{th}$ ve $V_{DS} < V_{GS} - V_{th}$):**\n"
                    "   Kanal boydan boya iletkendir, gerilim kontrollü direnç gibi davranır:\n"
                    "   $$I_D = \\mu_n C_{ox} \\frac{W}{L} \\left[ (V_{GS} - V_{th}) V_{DS} - \\frac{1}{2} V_{DS}^2 \\right]$$\n"
                    "3. **Doyum (Saturation / Pinch-off) Bölgesi ($V_{GS} \\ge V_{th}$ ve $V_{DS} \\ge V_{GS} - V_{th}$):**\n"
                    "   Kanal Drain ucunda boğulur (pinch-off). Akım $V_{DS}$'den bağımsız hale gelerek sabitlenir (yükselteç çalışma bölgesi):\n"
                    "   $$I_D = \\frac{1}{2} \\mu_n C_{ox} \\frac{W}{L} (V_{GS} - V_{th})^2 (1 + \\lambda V_{DS})$$\n"
                    "   Burada $\\lambda$ kanal boyu modülasyon parametresidir (Early etkisinin MOSFET karşılığı).\n\n"
                    "**CMOS Teknolojisi (Complementary MOS):**\n"
                    "Bir pMOS ve bir nMOS transistörün seri bağlanmasıyla oluşur. Mantık 0 veya 1 durumlarında transistörlerden biri daima kesimde olduğundan statik güç tüketimi sıfıra yakındır; güç yalnızca anahtarlama anında (dinamik: $P = C f V_{DD}^2$) tüketilir."
                )
            },
            {
                "section_name": "İşlemsel Yükselteçler (Op-Amp): Doğrusal ve Doğrusal Olmayan Devreler",
                "content": (
                    "### İşlemsel Yükselteçler (Op-Amp) ve Aktif Devreler\n\n"
                    "İdeal Op-Amp özellikleri: Sonsuz açık çevrim kazancı ($A_{OL} \\to \\infty$), sonsuz giriş empedansı ($R_{in} \\to \\infty \\implies I_+ = I_- = 0$), sıfır çıkış empedansı ($R_{out} = 0$) ve sonsuz bant genişliği.\n"
                    "Negatif geri besleme altında sanal kısa devre (Virtual Ground / Virtual Short) prensibi geçerlidir: $V_+ = V_-$.\n\n"
                    "**1. Doğrusal Op-Amp Konfigürasyonları:**\n"
                    "- **Evirmeyen (Non-Inverting) Yükselteç:** $V_{out} = \\left( 1 + \\frac{R_f}{R_1} \\right) V_{in}$\n"
                    "- **Eviren (Inverting) Yükselteç:** $V_{out} = -\\frac{R_f}{R_1} V_{in}$\n"
                    "- **İdeal İntegratör:** $V_{out}(t) = -\\frac{1}{R C} \\int_0^t V_{in}(\\tau) d\\tau + V_{out}(0)$\n"
                    "- **İdeal Türev Alıcı:** $V_{out}(t) = -R C \\frac{dV_{in}(t)}{dt}$ (Pratikte yüksek frekans gürültüsünü engellemek için seri giriş direnci eklenir).\n\n"
                    "**2. Schmitt Trigger (Histerezisli Karşılaştırıcı):**\n"
                    "Gürültülü sensör sinyallerinde çıkışın titreşmesini (chatter) önlemek için pozitif geri besleme kullanılır. İki farklı eşik gerilimi üretilir:\n"
                    "$$V_{TH} = \\frac{R_1}{R_1 + R_2} (+V_{sat}), \\quad V_{TL} = \\frac{R_1}{R_1 + R_2} (-V_{sat})$$\n"
                    "Histerezis bandı $V_H = V_{TH} - V_{TL}$ gürültü bağışıklığı sağlar.\n\n"
                    "**3. Aktif Sallen-Key 2. Derece Alçak Geçiren Filtre:**\n"
                    "İki direnç ve iki kondansatör ile $40\\text{ dB/decade}$ düşüş sağlayan transfer fonksiyonu:\n"
                    "$$H(s) = \\frac{K \\omega_0^2}{s^2 + \\frac{\\omega_0}{Q} s + \\omega_0^2}, \\quad f_c = \\frac{1}{2\\pi \\sqrt{R_1 R_2 C_1 C_2}}$$"
                )
            }
        ]
    }
]

def main():
    print(f"[*] Starting ingestion of {len(ELECTRONICS_LOGIC_BOOKS)} Logic & Electronics courses...")

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

    for book in ELECTRONICS_LOGIC_BOOKS:
        b_id = book["id"]
        b_title = book["title"]
        b_authors = book["authors"]
        b_year = book["year"]
        b_domain = book["domain"]

        print(f"\nProcessing: {b_id} - {b_title}")

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

    print("\n[OK] BTU Logic & Electronics books ingestion finished successfully!")

if __name__ == "__main__":
    main()
