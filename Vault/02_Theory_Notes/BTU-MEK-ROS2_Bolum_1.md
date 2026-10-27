# Robot İşletim Sistemi (ROS2 Humble) ve Otonom Mobil Robotlar - ROS2 Mimarisi, DDS Omurgası ve QoS Politikaları

**Kitap/Ders:** Robot İşletim Sistemi (ROS2 Humble) ve Otonom Mobil Robotlar
**Yazarlar:** Steven Macenski, Tully Foote & Open Robotics

### ROS2 Haberleşme Altyapısı ve Data Distribution Service (DDS)

ROS 1'in tekil merkezi master (`roscore`) bağımlılığı kaldırılarak endüstriyel standart OMG DDS (Data Distribution Service) üzerine inşa edilmiştir. Bu sayede sistem dağıtık, sıfır tek nokta arızalı (no single point of failure) ve gerçek zamanlı çalışabilir.

**Haberleşme Primitifleri:**
1. **Topics (Konular):** Çoktan-çoğa tek yönlü veri akışı (Publish / Subscribe). Sensör akışları ve telemetri için kullanılır.
2. **Services (Servisler):** Bire-bir eşzamanlı istek/cevap (Request / Response). Parametre yapılandırma veya anlık hesaplamalar için uygundur.
3. **Actions (Eylemler):** İstemci-sunucu tabanlı, uzun süren, iptal edilebilir ve ara geri bildirim (feedback) sağlayan hedefler (örneğin otonom navigasyonda hedef noktaya gitme).

**Quality of Service (QoS) Profilleri:**
- **Reliability (Güvenilirlik):** `BEST_EFFORT` (Kayıp paketleri umursamaz, düşük gecikme; kamera/LiDAR akışları için ideal) veya `RELIABLE` (TCP benzeri onay mekanizması, kritik komutlar için).
- **Durability (Kalıcılık):** `VOLATILE` (Sadece o anki abonelere gider) veya `TRANSIENT_LOCAL` (Son mesaj saklanır, yeni abone bağlandığında ona hemen iletilir; örneğin `/map` harita verisi).
