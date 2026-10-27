# Robot İşletim Sistemi (ROS2 Humble) ve Otonom Mobil Robotlar - URDF, Xacro ve TF2 Koordinat Dönüşüm Kütüphanesi

**Kitap/Ders:** Robot İşletim Sistemi (ROS2 Humble) ve Otonom Mobil Robotlar
**Yazarlar:** Steven Macenski, Tully Foote & Open Robotics

### Robot Modelleme ve Koordinat Dönüşümleri (TF2)

**1. URDF (Unified Robot Description Format) ve Xacro:**
Robotun 3B kinematik ve dinamik ağacı XML tabanlı tanımlanır:
- `<link>`: Kütle ($m$), eylemsizlik tensörü ($I_{xx}, I_{yy}, \dots$), görsel mesh (visual) ve çarpışma geometrisi (collision).
- `<joint>`: İki link arasındaki ilişki (revolute, continuous, prismatic, fixed). Eksen ve hareket limitleri tanımlanır.
- Xacro, makrolar ve matematiksel ifadeler ekleyerek kod tekrarını önler.

**2. TF2 Dönüşüm Ağacı (Transform Tree):**
Otonom robotlarda birden fazla sensör farklı konumlara monte edilmiştir. TF2, zaman damgalı dönüşümleri ($T_A^B(t)$) tamponlayarak koordinat dönüşümlerini yönetir:
Standart ROS koordinat hiyerarşisi:
`map` $\longrightarrow$ `odom` $\longrightarrow$ `base_link` $\longrightarrow$ `laser_frame` / `camera_link`
- `map -> odom`: Global yerelleştirme (AMCL veya SLAM) tarafından yayınlanan, kaymasız ancak kesintili sıçramalar yapabilen dönüşüm.
- `odom -> base_link`: Tekerlek odometrisi ve IMU tarafından yüksek frekansta yayınlanan, sürekli ancak zamanla drift yapan yerel dönüşüm.
- `base_link -> sensor`: URDF tarafından robot gövdesine göre sabit tanımlanan statik dönüşüm.
