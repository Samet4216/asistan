# Robot İşletim Sistemi (ROS2 Humble) ve Otonom Mobil Robotlar - Nav2 Navigasyon Yığını: Costmap2D, AMCL ve Yol Planlama

**Kitap/Ders:** Robot İşletim Sistemi (ROS2 Humble) ve Otonom Mobil Robotlar
**Yazarlar:** Steven Macenski, Tully Foote & Open Robotics

### ROS2 Nav2 Otonom Navigasyon Yığını

Mobil robotun bir harita üzerinde engellere çarpmadan hedef koordinata gitmesini sağlayan modüler çerçevedir.

**1. Costmap2D (Maliyet Haritaları):**
Robotun çevresini 2B ızgaralara bölerek maliyet değerleri (0: Serbest, 254: Engel, 255: Bilinmeyen) atar:
- **Static Layer:** Önceden çıkarılmış statik harita (`map`).
- **Obstacle Layer:** 2D LiDAR veya derinlik kamerasından anlık tespit edilen dinamik engeller.
- **Inflation Layer:** Robot yarıçapına bağlı olarak engellerin etrafına güvenlik çemberi şişirir.

**2. Yerelleştirme (AMCL - Adaptive Monte Carlo Localization):**
Parçacık filtresi (Particle Filter) kullanarak robotun olası pozisyonlarını binlerce parçacıkla temsil eder. Robot hareket ettikçe parçacıklar odometriye göre dağıtılır; sensör ölçümü geldikçe lazer ışınları haritayla kıyaslanarak parçacıkların ağırlıkları güncellenir ve robotun konumu santimetre mertebesinde kilitlenir.

**3. Global ve Local Planlayıcılar:**
- **Global Planner (NavFn / Smac Planner):** $A^*$ veya Dijkstra algoritmasıyla başlangıçtan hedefe statik en kısa yolu çizer.
- **Local Controller (DWB / TEB Local Planner):** Robotun kinematik ve dinamik limitlerini (maksimum hız, ivme, dönüş yarıçapı) dikkate alarak anlık engellerden kaçan hız komutlarını (`geometry_msgs/Twist - cmd_vel`) motor sürücüsüne gönderir.
