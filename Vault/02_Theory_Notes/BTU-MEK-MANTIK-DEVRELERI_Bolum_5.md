# Sayısal Mantık Devreleri ve Sayısal Sistem Tasarımı - Sonlu Durum Makineleri (FSM): Mealy ve Moore Modelleri

**Kitap/Ders:** Sayısal Mantık Devreleri ve Sayısal Sistem Tasarımı
**Yazarlar:** M. Morris Mano & Michael D. Ciletti, Thomas L. Floyd

### Sonlu Durum Makineleri (Finite State Machines - FSM)

Robotik kontrol mantığı, sıralayıcılar ve iletişim protokolü kontrolcülerinde durum geçişlerini modellemek için iki temel FSM mimarisi kullanılır:

**1. Moore Durum Makinesi:**
- Çıkışlar **yalnızca mevcut duruma** bağlıdır ($Y = g(S)$).
- Giriş sinyallerindeki anlık gürültüler (glitch) doğrudan çıkışa yansımaz; çıkış saat kenarıyla senkronize değişir.
- Durum geçiş diyagramında çıkışlar durum dairesinin içine yazılır (örneğin $S_0 / 0$). Genellikle Mealy'ye göre daha fazla durum gerektirir.

**2. Mealy Durum Makinesi:**
- Çıkışlar **hem mevcut duruma hem de o anki girişlere** bağlıdır ($Y = g(S, X)$).
- Girişte meydana gelen bir değişiklik saat darbesini beklemeden aynı çevrim içinde çıkışta anında görülebilir (asenkron çıkış tepkisi).
- Genellikle Moore makinesine göre daha az durum ile aynı işlevi gerçekleştirir ancak giriş hatlarındaki parazitlerin çıkışa sızma riski vardır.

**FSM Tasarım Adımları:**
1. Problem tanımından Durum Diyagramı (State Diagram) çizilir.
2. Durum Tablosu (State Table) oluşturulur ve durum indirgemesi (State Reduction) yapılır.
3. Durum Kodlama (State Assignment: Binary, Gray veya FPGA için hızlı One-Hot kodlama) seçilir.
4. Flip-Flop geçiş tabloları ve K-Map kullanılarak bir sonraki durum mantığı ve çıkış mantığı denklemleri türetilir.
