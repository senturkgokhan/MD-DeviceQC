# MD-DeviceQC — Teknik Dokümantasyon
### Kalite Kontrol Motoru ve Sistem Mimarisi

Bu dokümantasyon, MD-DeviceQC sisteminin teknik detaylarını, kurulum adımlarını ve konfigürasyon seçeneklerini içermektedir.
Sistem; YOLOv5s tabanlı nesne tespiti, üç aşamalı denetim akışı ve PySide6 arayüzünden oluşmaktadır.

## Sistem Mimarisi
Sistem iki ana katmandan oluşmaktadır:

- **qc_launcher.py** — PySide6 arayüz katmanı. Ana menü, ayarlar ve denetim ekranını yönetir. YOLO inference ayrı bir QThread içinde çalıştığından arayüz hiçbir zaman donmaz.

- **qc_engine.py** — Kalite kontrol motoru. YOLO tespit, cihaz tanıma, aşama yönetimi ve SQLite kaydını üstlenir.

```
qc_launcher.py  (PySide6 — Ana Arayüz)
└── CameraWorker (QThread)
        └── qc_engine.py — QualityControlApp
                ├── YOLOv5s (best.pt)
                ├── inspection_profiles.py
                ├── config/device_profiles.yaml
                ├── OpenCV kamera
                └── pyzbar barkod okuyucu
```
## Denetim Akışı

Sistem dört aşamalı bir durum makinesi (state machine) olarak çalışır. Her aşamanın belirli bir süre sınırı vardır. Süre dolduğunda sistem otomatik olarak bir sonraki adıma geçer.

### FRONT — Ön Yüz Kontrolü (20 sn)
Operatör cihazın ön yüzünü kameraya gösterir. YOLO modeli bileşenleri tespit eder ve sayar. Tespit edilen bileşen sayıları device_profiles.yaml içindeki referans değerleriyle karşılaştırılır. Sistem, bileşen sayım skoruna göre cihazın DM100 mu yoksa XIO110 mu olduğunu otomatik olarak belirler. 12 ardışık kare boyunca tutarlı sonuç alındığında tanıma kesinleşir ve sistem otomatik olarak LABEL aşamasına geçer. Operatör 1/2 tuşlarıyla cihaz tipini manuel olarak da seçebilir.

### LABEL — Etiket Kontrolü (15 sn)
Operatör cihazın etiketli yüzeyini kameraya gösterir. Sistem, Garanti ve Kalite Kontrol etiketlerinin varlığını kontrol eder. Her iki etiket de tespit edildiğinde 8 ardışık kare boyunca tutarlı sonuç alınırsa sistem otomatik olarak BARCODE aşamasına geçer.

### BARCODE — Barkod Okuma (15 sn)
Operatör cihazın barkodunu kameraya gösterir. pyzbar kütüphanesiile seri numarası okunur. Seri numarası başarıyla okunduğundasistem otomatik olarak RESULT aşamasına geçer.

### RESULT — Sonuç
Tüm aşamalar tamamlandığında sistem PASS veya FAIL kararını ekranda gösterir ve sonucu seri numarasıyla birlikte SQLite veritabanına kaydeder. 2 saniye sonra sistem otomatik olarak
sıfırlanır ve yeni ürün için FRONT aşamasına döner.

### FAIL Senaryoları

| Senaryo | Sistem Davranışı |
|---------|-----------------|
| FRONT başarısız | Direkt BARCODE aşamasına geçer, seri no ile FAIL kaydeder |
| LABEL başarısız | Direkt BARCODE aşamasına geçer, seri no ile FAIL kaydeder |
| BARCODE okunamaz | FAIL kaydedilir, seri no yerine session ID kullanılır |
| Süre dolması | İlgili aşama FAIL sayılır, BARCODE aşamasına yönlendirilir |

> **Temel tasarım kararı:** Herhangi bir aşama başarısız olsa bile sistem barkod okuma aşamasını atlamaz. Bu sayede hatalı cihazlar seri numarasıyla birlikte kayıt altına alınır ve izlenebilirlik hiçbir koşulda kopmaz.

## Model ve Dataset

### Dataset

Tüm eğitim verisi Mikrodev üretim ortamında, gerçek DM100 ve XIO110 cihazları üzerinde bizzat toplanmıştır. Görseller laptop kamerası ve IMX219 kamera kullanılarak farklı açı ve ışık koşullarında elde edilmiştir.

| | Görüntü |
|--|---------|
| Ham toplam | 1.682 |
| Train (aug. öncesi) | 1.190 |
| Train (aug. sonrası) | 3.570 |
| Validation | 334 |
| Test | 158 |
| **Toplam** | **4.062** |

### Etiketleme
Tüm görseller Roboflow platformu üzerinde 10 sınıf için her bileşen tek tek, bizzat etiketlenmiştir.

| Sınıf | Açıklama |
|-------|----------|
| DM100 | Cihaz tipi marker |
| XIO110 | Cihaz tipi marker |
| Ethernet | Ethernet portu |
| usb-b | USB-B portu |
| Klemens-Group | Terminal blok grubu |
| Led-Group | LED grubu |
| SD-Card | SD kart yuvası |
| Quality-Control | Kalite kontrol etiketi |
| Warranty | Garanti etiketi |
| ID-Switch | Kimlik anahtarı (yalnızca XIO110) |

### Augmentation
Augmentation yalnızca eğitim kümesine uygulanmıştır. Doğrulama ve test kümeleri ham halde tutulmuştur.

Uygulanan yöntemler: Rotation, Hue, Saturation, Brightness, Exposure, Blur, Noise
> Yatay çevirme ve 90° döndürme uygulanmamıştır — cihazlar 
> simetrik olmadığından bu dönüşümler geçersiz eğitim örnekleri üretir.

### Model Eğitimi ve Karşılaştırma

Sistem geliştirilirken YOLOv8s ve YOLOv5s modelleri farklı 
parametrelerle karşılaştırmalı olarak eğitilmiştir.

| Model | Epoch | Görüntü Boyutu | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|-------|----------------|---------|--------------|-----------|--------|
| YOLOv8s | 150 | 640 | ~0.987 | ~0.835 | ~0.970 | ~0.980 |
| YOLOv8s | 200 | 640 | ~0.987 | ~0.820 | ~0.970 | ~0.980 |
| YOLOv5s | 150 | 832 | ~0.980 | ~0.800 | ~0.980 | ~0.980 |

**Eğitim Grafikleri:**

**YOLOv8s — 150 Epoch**
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/55467db6-8170-4a59-bb7b-143fb51beb1a" />

**YOLOv8s — 200 Epoch**
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/dca84ba5-2f0e-46fc-8c0f-e667dbd26eb0" />

**YOLOv5s — 150 Epoch — 832 Görüntü Boyutu (Seçilen Model)**
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/ab882498-f498-42a2-8667-53aecf644888" />

### Neden YOLOv5s Seçildi?

- YOLOv8s metrik olarak biraz daha iyi çıkmıştır ancak fark üretim ortamında anlamlı bir fark yaratmamaktadır.
- YOLOv5s eğitim eğrisi çok daha stabil seyretmiştir.
- Sahada PC ve test laptobu üzerinde YOLOv5s daha akıcı çalışmaktadır.
- Endüstriyel kullanımda doğruluk ve gerçek zamanlı performans dengesi kritik öneme sahiptir.



