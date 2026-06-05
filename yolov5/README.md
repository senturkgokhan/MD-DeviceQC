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








## Ana dosyalar

| Dosya | Açıklama |
|-------|----------|
| `qc_launcher.py` | PySide6 arayüz |
| `qc_engine.py` | Kamera, YOLO, aşamalı QC |
| `config/device_profiles.yaml` | Cihaz profilleri |
| `config/app_settings.json` | Kullanıcı ayarları (yerelde oluşur) |
| `results/devices.db` | SQLite kayıtları |

## Taban

[Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) üzerine inşa edilmiştir. Güncellemeler için `upstream` remote kullanılabilir.
