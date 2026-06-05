# MD-DeviceQC — Teknik Dokümantasyon
### Kalite Kontrol Motoru ve Sistem Mimarisi

Bu dokümantasyon, MD-DeviceQC sisteminin teknik detaylarını, kurulum adımlarını ve konfigürasyon seçeneklerini içermektedir.
Sistem; YOLOv5s tabanlı nesne tespiti, dört aşamalı denetim akışı ve PySide6 arayüzünden oluşmaktadır.

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
Operatör cihazın barkodunu kameraya gösterir. pyzbar kütüphanesiile seri numarası okunur. Seri numarası başarıyla okunduğunda sistem otomatik olarak RESULT aşamasına geçer.

### RESULT — Sonuç
Tüm aşamalar tamamlandığında sistem PASS veya FAIL kararını ekranda gösterir ve sonucu seri numarasıyla birlikte SQLite veritabanına kaydeder. 2 saniye sonra sistem otomatik olarak
sıfırlanır ve yeni ürün için FRONT aşamasına döner.

### Cihaz Tanıma Mekanizması

DM100 ve XIO110 cihazları ön yüz bileşen sayıları kullanılarak otomatik olarak tanımlanır. Sistem her kare için tespit edilen bileşen sayılarını device_profiles.yaml içindeki referans değerleriyle karşılaştırarak bir uyum skoru üretir.

**Referans Bileşen Sayıları:**

| Bileşen | DM100 | XIO110 |
|---------|-------|--------|
| Ethernet | 1 | — |
| USB-B | 1 | — |
| SD-Card | 1 | — |
| Klemens-Group | 2 | 3 |
| Led-Group | 2 | 1 |
| ID-Switch | — | 1 |

**Tanıma Süreci:**

- Her kare için her profile karşı bir uyum skoru (0-1) hesaplanır
- En yüksek skorlu profil aday olarak seçilir
- Minimum skor 0.88, iki aday arasındaki minimum fark 0.12 olmalıdır
- 12 ardışık kare boyunca aynı sonuç alındığında tanıma kesinleşir ve sistem otomatik olarak LABEL aşamasına geçer
- Operatör istediği zaman 1/2 tuşlarıyla cihaz tipini manuel olarak da belirleyebilir

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

Tüm eğitim verisi Mikrodev üretim ortamında, gerçek DM100 ve XIO110 cihazları üzerinde bizzat toplanmıştır. Görseller laptop kamerası ve IMX219 kamera kullanılarak farklı açı ve ışık koşullarında elde edilmiştir. Toplanan 1.682 ham görüntü %70/%20/%10 oranında eğitim, doğrulama ve test kümelerine ayrılmıştır. Augmentation yalnızca eğitim kümesine uygulanarak eğitim seti yaklaşık üç katına çıkarılmıştır.

```
Ham Görüntü Sayısı : 1.682

Train      : 1.190  (%70)
Validation :   334  (%20)
Test       :   158  (%10)

Augmentation Sonrası Train : 3.570

Toplam Eğitim Görüntüsü   : 4.062
```

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

Sistem geliştirilirken YOLOv8s ve YOLOv5s modelleri farklı parametrelerle karşılaştırmalı olarak eğitilmiştir. Tüm eğitimler Google Colab üzerinde gerçekleştirilmiştir.

| Model | Epoch | Görüntü Boyutu | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|-------|----------------|---------|--------------|-----------|--------|
| YOLOv8s | 150 | 640 | 0.987 | 0.835 | 0.970 | 0.980 |
| YOLOv8s | 200 | 640 | 0.987 | 0.820 | 0.970 | 0.980 |
| **YOLOv5s** | **150** | **832** | **0.992** | **0.944** | **0.967** | **0.978** |

**Eğitim Grafikleri:**

**YOLOv8s — 150 Epoch**
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/55467db6-8170-4a59-bb7b-143fb51beb1a" />

**YOLOv8s — 200 Epoch**
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/dca84ba5-2f0e-46fc-8c0f-e667dbd26eb0" />

**YOLOv5s — 150 Epoch — 832 Görüntü Boyutu (Seçilen Model)**
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/ab882498-f498-42a2-8667-53aecf644888" />

### Neden YOLOv5s Seçildi?
YOLOv8s bazı deneylerde daha yüksek metrikler üretmesine rağmen,sistem geliştirme sürecinde YOLOv5s aşağıdaki avantajları sağlamıştır:
- **Daha kararlı eğitim eğrileri** — YOLOv5s eğitim süreci boyunca çok daha stabil bir grafik sergilemiştir
- **Daha düşük inference gecikmesi** — Sahada PC ve test laptobu üzerinde daha akıcı çalışmaktadır
- **Jetson Nano uyumluluğu** — Deploy sürecinde YOLOv5s daha kolay entegre edilebilmiştir
- **Mevcut sistem mimarisiyle yüksek uyumluluk** — YOLOv5 utils ve model yapısı sisteme doğrudan entegre edilmiştir
- **Üretim ortamında yeterli doğruluk** — mAP@0.5 0.992, Precision 0.967, Recall 0.978 değerleriyle endüstriyel kullanım için yeterli doğruluk seviyesi sağlanmıştır
Bu nedenle nihai sistem modeli olarak YOLOv5s tercih edilmiştir.

## Sınırlamalar
- Sistem yalnızca DM100 ve XIO110 cihazlarını desteklemektedir. Yeni cihaz tipi eklemek için device_profiles.yaml güncellenmeli ve model yeniden eğitilmelidir.
- Kamera görüş alanı dışında kalan bileşenler doğrulanamaz. Operatörün cihazı doğru açıyla göstermesi gerekmektedir.
- Barkodun okunabilmesi için yeterli görüntü kalitesi ve mesafe gereklidir. Aşırı yakın veya uzak açılar okuma başarısını düşürebilir.
- Aşırı yansıma ve hareket bulanıklığı tespit başarısını olumsuz etkileyebilir.
- Sistem şu an yalnızca Windows üzerinde test edilmiştir. Linux desteği için win32_helper.py düzenlemesi gerekebilir.
  



