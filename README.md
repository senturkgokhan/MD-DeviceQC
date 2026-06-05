# MD-DeviceQC
### YOLOv5 Tabanlı Endüstriyel Cihaz Kalite Kontrol Sistemi

MD-DeviceQC, Mikrodev Bilişim A.Ş. üretim hattından çıkan endüstriyel cihazların görsel kalite kontrolünü otomatikleştiren, YOLOv5 tabanlı gerçek zamanlı bir denetim sistemidir. Sistem; operatörün ürünü kameraya göstermesiyle çalışır ve ön yüz bileşen sayımı, etiket kontrolü ve barkod okuma aşamalarını sırasıyla gerçekleştirerek her cihaz için PASS veya FAIL kararı üretir ve sonucu seri numarasıyla birlikte kayıt altına alır.

## Neden Geliştirildi?

Endüstriyel cihaz üretiminde görsel kalite kontrol, geleneksel olarak üretim hattının sonunda eğitimli personel tarafından manuel şekilde yürütülmektedir. Bu yöntem;
- İnsan dikkatine bağlı olduğundan zaman içinde hata oranı artmaktadır
- Kontrol süreci kişiden kişiye farklılık göstermekte, standart dışı kalmaktadır
- Hatalı geçen cihazlar izlenebilir biçimde kayıt altına alınamamaktadır
- Üretim hızı arttıkça manuel kontrol bir darboğaz haline gelmektedir

MD-DeviceQC bu sorunları ortadan kaldırmak amacıyla geliştirilmiştir. Her denetimi aynı standartla gerçekleştirir ve sonucu seri numarasıyla birlikte otomatik olarak kayıt altına alır.

## Özgünlük

Bu proje aşağıdaki açılardan özgün bir çalışmadır:

- **Kendine özgü dataset:** Tüm eğitim verisi Mikrodev üretim ortamında, gerçek DM100 ve XIO110 cihazları üzerinde bizzat toplanmıştır. Kamuya açık hiçbir veri kümesi kullanılmamıştır.
- **Sektöre özel etiketleme:** Endüstriyel RTU/PLC cihazlarına yönelik 10 sınıf ve 4.062 görüntüden oluşan özgün bir dataset oluşturulmuştur.Tüm görseller Roboflow platformu üzerinde her bileşen tek tek, bizzat etiketlenmiştir.
- **Özgün pipeline:** YOLOv5 + OpenCV + pyzbar + PySide6 bileşenlerinin bu tip endüstriyel cihazlar için entegrasyonu özgün bir yazılım mimarisi ortaya koymaktadır.
- **Üretime uygun tasarım:** Sistem bileşenleri ve denetim adımları, Mikrodev üretim hattının gerçek koşulları analiz edilerek belirlenmiştir:
  - Kamera, üretim ortamındaki çekim mesafesi ve görüntü kalitesi kriterleri göz önünde bulundurularak seçilmiştir
  - Operatörün cihazı elle tutarak kameraya göstermesi şeklinde çalışan bir denetim akışı tasarlanmıştır
  - Tutarlı görüntü kalitesi için özel bir aydınlatma ortamı oluşturulmuştur
  - Sistem, mevcut üretim akışını bozmadan entegre olacak şekilde tasarlanmıştır
- **Uygulanabilir çözüm:** Düşük maliyetli ve kurulumu kolay yapısıyla küçük ve orta ölçekli Türk endüstriyel üreticilere doğrudan uygulanabilir bir kalite kontrol çözümü sunmaktadır.

## Özellikler

  - 🎯 **Gerçek zamanlı bileşen tespiti** — YOLOv5s modeli ile 10 sınıf
  - 🔍 **Otomatik cihaz tanıma** — DM100 / XIO110 komponent sayım skoruna göre otomatik belirlenir, manuel seçim de desteklenir
  - 📋 **Üç aşamalı denetim akışı** — Yüzey Kontrolü → Etiket → Barkod → Sonuc
  - ⚠️ **FAIL yönlendirme** — Herhangi bir aşama başarısız olsa bile barkod okunur, izlenebilirlik hiç kopmaz
  - 💾 **Otomatik kayıt** — Tüm sonuçlar seri numarasıyla birlikte SQLite veritabanına kaydedilir
  - 🖥️ **Modern arayüz** — PySide6 ile canlı kamera, durum paneli ve geçmiş sonuçlar tek ekranda
  - ⚙️ **Kolay konfigürasyon** — Yeni cihaz tipi eklemek için sadece YAML dosyası güncellenir, kod değişikliği gerekmez.
    
## Sistem Önizlemesi
### Main Menu
Uygulamanın açılış ekranı. Start Inspection butonu ile kalite kontrol süreci başlatılır, Test Results ile daha önce denetlenen cihazların sonuçları görüntülenir, Settings ile kamera ve model ayarları yapılandırılır.

<img width="497" height="644" alt="Ekran görüntüsü 2026-06-05 092314" src="https://github.com/user-attachments/assets/78e5ff97-2282-464e-8e18-02070afb1603" />

### Start Inspection
Kalite kontrol sürecinin ana ekranı. Sol panelde mevcut aşama, kalan süre ve bileşen sayım detayları yer alır. Ortada canlı kamera görüntüsü ve YOLO tespit kutuları, sağ panelde ise son denetim sonuçları anlık olarak görünür.
<img width="1911" height="1013" alt="Ekran görüntüsü 2026-06-05 092223" src="https://github.com/user-attachments/assets/b0638a9b-cece-421b-a15d-c2028bab34d4" />

### Test Results
Denetlenen tüm cihazların geçmişi; tarih, cihaz tipi, seri numarası, PASS/FAIL sonucu ve varsa başarısız olan aşama bilgisiyle birlikte listelenir.
<img width="1156" height="782" alt="Ekran görüntüsü 2026-06-05 092241" src="https://github.com/user-attachments/assets/854cbb3d-4ec7-4eef-bd71-c1f715a9d075" />

### Settings
Sistem üç ayar sekmesinden oluşur. Camera sekmesinde kamera kaynağı ve FPS gösterimi ayarlanır. Model sekmesinde YOLO ağırlık dosyası, GPU/CPU seçimi ve görüntü boyutu belirlenir. Detection sekmesinde ise güven eşiği, IOU değeri ve strict mod yapılandırılır. Tüm değişiklikler bir sonraki denetimde geçerli olur.
<table>
  <tr>
    <td align="center"><b>Camera</b><br/><img width="280" src="https://github.com/user-attachments/assets/23fcd4c0-613b-492d-9298-01dba4f9d4a4"/></td>
    <td align="center"><b>Model</b><br/><img width="280" src="https://github.com/user-attachments/assets/9910b0f5-2edd-4603-a4eb-03e803393423"/></td>
    <td align="center"><b>Detection</b><br/><img width="280" src="https://github.com/user-attachments/assets/dca5e921-65b8-4355-a7ad-d2a5718b42ab"/></td>
  </tr>
</table>
















## Repo icerigi

| Oge | Aciklama |
|-----|----------|
| `yolov5/` | QC uygulamasi (`qc_engine.py`, `qc_launcher.py`, profiller, veritabani araclari) |
| `best.pt` | Birincil egitilmis model agirligi (~14 MB) |
| `best1.pt` | Ikincil model agirligi (~22 MB) |

## Calistirma

```bash
cd yolov5
pip install -r requirements.txt
python qc_launcher.py
```

Model dosyalari repo kokundedir; ayarlarda `weights` yolu genelde `../best.pt` olmalidir.
