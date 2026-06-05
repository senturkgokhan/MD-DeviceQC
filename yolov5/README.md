# MD-DeviceQC

Mikrodev endüstriyel cihazlar için YOLOv5 tabanlı görüntülü kalite kontrol sistemi (DM100, XIO110).

## Çalıştırma

```bash
pip install -r requirements.txt
python qc_launcher.py
```

Sadece OpenCV motoru:

```bash
python qc_engine.py --windowed
```

Eski Tkinter launcher: `python Scripts/app.py`

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
