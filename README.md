# MD-DeviceQC

Mikrodev cihaz kalite kontrolu (YOLOv5 + PySide6).

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
