# Scripts — Geliştirme Süreci ve Denemeler

Bu klasördeki dosyalar geliştirme ve araştırma sürecinde kullanılmıştır. Nihai sistem doğrudan qc_engine.py ve qc_launcher.py üzerinden çalışmaktadır.
---

## 1- Jetson Nano Denemeleri
Proje başlangıcında sistem, NVIDIA Jetson Nano 4GB üzerinde deploy edilmesi planlanmıştır. Bu amaçla YOLOv5 modeli ve IMX219 kamera Jetson Nano üzerinde test edilmiştir.

**Sonuç:**  
YOLOv5 modeli ve kamera aynı anda çalıştırıldığında Jetson Nano 4GB üzerindeki bellek ve işlem yükü kritik sınırları aşmıştır. Kamera FPS değeri aşırı düşmüş ve sistem donmuştur. Bu nedenle Jetson Nano bu proje kapsamında uygun bir platform olarak değerlendirilmemiştir.

**Gelecek:**  
Daha yüksek kapasiteli bir edge board (örn. Jetson Orin Nano) ile deploy yapılması planlanmaktadır.

## 2- YOLOv8 Denemeleri

Sistem geliştirme sürecinde YOLOv5s ile karşılaştırmalı olarak YOLOv8s modeli de aynı dataset üzerinde eğitilmiştir.

**Eğitim Denemeleri:**

| Model | Epoch | Görüntü Boyutu | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|-------|----------------|---------|--------------|-----------|--------|
| YOLOv8s | 150 | 640 | 0.987 | 0.835 | 0.974 | 0.983 |
| YOLOv8s | 200 | 640 | 0.980 | 0.833 | 0.973 | 0.985 |

**Sonuç:**  
YOLOv8s metrik olarak iyi sonuçlar üretmiştir ancak sahada PC ve test laptobu üzerinde YOLOv5s'e kıyasla daha yavaş çalışmıştır. Mevcut sistem mimarisiyle entegrasyonu da daha karmaşık olmuştur. Bu nedenle nihai model olarak YOLOv5s  tercih edilmiştir.

## 3- Camera.py — Dataset Görüntü Toplama
Dataset oluşturma sürecinde kullanılan görüntü toplama scriptidir. Kameradan gerçek zamanlı görüntü alarak belirtilen klasöre kaydeder.

**Amaç:**  
Mikrodev üretim ortamında DM100 ve XIO110 cihazlarının farklı açı, mesafe ve ışık koşullarında görüntülerini toplamak için kullanılmıştır.

**Kullanım:**
```bash
python Camera.py
```
## 4- OCR3.py — Etiket Kutusu Tespiti
Klasik görüntü işleme yöntemleriyle etiket yüzeyindeki kutuları tespit eden geliştirme scriptidir.

**Amaç:**  
YOLO modelinin etiket tespitini desteklemek amacıyla OpenCV tabanlı klasik CV teknikleri kullanılarak etiket kutularının konumu ve sınırları tespit edilmektedir.

**Kullanılan Yöntemler:**
- HSV renk uzayında beyaz/açık renkli yüzey maskesi
- Otsu eşikleme ile ikili görüntü oluşturma
- Morfolojik işlemler ile gürültü temizleme
- Kontur tespiti ve dörtgen filtresi
- Barkod konumu ile etiket kutusunun örtüşme kontrolü

**Sistem içindeki yeri:**  
Bu script, BARCODE aşamasında barkod okunduktan sonra etiket kutusunun varlığını doğrulamak için kullanılmıştır. Geliştirme sürecinde test edilmiş, nihai sistemde `qc_engine.py` içine entegre edilmiştir.

## 5- Colab_Training_Code_Yolov5.ipynb — Model Eğitimi
YOLOv5s modelinin Google Colab üzerinde eğitilmesi için kullanılan Jupyter Notebook dosyasıdır.

**Amaç:**  
Roboflow'dan export edilen dataset ile YOLOv5s modelini Google Colab GPU ortamında eğitmek için kullanılmıştır.

**Eğitim Parametreleri:**
- Model: YOLOv5s
- Epoch: 150
- Görüntü Boyutu: 832×832
- Batch: 8
- Optimizer: SGD (patience=30)
- Transfer Learning: yolov5s.pt ön eğitimli ağırlıklar

**İçerik:**
- Gerekli kütüphanelerin kurulumu
- Roboflow dataset bağlantısı ve indirme
- Model eğitimi ve doğrulama
- Eğitim grafiklerinin görselleştirilmesi
- En iyi modelin (best.pt) kaydedilmesi

