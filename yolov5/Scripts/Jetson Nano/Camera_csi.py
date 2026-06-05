# -*- coding: utf-8 -*-

# CSI kamera kullanarak görüntü kaydetme scripti. Kayıtlar "negatif_00001.jpg", "negatif_00002.jpg" şeklinde isimlendirilir ve belirtilen klasöre kaydedilir. Kayıt sırasında kameradan canlı görüntü de gösterilir. Çıkmak için 'q' tuşuna basabilirsiniz.

import cv2
import os
import time
import re

# === Ayarlar ===
save_dir = os.path.expanduser("~/Desktop/Dataset")
os.makedirs(save_dir, exist_ok=True)

fps = 10
delay = 1 / fps

# === Kayıt numarasını kaldığı yerden başlat ===
existing_files = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]

pattern = r"negatif_(\d+)\.jpg"

existing_nums = [
    int(re.findall(pattern, f)[0])
    for f in existing_files
    if re.findall(pattern, f)
]

start_index = max(existing_nums) + 1 if existing_nums else 1
frame_count = start_index

# === CSI Kamera Pipeline ===
def gstreamer_pipeline():
    return (
        "nvarguscamerasrc ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )

# === Kamera başlat ===
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

print("CSI kamera aktif.")
print(f"Kayıt klasörü: {save_dir}")
print(f"Yaklaşık {fps} FPS ile kayıt yapılıyor.")
print("Çıkmak için q tuşuna bas.")

while True:
    start = time.time()

    ret, frame = cap.read()

    if not ret:
        print("Frame alınamadı.")
        break

    # === Dosya adı oluştur ===
    filename = f"negatif_{frame_count:05d}.jpg"
    filepath = os.path.join(save_dir, filename)

    # === Görüntüyü kaydet ===
    cv2.imwrite(filepath, frame)

    print(f"Kaydedildi: {filename}")

    frame_count += 1

    # === Kamerayı göster ===
    cv2.imshow("CSI Kamera", frame)

    # === Çıkış kontrolü ===
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Kullanıcı tarafından durduruldu.")
        break

    # === FPS kontrolü ===
    elapsed = time.time() - start
    sleep_time = max(0, delay - elapsed)
    time.sleep(sleep_time)

# === Temizlik ===
cap.release()
cv2.destroyAllWindows()

print(f"\nToplam {frame_count - start_index} yeni görüntü kaydedildi.")
print(f"Konum: {save_dir}")
