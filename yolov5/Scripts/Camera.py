
#Dataset oluşturmak için kamera ile fotoğraf çekme kodu#

import cv2
import os
import time
import re

# === Ayarlar ===
save_dir = r"C:\Users\sentu\OneDrive\Desktop\negatif-gorseller"
os.makedirs(save_dir, exist_ok=True)

fps = 5
delay = 1 / fps
start_delay = 5

# === Kayıt numarasını kaldığı yerden başlat === 
existing_files = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
pattern = r"negatif_(\d+)\.jpg"
existing_nums = [int(re.findall(pattern, f)[0]) for f in existing_files if re.findall(pattern, f)]

start_index = max(existing_nums) + 1 if existing_nums else 1
frame_count = start_index

# === Kamera başlat ===
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

print(f"📸 Kamera aktif. {fps} FPS ile kayıt yapılıyor. 'q' tuşu ile çıkabilirsin.")

print(f"⏳ Kamera önizlemesi açıldı. {start_delay} saniye sonra çekim başlayacak.")

countdown_start = time.time()
while time.time() - countdown_start < start_delay:
    ret, frame = cap.read()
    if not ret:
        break

    remaining = int(start_delay - (time.time() - countdown_start)) + 1
    preview = frame.copy()
    cv2.putText(
        preview,
        f"Cekim basliyor: {remaining} sn",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )
    cv2.imshow("Kamera", preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⛔ Kullanıcı tarafından durduruldu.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    filename = f"negatif_{frame_count:05d}.jpg"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_count += 1

    # Ekranda göster (isteğe bağlı)
    cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⛔ Kullanıcı tarafından durduruldu.")
        break

    # FPS ayarı
    elapsed = time.time() - start
    sleep_time = max(0, delay - elapsed)
    time.sleep(sleep_time)

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Toplam {frame_count - start_index} yeni fotoğraf kaydedildi → {save_dir}")
