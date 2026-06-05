
#Barkod tespit ve pyzbar kütüphanesi ile OCR işlemi yapar. Pyzbar başarısızsa, çizgi deseninden barkod bölgesi tahmin eder ve OCR'ı o bölgede dener. Ayrıca etiket benzeri bölgeleri de tespit eder.#

import cv2
import numpy as np
from pyzbar import pyzbar
import pytesseract

# =============== AYARLAR ===============
MIN_AREA_FRAC   = 0.005
MAX_AREA_FRAC   = 0.70
MIN_RECT_SCORE  = 0.58
ASPECT_MIN_MAX  = (0.8, 10.0)
BORDER_MARGIN   = 6
WHITE_S_MAX     = 85
WHITE_V_MIN     = 165
BARCODE_MARGIN  = 6
DEBUG_START     = False

# Barkod-benzeri çizgi tespiti (heuristic) için
STRIPE_MIN_PEAKS       = 10      # min. tekrarlı çizgi sayısı (ortalama)
STRIPE_PROFILE_SMOOTH  = 7       # 1D smoothing kernel
STRIPE_SCORE_VAR_MIN   = 8.0     # profil varyansı alt sınır
BARLIKE_MIN_AR         = 2.0     # min. uzun/kısa kenar oranı
BARLIKE_AREA_FRAC_MIN  = 0.001   # görüntüye göre minimum alan
BARLIKE_AREA_FRAC_MAX  = 0.40    # görüntüye göre maksimum alan

# ---->> Buraya tek bir resim yolu ver
IMG_PATH = r"C:\Users\sentu\OneDrive\Resimler\Camera Roll\WIN_20250818_10_49_24_Pro_out.jpg"

# =============== BARKOD ÇİZİMİ ===============
def draw_barcode(frame, obj, text):
    pts = obj.polygon
    pts = [(p.x, p.y) if hasattr(p, "x") else (p[0], p[1]) for p in pts]
    if len(pts) >= 4:
        cnt = np.array(pts, dtype=np.int32)
        rect = cv2.minAreaRect(cnt.astype(np.float32))
        box  = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(frame, [box], True, (0,255,0), 2)
    else:
        x, y, w, h = obj.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
    print(text)

# =============== YARDIMCI ===============
def union_barcode_rect(barcodes):
    if not barcodes:
        return None
    xs = min(b.rect[0] for b in barcodes)
    ys = min(b.rect[1] for b in barcodes)
    xe = max(b.rect[0] + b.rect[2] for b in barcodes)
    ye = max(b.rect[1] + b.rect[3] for b in barcodes)
    return (xs, ys, xe - xs, ye - ys)

def rect_contains(outer, inner, margin=6):
    x, y, w, h = outer
    xi, yi, wi, hi = inner
    return (xi >= x - margin and yi >= y - margin and
            xi + wi <= x + w + margin and yi + hi <= y + h + margin)

def is_quadrilateral(cnt, angle_tol=40):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        return False
    def angle(p0, p1, p2):
        v1 = p0 - p1; v2 = p2 - p1
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    pts = approx[:,0,:]
    for i in range(4): 
        a = angle(pts[(i-1)%4], pts[i], pts[(i+1)%4])
        if abs(a-90) > angle_tol:
            return False
    return True

def crop_rotated_rect(img, rect):
    # rect: (center(x,y), (w,h), angle)
    (cx, cy), (w, h), angle = rect
    w, h = int(w), int(h)
    if w <= 0 or h <= 0:
        return None
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    warped = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    x0 = int(cx - w/2); y0 = int(cy - h/2)
    x1 = x0 + w;       y1 = y0 + h
    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
        # Emniyetli kırpma
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(img.shape[1], x1); y1 = min(img.shape[0], y1)
    roi = warped[y0:y1, x0:x1]
    return roi

# =============== BARKOD-BENZERİ ÇİZGİ TESPİTİ ===============
def detect_barcode_like(frame, debug=False):
    """pyzbar başarısızsa, çizgi deseninden barkod bölgesi tahmin eder.
       Dönen değer: (brect, rect_points) veya (None, None)"""
    H, W = frame.shape[:2]
    area_total = W * H
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # Hem dikey hem yatay kenarları kontrol et (rotasyon toleransı)
    gradx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradx = cv2.convertScaleAbs(gradx)
    grady = cv2.convertScaleAbs(grady)

    # İki yönden biri güçlü ise al
    grad = cv2.max(gradx, grady)
    _, bw  = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Çizgileri bloklaştır (hem yatay hem dikey çekirdeklerle birleştir)
    kern1 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 21))
    close1 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kern1, iterations=1)
    close2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kern2, iterations=1)
    merged = cv2.max(close1, close2)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        rect = cv2.minAreaRect(c)            # (center, (w,h), angle)
        (w, h) = rect[1]
        if w < 5 or h < 5:
            continue

        long_side = max(w, h)
        short_side = min(w, h)
        ar = (long_side + 1e-6) / (short_side + 1e-6)

        area = w * h
        if not (BARLIKE_AREA_FRAC_MIN * area_total <= area <= BARLIKE_AREA_FRAC_MAX * area_total):
            continue
        if ar < BARLIKE_MIN_AR:
            continue

        # Rotated ROI üzerinde "şerit tekrarı" kontrolü (profil varyansı + tepe sayısı)
        roi = crop_rotated_rect(grad, rect)
        if roi is None or roi.size == 0:
            continue

        # Uygun yöne indirgeme (uzun taraf yatay olacak şekilde yeniden örnekle)
        rh, rw = roi.shape[:2]
        if rh > rw:
            roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
            rh, rw = roi.shape[:2]

        # 1D kolon profili: her sütunun ortalaması
        prof = roi.mean(axis=0)
        # yumuşat
        k = max(3, STRIPE_PROFILE_SMOOTH | 1)  # tek sayı
        prof_s = cv2.blur(prof.reshape(1, -1).astype(np.float32), (k,1)).ravel()

        # Tepe/çukur sayısı ~ şerit tekrarı
        # Basit sıfır geçiş benzeri: türev işareti değişimi
        d = np.diff(prof_s)
        sign = np.sign(d)
        sign[sign == 0] = 1
        zero_cross = np.where(np.diff(sign) != 0)[0]
        peaks = len(zero_cross)

        var = np.var(prof_s)

        if peaks >= STRIPE_MIN_PEAKS and var >= STRIPE_SCORE_VAR_MIN:
            score = float(area) * (var + 1.0) * (1.0 + 0.02 * peaks)
            box = cv2.boxPoints(rect).astype(np.int32)
            x, y, w_b, h_b = cv2.boundingRect(box)
            cand = (score, (x, y, w_b, h_b), box)
            if best is None or cand[0] > best[0]:
                best = cand

    if debug:
        cv2.imshow("grad", grad)
        cv2.imshow("bw", bw)
        cv2.imshow("merged", merged)

    if best is None:
        return None, None
    return best[1], best[2]  # brect, box_points


# =============== ETİKET ADAYI ===============
def detect_label_candidate(frame, brect=None, debug=False):
    H, W = frame.shape[:2]
    total_area = W * H

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    _, th  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
    _, thi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    kernel_h   = cv2.getStructuringElement(cv2.MORPH_RECT, (25,7))
    thc  = cv2.morphologyEx(th,  cv2.MORPH_CLOSE, kernel_big, iterations=1)
    thci = cv2.morphologyEx(thi, cv2.MORPH_CLOSE, kernel_h,   iterations=1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_like = cv2.inRange(hsv, (0, 0, WHITE_V_MIN), (179, WHITE_S_MAX, 255))

    mask = cv2.bitwise_or(cv2.bitwise_or(thc, thci), white_like)

    if debug:
        cv2.imshow("th", th); cv2.imshow("thi", thi)
        cv2.imshow("thc", thc); cv2.imshow("thci", thci)
        cv2.imshow("white_like", white_like); cv2.imshow("mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for c in contours:
        cnt_area = cv2.contourArea(c)
        if cnt_area < 400:
            continue
        x,y,w,h = cv2.boundingRect(c)

        if x < BORDER_MARGIN or y < BORDER_MARGIN or (x+w) > (W - BORDER_MARGIN) or (y+h) > (H - BORDER_MARGIN):
            continue

        area = w*h
        if not (MIN_AREA_FRAC * total_area <= area <= MAX_AREA_FRAC * total_area):
            continue

        rect_score = float(cnt_area) / float(area + 1e-6)
        if rect_score < MIN_RECT_SCORE:
            continue

        if not is_quadrilateral(c, angle_tol=40):
            continue

        asp = w / float(h)
        asp_min, asp_max = ASPECT_MIN_MAX
        asp_ok = asp_min <= asp <= asp_max

        if brect is not None and not rect_contains((x,y,w,h), brect, margin=BARCODE_MARGIN):
            continue

        roi_hsv = hsv[y:y+h, x:x+w]
        if roi_hsv.size == 0:
            continue
        sat = roi_hsv[...,1]; val = roi_hsv[...,2]
        lowS  = (sat <= WHITE_S_MAX).mean()
        highV = (val >= WHITE_V_MIN).mean()
        paper_like = 0.4*lowS + 0.6*highV

        score = area * rect_score * (1.2 if asp_ok else 0.8) * (0.5 + 0.5*paper_like)

        if best is None or score > best[0]:
            best = (score, (x,y,w,h))

    if best is None:
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), 1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 1200:
                continue
            x,y,w,h = cv2.boundingRect(c)
            area = w*h
            if not (MIN_AREA_FRAC * total_area <= area <= MAX_AREA_FRAC * total_area):
                continue
            if not is_quadrilateral(c, angle_tol=45):
                continue
            if brect is not None and not rect_contains((x,y,w,h), brect, margin=BARCODE_MARGIN):
                continue
            score = area
            if best is None or score > best[0]:
                best = (score, (x,y,w,h))

    if best is None:
        return None, 0.0, (mask if debug else None)
    return best[1], float(best[0]), (mask if debug else None)

# =============== TEK GÖRSEL İŞLE ===============
def run_on_image(img_path, debug=DEBUG_START):
    img = cv2.imread(img_path)  
    brect = None
    if img is None:
        raise SystemExit(f"Görsel bulunamadı: {img_path}")

    # 1) pyzbar: barkodu oku/çiz
    barcodes = pyzbar.decode(img)
    for obj in barcodes:
        data = obj.data.decode("utf-8", errors="ignore")
        print(f"Barkod: {data}")
        draw_barcode(img, obj, data)

        # Barkodun etrafındaki alanı kırp OCR işlemi
        x, y, w, h = obj.rect
        roi = img[max(0, y-100):y+h+200, max(0, x-100):x+w+200]  # Barkod çevresinden biraz geniş kırp

        # OCR ile metin oku
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Gürültü azalt
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        # Kontrastı otomatik düzelt
        gray = cv2.equalizeHist(gray)
        # Dikey veya yatay eğiklik varsa düzleştir (isteğe bağlı)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # Net eşikleme
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        custom_config = r'--oem 3 --psm 11'
        text = pytesseract.image_to_string(gray, lang='eng+tur', config=custom_config)

        print("OCR İyileştirilmiş Çıktı:\n", text)
            
    print("Bulunan barkod sayısı:", len(barcodes))

    # 2) Barkod dikdörtgenini belirle (pyzbar varsa union; yoksa OpenCV stripe fallback)
    
    if brect is None:
        # Barkod okunamadı ⇒ barkod-benzeri desen ara
        brect_fallback, box_pts = detect_barcode_like(img, debug=debug)
        if brect_fallback is not None:
            brect = brect_fallback
            brect_box = box_pts

            # Görsel geri bildirim
            if brect_box is not None:
                cv2.polylines(img, [brect_box], True, (0, 180, 255), 2)

            # ===== Fallback OCR işlemi =====
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

            x, y, w, h = brect
            roi = img[max(0, y-30):y+h+60, max(0, x-30):x+w+60]

            if roi is not None and roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                text = pytesseract.image_to_string(gray, lang='eng')
                print("Fallback OCR ÇIKTISI:\n", text)
            else:
                print("Fallback ROI boş, OCR yapılamadı")

    # 3) Etiket: her durumda ara
    cand_box, cand_score, _ = detect_label_candidate(img, brect=brect, debug=debug) if brect is not None \
                              else detect_label_candidate(img, brect=None, debug=debug)

    if cand_box is not None:
        x,y,w,h = cand_box
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(img, "Etiket", (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow("Sonuc", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        run_on_image(IMG_PATH, debug=DEBUG_START)
    except OSError as e:
        print("Hata:", e)
        print("Muhtemelen ZBar kurulu degil. Windows icin 'choco install zbar' deneyin.")
