

import argparse
import pathlib
import sys
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

FILE = Path(__file__).resolve()
YOLOV5_ROOT = FILE.parent
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]

try:
    from pyzbar import pyzbar

    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False
    pyzbar = None  # type: ignore[assignment, misc]

from models.experimental import attempt_load  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.general import non_max_suppression, scale_coords as scale_boxes  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

# --- Etiket adayı (OCR2 ile uyumlu sabitler) ---
MIN_AREA_FRAC = 0.005
MAX_AREA_FRAC = 0.70
MIN_RECT_SCORE = 0.58
ASPECT_MIN_MAX = (0.8, 10.0)
BORDER_MARGIN = 6
WHITE_S_MAX = 85
WHITE_V_MIN = 165
BARCODE_MARGIN = 6


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
    return (
        xi >= x - margin
        and yi >= y - margin
        and xi + wi <= x + w + margin
        and yi + hi <= y + h + margin
    )


def is_quadrilateral(cnt, angle_tol=40):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        return False

    def angle(p0, p1, p2):
        v1 = p0 - p1
        v2 = p2 - p1
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    pts = approx[:, 0, :]
    for i in range(4):
        a = angle(pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4])
        if abs(a - 90) > angle_tol:
            return False
    return True


def detect_label_candidate(frame, brect=None, debug=False):
    """brect varsa etiket kutusu barkodu kapsamalı (OCR2 ile aynı)."""
    H, W = frame.shape[:2]
    total_area = W * H

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    thc = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_big, iterations=1)
    thci = cv2.morphologyEx(thi, cv2.MORPH_CLOSE, kernel_h, iterations=1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_like = cv2.inRange(hsv, (0, 0, WHITE_V_MIN), (179, WHITE_S_MAX, 255))

    mask = cv2.bitwise_or(cv2.bitwise_or(thc, thci), white_like)

    if debug:
        cv2.imshow("th", th)
        cv2.imshow("thi", thi)
        cv2.imshow("thc", thc)
        cv2.imshow("thci", thci)
        cv2.imshow("white_like", white_like)
        cv2.imshow("mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for c in contours:
        cnt_area = cv2.contourArea(c)
        if cnt_area < 400:
            continue
        x, y, w, h = cv2.boundingRect(c)

        if (
            x < BORDER_MARGIN
            or y < BORDER_MARGIN
            or (x + w) > (W - BORDER_MARGIN)
            or (y + h) > (H - BORDER_MARGIN)
        ):
            continue

        area = w * h
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

        if brect is not None and not rect_contains((x, y, w, h), brect, margin=BARCODE_MARGIN):
            continue

        roi_hsv = hsv[y : y + h, x : x + w]
        if roi_hsv.size == 0:
            continue
        sat = roi_hsv[..., 1]
        val = roi_hsv[..., 2]
        lowS = (sat <= WHITE_S_MAX).mean()
        highV = (val >= WHITE_V_MIN).mean()
        paper_like = 0.4 * lowS + 0.6 * highV

        score = area * rect_score * (1.2 if asp_ok else 0.8) * (0.5 + 0.5 * paper_like)

        if best is None or score > best[0]:
            best = (score, (x, y, w, h))

    if best is None:
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 1200:
                continue
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if not (MIN_AREA_FRAC * total_area <= area <= MAX_AREA_FRAC * total_area):
                continue
            if not is_quadrilateral(c, angle_tol=45):
                continue
            if brect is not None and not rect_contains((x, y, w, h), brect, margin=BARCODE_MARGIN):
                continue
            score = area
            if best is None or score > best[0]:
                best = (score, (x, y, w, h))

    if best is None:
        return None, 0.0, (mask if debug else None)
    return best[1], float(best[0]), (mask if debug else None)


def draw_barcode(frame, obj, text: str):
    pts = obj.polygon
    pts = [(p.x, p.y) if hasattr(p, "x") else (p[0], p[1]) for p in pts]
    if len(pts) >= 4:
        cnt = np.array(pts, dtype=np.int32)
        rect = cv2.minAreaRect(cnt.astype(np.float32))
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(frame, [box], True, (0, 255, 128), 2)
        x, y = int(box[:, 0].min()), int(box[:, 1].min())
    else:
        x, y, w, h = obj.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 128), 2)
    short = text if len(text) <= 32 else text[:29] + "..."
    cv2.putText(
        frame,
        short,
        (x, max(0, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 200, 100),
        1,
        cv2.LINE_AA,
    )


def parse_opt():
    p = argparse.ArgumentParser()
    default_weights = YOLOV5_ROOT.parent / "best.pt"
    p.add_argument("--weights", type=str, default=str(default_weights), help="model .pt yolu")
    p.add_argument("--source", type=str, default="0", help="kamera indeksi veya video yolu")
    p.add_argument("--device", default="", help="cuda veya cpu; boş = otomatik")
    p.add_argument("--img-size", type=int, default=640, help="inference giriş boyutu")
    p.add_argument("--conf", type=float, default=0.25, help="güven eşiği")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    p.add_argument("--no-dshow", action="store_true", help="CAP_DSHOW kullanma")
    p.add_argument("--no-label", action="store_true", help="barkod/etiket tespitini kapat (sadece YOLO)")
    p.add_argument("--label-debug", action="store_true", help="etiket mask debug pencereleri (başlangıç açık)")
    p.add_argument("--cam-width", type=int, default=1280, help="Jetson CSI kamera genişliği")
    p.add_argument("--cam-height", type=int, default=720, help="Jetson CSI kamera yüksekliği")
    p.add_argument("--cam-fps", type=int, default=30, help="Jetson CSI kamera FPS")
    p.add_argument("--sensor-mode", type=int, default=4, help="IMX219 sensor mode; 4 = 1280x720")
    return p.parse_args()


def main():
    opt = parse_opt()
    weights_path = Path(opt.weights).expanduser().resolve()
    if not weights_path.is_file():
        print(f"Ağırlık dosyası bulunamadı: {weights_path}")
        sys.exit(1)

    use_label = not opt.no_label and HAS_PYZBAR
    if not opt.no_label and not HAS_PYZBAR:
        print("Uyarı: pyzbar yüklü değil; barkod/etiket devre dışı. pip install pyzbar (+ ZBar)")

    label_debug = bool(opt.label_debug)

    device = select_device(opt.device if opt.device else ("0" if torch.cuda.is_available() else "cpu"))
    model = attempt_load(str(weights_path), device=device)
    model.eval()
    names = model.names
    names_list = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)

    img_size = opt.img_size
    conf_thres = opt.conf
    iou_thres = opt.iou

    if device.type != "cpu":
        model.half()
        torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        if device.type != "cpu":
            dummy = dummy.half()
        _ = model(dummy)

    src = opt.source
    try:
        int(src)
        print(
            f"Jetson IMX219 kamera OpenCV GStreamer ile açılıyor: "
            f"{opt.cam_width}x{opt.cam_height}@{opt.cam_fps}, sensor-mode={opt.sensor_mode}"
        )

        gst_pipeline = (
            f"nvarguscamerasrc sensor-mode={opt.sensor_mode} ! "
            f"video/x-raw(memory:NVMM), width={opt.cam_width}, height={opt.cam_height}, framerate={opt.cam_fps}/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink max-buffers=1 drop=true sync=false"
        )

        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    except ValueError:
        cap = cv2.VideoCapture(str(src))

    if not cap.isOpened():
        print(f"Kaynak açılamadı: {opt.source}")
        print("Kamera test komutu: gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! xvimagesink")
        sys.exit(1)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    win_name = "YOLOv5 + Etiket - Result2"
    cv2.namedWindow(win_name)

    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thickness = 0.5, 1
    color_text = (0, 0, 0)
    PANEL_W = 300
    PANEL_BG = (200, 200, 200)
    PAD_X, PAD_Y = 10, 26

    print("q=çıkış, d=etiket debug mask aç/kapa")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        img0 = frame.copy()
        n_barcodes = 0
        has_label_box = False

        if use_label:
            barcodes = pyzbar.decode(frame)
            n_barcodes = len(barcodes)
            for obj in barcodes:
                data = obj.data.decode("utf-8", errors="ignore")
                draw_barcode(img0, obj, data)

            brect = union_barcode_rect(barcodes) if barcodes else None
            cand_box, _, _ = detect_label_candidate(frame, brect=brect, debug=label_debug)
            if cand_box is not None:
                has_label_box = True
                x, y, w, h = cand_box
                cv2.rectangle(img0, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    img0,
                    "Etiket",
                    (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        lb = letterbox(frame, new_shape=img_size)
        img_resized = lb[0] if isinstance(lb, tuple) else lb

        img = img_resized[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).to(device, non_blocking=True)
        img_tensor = (img_tensor.half() if device.type != "cpu" else img_tensor.float()) / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)

        class_counts = {i: 0 for i in range(len(names_list))}
        class_conf_sum = {i: 0.0 for i in range(len(names_list))}

        if pred[0] is not None and len(pred[0]):
            pred[0][:, :4] = scale_boxes(img.shape[1:], pred[0][:, :4], img0.shape).round()

            for *xyxy, conf, cls in pred[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                cid = int(cls)
                class_counts[cid] += 1
                class_conf_sum[cid] += float(conf)
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{names_list[cid]} {float(conf):.2f}"
                cv2.putText(
                    img0,
                    label,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 180, 0),
                    1,
                    cv2.LINE_AA,
                )

        H, W = img0.shape[:2]
        panel = np.zeros((H, PANEL_W, 3), dtype=np.uint8)
        panel[:] = PANEL_BG

        x_offset, y_offset = PAD_X, PAD_Y
        total_det = sum(class_counts.values())
        cv2.putText(panel, f"YOLO: {total_det}", (x_offset, y_offset), font, scale, color_text, thickness, cv2.LINE_AA)
        y_offset += 24

        if use_label:
            cv2.putText(
                panel,
                f"Barkod: {n_barcodes}",
                (x_offset, y_offset),
                font,
                scale,
                color_text,
                thickness,
                cv2.LINE_AA,
            )
            y_offset += 24
            et = "Etiket: var" if has_label_box else "Etiket: -"
            cv2.putText(panel, et, (x_offset, y_offset), font, scale, color_text, thickness, cv2.LINE_AA)
            y_offset += 26
        elif not opt.no_label and not HAS_PYZBAR:
            cv2.putText(
                panel,
                "pyzbar yok",
                (x_offset, y_offset),
                font,
                0.45,
                (0, 0, 160),
                1,
                cv2.LINE_AA,
            )
            y_offset += 24

        for cid, cname in enumerate(names_list):
            cnt = class_counts[cid]
            avg = (class_conf_sum[cid] / cnt) if cnt > 0 else 0.0
            line = f"{cname}: {cnt} | {avg:.2f}"
            cv2.putText(panel, line, (x_offset, y_offset), font, scale, color_text, thickness, cv2.LINE_AA)
            y_offset += 23
            if y_offset > H - 12:
                break

        combined = np.concatenate([panel, img0], axis=1)
        cv2.imshow(win_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d") and use_label:
            label_debug = not label_debug
            if not label_debug:
                for wn in ("th", "thi", "thc", "thci", "white_like", "mask"):
                    try:
                        cv2.destroyWindow(wn)
                    except cv2.error:
                        pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
