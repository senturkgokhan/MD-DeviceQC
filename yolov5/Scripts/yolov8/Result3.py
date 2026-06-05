"""
Result3 (YOLOv8): Aşamalı endüstriyel kalite kontrolü
ön yüz → etiket → barkod → PASS/FAIL + SQLite.

Result2.py ile aynı akış; YOLO motoru Ultralytics YOLOv8.

Varsayılan ağırlık: C:\\Users\\sentu\\OneDrive\\Desktop\\MP110DetectionYoloV5_5\\best1.pt
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
import torch

from inspection_profiles import (
    build_name_to_count,
    filter_for_front_id,
    format_counts_line,
    format_mismatches,
    format_raw_front_lines,
    identify_device_from_components,
    load_profiles,
    validate_section,
)
from utils import db as dbm

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]

try:
    from pyzbar import pyzbar

    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False
    pyzbar = None  # type: ignore[assignment, misc]

try:
    from ultralytics import YOLO  # type: ignore

    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False
    YOLO = None  # type: ignore[assignment, misc]

# --- Etiket adayı (OCR2 ile uyumlu) ---
MIN_AREA_FRAC = 0.005
MAX_AREA_FRAC = 0.70
MIN_RECT_SCORE = 0.58
ASPECT_MIN_MAX = (0.8, 10.0)
BORDER_MARGIN = 6
WHITE_S_MAX = 85
WHITE_V_MIN = 165
BARCODE_MARGIN = 6

RESULTS_DIR = ROOT / "results"
DB_PATH = RESULTS_DIR / "devices.db"

PANEL_W = 360
HISTORY_PANEL_W = 300
HISTORY_ROW_LIMIT = 10
PHASE_LABEL_TR = {"FRONT": "On yuz", "LABEL": "Etiket", "BARCODE": "Barkod", "": ""}

PANEL_BG = (242, 244, 248)
PANEL_ACCENT = (25, 84, 166)
PANEL_ACCENT_2 = (14, 132, 115)
PANEL_WARN = (0, 140, 220)
PANEL_OK = (40, 160, 80)
PANEL_DANGER = (50, 50, 220)  # BGR kırmızı
COLOR_PASS_BG = (70, 175, 95)
COLOR_FAIL_BG = (55, 55, 230)
COLOR_TEXT = (35, 38, 45)
COLOR_MUTED = (105, 108, 118)

PAD_X, PAD_Y = 16, 20
STABLE_FRAMES_FRONT = 12
STABLE_FRAMES_LABEL = 8
RESULT_DISPLAY_SECONDS = 2.0


class Phase(Enum):
    FRONT = auto()
    LABEL = auto()
    BARCODE = auto()
    RESULT = auto()


PHASE_TITLES = {
    Phase.FRONT: "ON YUZU GOSTERIN",
    Phase.LABEL: "ETIKETLI YUZEYI GOSTERIN",
    Phase.BARCODE: "BARKODU GOSTERIN",
    Phase.RESULT: "SONUC",
}

PHASE_TIMEOUT_SEC = {
    Phase.FRONT: 20.0,
    Phase.LABEL: 15.0,
    Phase.BARCODE: 15.0,
}


@dataclass
class InspectionSession:
    session_id: str = ""
    device_key: str | None = None
    device_display: str = ""
    device_identified: bool = False
    front_by_name: dict[str, int] = field(default_factory=dict)
    front_ok: bool = False
    label_by_name_peak: dict[str, int] = field(default_factory=dict)
    label_cv_box_seen: bool = False
    label_ok: bool = False
    label_surfaces_done: int = 0
    serial_number: str = ""
    barcode_raw: str = ""
    barcode_ok: bool = False
    overall: str = "PENDING"
    front_status: str = "PENDING"
    label_status: str = "PENDING"
    barcode_status: str = "PENDING"
    fail_phase: str = ""
    fail_reasons: list[str] = field(default_factory=list)
    result_saved: bool = False


@dataclass
class YoloCache:
    boxes: list[tuple[int, int, int, int, float, int]] = field(default_factory=list)
    class_counts: dict[int, int] = field(default_factory=dict)
    class_conf_sum: dict[int, float] = field(default_factory=dict)


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
        for wn, im in (
            ("th", th),
            ("thi", thi),
            ("thc", thc),
            ("thci", thci),
            ("white_like", white_like),
            ("mask", mask),
        ):
            cv2.imshow(wn, im)

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
        asp_ok = ASPECT_MIN_MAX[0] <= asp <= ASPECT_MIN_MAX[1]
        if brect is not None and not rect_contains((x, y, w, h), brect, margin=BARCODE_MARGIN):
            continue
        roi_hsv = hsv[y : y + h, x : x + w]
        if roi_hsv.size == 0:
            continue
        sat = roi_hsv[..., 1]
        val = roi_hsv[..., 2]
        paper_like = 0.4 * (sat <= WHITE_S_MAX).mean() + 0.6 * (val >= WHITE_V_MIN).mean()
        score = area * rect_score * (1.2 if asp_ok else 0.8) * (0.5 + 0.5 * paper_like)
        if best is None or score > best[0]:
            best = (score, (x, y, w, h))

    if best is None:
        return None, 0.0
    return best[1], float(best[0])


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
    short = text if len(text) <= 28 else text[:25] + "..."
    cv2.putText(frame, short, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1, cv2.LINE_AA)


def _normalize_names(names) -> list[str]:
    if isinstance(names, dict):
        return [names[i] for i in range(len(names))]
    return list(names)


def draw_yolo_boxes(img0, boxes: list[tuple[int, int, int, int, float, int]], names_list: list[str]) -> None:
    for x1, y1, x2, y2, conf, cid in boxes:
        cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img0, f"{names_list[cid]} {conf:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1, cv2.LINE_AA)


def run_yolo_v8(model, frame, img_size: int, conf: float, iou: float, device: str, names_list: list[str], draw: bool = True):
    img0 = frame.copy()
    results = model.predict(source=frame, imgsz=img_size, conf=conf, iou=iou, device=device, verbose=False)
    r0 = results[0]
    boxes_obj = getattr(r0, "boxes", None)
    class_counts = {i: 0 for i in range(len(names_list))}
    class_conf_sum = {i: 0.0 for i in range(len(names_list))}
    boxes: list[tuple[int, int, int, int, float, int]] = []
    if boxes_obj is not None and len(boxes_obj) > 0:
        xyxy = boxes_obj.xyxy.detach().cpu().numpy()
        cls = boxes_obj.cls.detach().cpu().numpy().astype(int)
        confs = boxes_obj.conf.detach().cpu().numpy()
        for (x1, y1, x2, y2), cid, c in zip(xyxy, cls, confs):
            cid_i = int(cid)
            class_counts[cid_i] += 1
            class_conf_sum[cid_i] += float(c)
            boxes.append((int(x1), int(y1), int(x2), int(y2), float(c), cid_i))
    if draw and boxes:
        draw_yolo_boxes(img0, boxes, names_list)
    return img0, class_counts, class_conf_sum, boxes


def run_yolo_throttled(model, frame, img_size: int, conf: float, iou: float, device: str, names_list: list[str], cache: YoloCache, frame_idx: int, infer_every: int):
    if infer_every <= 1 or frame_idx % infer_every == 0:
        img0, cc, cs, boxes = run_yolo_v8(model, frame, img_size, conf, iou, device, names_list, draw=True)
        cache.boxes = boxes
        cache.class_counts = cc
        cache.class_conf_sum = cs
        return img0, cc, cs
    img0 = frame.copy()
    if cache.boxes:
        draw_yolo_boxes(img0, cache.boxes, names_list)
    return img0, dict(cache.class_counts), dict(cache.class_conf_sum)


def merge_peak_counts(peak: dict[str, int], by_name: dict[str, int]) -> None:
    for name, cnt in by_name.items():
        if cnt > peak.get(name, 0):
            peak[name] = cnt


def new_session() -> InspectionSession:
    return InspectionSession(session_id=datetime.now().strftime("%Y%m%d_%H%M%S_%f"))


def inspection_barcode_key(session: InspectionSession) -> str:
    serial = (session.serial_number or session.barcode_raw or "").strip()
    return serial if serial else session.session_id


def save_inspection_sqlite(db_path: Path, session: InspectionSession) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pk = inspection_barcode_key(session)
    dbm.save_inspection(
        db_path,
        barcode=pk,
        session_id=session.session_id,
        device_type=session.device_key or "",
        front_status=session.front_status,
        front_components=session.front_by_name,
        label_status=session.label_status,
        label_components=session.label_by_name_peak,
        barcode_status=session.barcode_status,
        overall=session.overall,
        fail_phase=session.fail_phase,
        fail_reasons=session.fail_reasons,
    )


def finalize_pass_fail(session: InspectionSession, profiles, strict: bool) -> None:
    reasons: list[str] = []
    profile = profiles.get(session.device_key) if session.device_key else None
    if not session.device_key or profile is None:
        reasons.append("cihaz_tipi_belirsiz")
    if strict and profile:
        front_src = filter_for_front_id(session.front_by_name, profiles)
        front_val = validate_section(front_src, profile.front_components, "front")
        session.front_ok = front_val.ok
        if not front_val.ok:
            reasons.extend(format_mismatches(front_val.mismatches))
        label_val = validate_section(session.label_by_name_peak, profile.label_components, "labels")
        session.label_ok = label_val.ok
        if not label_val.ok:
            reasons.extend(format_mismatches(label_val.mismatches))
    if not session.front_ok:
        reasons.append("on_yuz_tamamlanmadi")
    if not session.label_ok:
        reasons.append("etiket_kontrolu_eksik")
    if not session.barcode_ok or not session.serial_number:
        reasons.append("barkod_okunamadi")
    session.fail_reasons = reasons
    session.front_status = "PASS" if session.front_ok else "FAIL"
    session.label_status = "PASS" if session.label_ok else "FAIL"
    session.barcode_status = "PASS" if session.barcode_ok else "FAIL"
    session.overall = "PASS" if not reasons else "FAIL"


def abort_to_fail(session: InspectionSession, failed_phase: Phase, reasons: list[str]) -> None:
    session.overall = "FAIL"
    session.fail_phase = failed_phase.name
    session.fail_reasons = list(reasons)
    if failed_phase == Phase.FRONT:
        session.front_status = "FAIL"
        session.label_status = "SKIPPED"
        session.barcode_status = "SKIPPED"
    elif failed_phase == Phase.LABEL:
        session.front_status = "PASS" if session.front_ok else "FAIL"
        session.label_status = "FAIL"
        session.barcode_status = "SKIPPED"
    elif failed_phase == Phase.BARCODE:
        session.front_status = "PASS" if session.front_ok else "FAIL"
        session.label_status = "PASS" if session.label_ok else "FAIL"
        session.barcode_status = "FAIL"


def go_to_result(session: InspectionSession, profiles, strict: bool) -> None:
    if session.overall == "PENDING":
        finalize_pass_fail(session, profiles, strict)
    if not session.result_saved:
        save_inspection_sqlite(DB_PATH, session)
        session.result_saved = True


def draw_banner(img, text: str, color=(40, 40, 40)):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 58), (20, 20, 24), -1)
    cv2.rectangle(overlay, (0, 0), (w, 6), color, -1)
    cv2.rectangle(overlay, (0, 58), (w, 60), (210, 210, 210), -1)
    cv2.addWeighted(overlay, 0.92, img, 0.08, 0, img)
    cv2.putText(img, text, (16, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, (250, 250, 250), 2, cv2.LINE_AA)


def draw_round_rect(img, pt1, pt2, color, radius=14, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    radius = max(2, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y1 + thickness), color, -1)
        cv2.rectangle(img, (x1 + radius, y2 - thickness), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x1 + thickness, y2 - radius), color, -1)
        cv2.rectangle(img, (x2 - thickness, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_chip(panel, x, y, text, bg, fg=(255, 255, 255), pad_x=10, pad_y=6, font_scale=0.45, bold=False):
    font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
    w = tw + pad_x * 2
    h = th + baseline + pad_y * 2
    draw_round_rect(panel, (x, y), (x + w, y + h), bg, radius=14, thickness=-1)
    cv2.putText(panel, text, (x + pad_x, y + h - pad_y - baseline), font, font_scale, fg, 1, cv2.LINE_AA)
    return w, h


def draw_pass_fail_badge(panel, x, y, overall: str) -> tuple[int, int]:
    label = (overall or "?").upper()
    is_pass = label == "PASS"
    bg = COLOR_PASS_BG if is_pass else COLOR_FAIL_BG
    return draw_chip(panel, x, y, label, bg, (255, 255, 255), pad_x=10, pad_y=5, font_scale=0.44, bold=True)


def draw_metric_card(panel, x, y, w, h, title, value, subtitle=None, accent=(25, 84, 166)):
    draw_round_rect(panel, (x, y), (x + w, y + h), (255, 255, 255), radius=16, thickness=-1)
    draw_round_rect(panel, (x, y), (x + w, y + h), (228, 232, 238), radius=16, thickness=1)
    cv2.rectangle(panel, (x, y), (x + 6, y + h), accent, -1)
    cv2.putText(panel, title, (x + 16, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 96), 1, cv2.LINE_AA)
    cv2.putText(panel, value, (x + 16, y + 52), cv2.FONT_HERSHEY_DUPLEX, 0.62, (25, 25, 28), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(panel, subtitle, (x + 16, y + h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (110, 110, 116), 1, cv2.LINE_AA)


def _phase_ui_meta(phase_name: str) -> tuple[str, str, tuple[int, int, int]]:
    key = (phase_name or "").strip().upper()
    meta = {
        "FRONT": ("On yuz taramasi", "Komponent sayimi ve cihaz tipi", PANEL_ACCENT_2),
        "LABEL": ("Etiket kontrolu", "Garanti ve kalite etiketi", (120, 90, 200)),
        "BARCODE": ("Barkod okuma", "Seri numarasi kaydi", (90, 130, 200)),
        "RESULT": ("Denetim sonucu", "PASS / FAIL ozeti", PANEL_ACCENT),
    }
    return meta.get(key, ("Kalite kontrol", "Canli denetim", PANEL_ACCENT))


def build_panel(H: int, lines: list[str], result_color=None) -> np.ndarray:
    panel = np.empty((H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = PANEL_BG

    phase = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("Asama:")), "FRONT")
    ident = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("ID:")), "-")
    phase_title, phase_sub, phase_accent = _phase_ui_meta(phase)

    header_bottom = 76
    draw_round_rect(panel, (12, 10), (PANEL_W - 12, header_bottom), (255, 255, 255), radius=18, thickness=-1)
    cv2.rectangle(panel, (12, 10), (PANEL_W - 12, 16), PANEL_ACCENT, -1)
    cv2.putText(panel, "Kalite Kontrol", (22, 42), cv2.FONT_HERSHEY_DUPLEX, 0.82, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(panel, "Canli uretim hatti", (22, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_MUTED, 1, cv2.LINE_AA)

    chip_y = header_bottom + 12
    _, chip_h1 = draw_chip(panel, 16, chip_y, phase, phase_accent, bold=True)
    short_id = _truncate(ident, 16)
    _, chip_h2 = draw_chip(panel, 16, chip_y + chip_h1 + 8, f"ID {short_id}", (88, 92, 102), font_scale=0.40)
    y = chip_y + chip_h1 + chip_h2 + 16

    timer_line = next((ln for ln in lines if ln.startswith("Kalan:")), None)
    if timer_line:
        draw_round_rect(panel, (14, y), (PANEL_W - 14, y + 44), (255, 252, 240), radius=14, thickness=-1)
        cv2.rectangle(panel, (14, y), (18, y + 44), PANEL_WARN, -1)
        cv2.putText(panel, timer_line, (26, y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.58, (40, 90, 140), 2, cv2.LINE_AA)
        y += 52

    card_h = 68
    draw_round_rect(panel, (14, y), (PANEL_W - 14, y + card_h), (255, 255, 255), radius=16, thickness=-1)
    draw_round_rect(panel, (14, y), (PANEL_W - 14, y + card_h), (228, 232, 238), radius=16, thickness=1)
    cv2.rectangle(panel, (14, y), (20, y + card_h), phase_accent, -1)
    cv2.putText(panel, phase_title, (26, y + 24), cv2.FONT_HERSHEY_DUPLEX, 0.52, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(panel, phase_sub, (26, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_MUTED, 1, cv2.LINE_AA)
    y += card_h + 14

    skip_prefixes = ("Asama:", "ID:", "FPS:", "SONUC:", "SN:", "r =", "Hata asamasi:")
    content_lines = [ln for ln in lines if not any(ln.startswith(p) for p in skip_prefixes) and not ln.startswith("Kalan:")]
    content_top = y
    content_bottom = H - 58
    content_h = max(80, content_bottom - content_top)
    draw_round_rect(panel, (14, content_top), (PANEL_W - 14, content_top + content_h), (255, 255, 255), radius=16, thickness=-1)
    draw_round_rect(panel, (14, content_top), (PANEL_W - 14, content_top + content_h), (228, 232, 238), radius=16, thickness=1)

    cy = content_top + 22
    line_h = 21
    max_lines = max(1, (content_top + content_h - cy - 8) // line_h)
    used = 0
    for line in content_lines:
        if used >= max_lines:
            break
        if line.startswith("1) ") or line.startswith("2) "):
            cv2.putText(panel, line, (PAD_X, cy), cv2.FONT_HERSHEY_DUPLEX, 0.48, COLOR_TEXT, 1, cv2.LINE_AA)
        elif line.startswith("   "):
            cv2.putText(panel, line.strip(), (PAD_X + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_MUTED, 1, cv2.LINE_AA)
        elif line.startswith("Komponent:"):
            cv2.putText(panel, line, (PAD_X, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_MUTED, 1, cv2.LINE_AA)
        elif line.startswith("Cihaz:") or line.startswith("Tahmin:"):
            cv2.putText(panel, line, (PAD_X, cy), cv2.FONT_HERSHEY_DUPLEX, 0.46, (90, 50, 150), 1, cv2.LINE_AA)
        elif line.startswith(">>>"):
            cv2.putText(panel, line, (PAD_X, cy), cv2.FONT_HERSHEY_DUPLEX, 0.46, PANEL_OK, 2, cv2.LINE_AA)
        elif line.startswith("Sabitlik:"):
            cv2.putText(panel, line, (PAD_X, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PANEL_WARN, 1, cv2.LINE_AA)
        elif line.startswith("On yuz:") or line.startswith("Garanti+"):
            cv2.putText(panel, line, (PAD_X, cy), cv2.FONT_HERSHEY_DUPLEX, 0.44, PANEL_DANGER, 1, cv2.LINE_AA)
        elif line.startswith("Barkod") or line.startswith("Onceki"):
            cv2.putText(panel, line, (PAD_X, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_MUTED, 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, line, (PAD_X, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (62, 65, 72), 1, cv2.LINE_AA)
        cy += line_h
        used += 1

    footer_y = H - 50
    result_line = next((ln for ln in lines if ln.startswith("SONUC:")), None)
    sn_line = next((ln for ln in lines if ln.startswith("SN:")), None)
    if result_line or sn_line:
        draw_round_rect(panel, (14, footer_y - 8), (PANEL_W - 14, H - 12), (255, 255, 255), radius=14, thickness=-1)
        fy = footer_y + 6
        if result_line:
            col = result_color or (PANEL_OK if "PASS" in result_line else PANEL_DANGER)
            cv2.putText(panel, result_line, (PAD_X, fy), cv2.FONT_HERSHEY_DUPLEX, 0.58, col, 2, cv2.LINE_AA)
            fy += 22
        if sn_line:
            cv2.putText(panel, sn_line, (PAD_X, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 110, 95), 1, cv2.LINE_AA)

    cv2.putText(panel, "Space:onay  d:debug  r:yeni", (PAD_X, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_MUTED, 1, cv2.LINE_AA)
    return panel


def build_history_panel(H: int, db_path: Path, profiles: dict) -> np.ndarray:
    panel = np.empty((H, HISTORY_PANEL_W, 3), dtype=np.uint8)
    panel[:] = PANEL_BG
    draw_round_rect(panel, (10, 10), (HISTORY_PANEL_W - 10, 70), (255, 255, 255), radius=16, thickness=-1)
    cv2.rectangle(panel, (10, 10), (HISTORY_PANEL_W - 10, 16), PANEL_ACCENT_2, -1)
    cv2.putText(panel, "Sonuclar", (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(panel, "Son kayitlar", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_MUTED, 1, cv2.LINE_AA)
    try:
        rows = dbm.list_inspections(db_path, limit=HISTORY_ROW_LIMIT)
    except Exception:
        rows = []
    y = 82
    row_h = 68
    if not rows:
        draw_round_rect(panel, (12, y), (HISTORY_PANEL_W - 12, y + 48), (255, 255, 255), radius=12, thickness=-1)
        cv2.putText(panel, "Henuz kayit yok", (22, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_MUTED, 1, cv2.LINE_AA)
        return panel
    for row in rows:
        if y + row_h > H - 12:
            break
        card_y1, card_y2 = y, y + row_h - 8
        overall = (row["overall"] or "").upper()
        is_pass = overall == "PASS"
        accent = COLOR_PASS_BG if is_pass else COLOR_FAIL_BG
        draw_round_rect(panel, (12, card_y1), (HISTORY_PANEL_W - 12, card_y2), (255, 255, 255), radius=12, thickness=-1)
        draw_round_rect(panel, (12, card_y1), (HISTORY_PANEL_W - 12, card_y2), (228, 232, 238), radius=12, thickness=1)
        cv2.rectangle(panel, (12, card_y1), (16, card_y2), accent, -1)
        dev_name = _truncate(_device_display_name(row["device_type"], profiles), 20)
        barcode = _truncate(row["barcode"], 18)
        cv2.putText(panel, dev_name, (22, card_y1 + 22), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(panel, barcode, (22, card_y1 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_MUTED, 1, cv2.LINE_AA)
        draw_pass_fail_badge(panel, HISTORY_PANEL_W - 12 - 72, card_y1 + 12, overall)
        if not is_pass and row.get("fail_phase"):
            phase_tr = PHASE_LABEL_TR.get(row["fail_phase"], row["fail_phase"])
            cv2.putText(panel, f"Asama: {phase_tr}", (22, card_y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, PANEL_DANGER, 1, cv2.LINE_AA)
        y += row_h
    return panel


def _letterbox_to_screen(img: np.ndarray, screen_w: int, screen_h: int, bg_color=(0, 0, 0)) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return img
    scale = min(screen_w / w, screen_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.full((screen_h, screen_w, 3), bg_color, dtype=np.uint8)
    x0 = (screen_w - new_w) // 2
    y0 = (screen_h - new_h) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def _get_screen_size_win32() -> tuple[int, int] | None:
    try:
        import ctypes

        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return None

# NOTE: build_history_panel is defined above with the full UI.


def _truncate(text: str, max_len: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _device_display_name(device_key: str, profiles: dict) -> str:
    if not device_key:
        return "-"
    profile = profiles.get(device_key)
    if profile is not None:
        return getattr(profile, "display_name", None) or str(device_key)
    return device_key


def parse_opt():
    p = argparse.ArgumentParser(description="Aşamalı kalite kontrolü (YOLOv8) - Result3")
    default_weights = r"C:\Users\sentu\OneDrive\Desktop\MP110DetectionYoloV5_5\best1.pt"
    p.add_argument("--weights", type=str, default=str(default_weights))
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--device", default="", help="0 / cpu gibi")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--conf-front", type=float, default=0.85)
    p.add_argument("--conf-label", type=float, default=0.85)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--no-dshow", action="store_true")
    p.add_argument("--strict", action="store_true", default=True)
    p.add_argument("--no-strict", action="store_false", dest="strict")
    p.add_argument("--profiles", type=str, default="")
    p.add_argument("--label-debug", action="store_true")
    p.add_argument("--infer-every", type=int, default=2)
    p.add_argument("--show-fps", action="store_true")
    return p.parse_args()


def main():
    opt = parse_opt()
    weights_path = Path(opt.weights).expanduser().resolve()
    if not weights_path.is_file():
        print(f"Ağırlık dosyası bulunamadı: {weights_path}")
        sys.exit(1)
    if not HAS_ULTRALYTICS:
        print("Hata: ultralytics yüklü değil. pip install ultralytics")
        sys.exit(1)
    if not HAS_PYZBAR:
        print("Uyarı: pyzbar yok — barkod aşaması çalışmaz. pip install pyzbar (+ ZBar)")

    profiles_path = Path(opt.profiles) if opt.profiles else None
    profiles = load_profiles(profiles_path)
    try:
        dbm.init_db(DB_PATH)
        dbm.load_profiles_into_db(DB_PATH, profiles)
    except Exception:
        pass

    label_debug = bool(opt.label_debug)
    infer_every = max(1, int(opt.infer_every))

    t0 = time.perf_counter()
    device_str = opt.device.strip() if opt.device else ("0" if torch.cuda.is_available() else "cpu")
    print("Model yukleniyor (YOLOv8)...")
    model = YOLO(str(weights_path))
    names_list = _normalize_names(model.names)
    img_size = int(opt.img_size)
    print(f"Hazir ({time.perf_counter() - t0:.1f}s) | device={device_str} | img={img_size} | infer_every={infer_every}")

    src = opt.source
    try:
        cam_id = int(src)
        cap = cv2.VideoCapture(cam_id) if opt.no_dshow else cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    except ValueError:
        cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"Kaynak açılamadı: {opt.source}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    win_name = "Kalite Kontrol - Result3 (YOLOv8)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    screen_size = _get_screen_size_win32() if sys.platform == "win32" else None
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass
    if screen_size:
        try:
            sw, sh = screen_size
            cv2.resizeWindow(win_name, sw, sh)
        except Exception:
            pass

    phase = Phase.FRONT
    session = new_session()
    stable_count = 0
    last_stable_signature = ""
    yolo_cache = YoloCache()
    frame_idx = 0
    fps_ema = 0.0
    loop_t0 = time.perf_counter()
    result_start_time = 0.0
    phase_started_at = time.perf_counter()
    # Hızlı otomatik geçiş: 2 ardışık karede tam eşleşme
    AUTO_STREAK = 2
    front_ok_streak = 0
    label_ok_streak = 0

    print("Space=onay (OK ise) | 1=DM100 2=XIO110 | r=yeni urun | q=cikis | d=debug")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        dt = time.perf_counter() - loop_t0
        loop_t0 = time.perf_counter()
        if dt > 0:
            fps_ema = fps_ema * 0.9 + (1.0 / dt) * 0.1

        panel_lines: list[str] = [f"Asama: {phase.name}", f"ID: {session.session_id[-12:]}"]
        if opt.show_fps:
            panel_lines.append(f"FPS: {fps_ema:.0f}")

        # Timeout
        timed_out = False
        if phase in (Phase.FRONT, Phase.LABEL, Phase.BARCODE):
            limit = PHASE_TIMEOUT_SEC[phase]
            elapsed = time.perf_counter() - phase_started_at
            remaining = max(0.0, limit - elapsed)
            panel_lines.append(f"Kalan: {int(remaining)} sn")
            if elapsed >= limit:
                abort_to_fail(session, phase, [f"{phase.name.lower()}_timeout"])
                save_inspection_sqlite(DB_PATH, session)
                phase = Phase.RESULT
                result_start_time = time.perf_counter()
                timed_out = True

        if timed_out or phase == Phase.RESULT:
            img0 = frame.copy()
            if not session.result_saved:
                go_to_result(session, profiles, opt.strict)
            draw_banner(img0, session.overall, COLOR_PASS_BG if session.overall == "PASS" else COLOR_FAIL_BG)
            panel_lines.append(f"SONUC: {session.overall}")
            panel_lines.append(f"Cihaz: {session.device_display or session.device_key or '-'}")
            panel_lines.append(f"SN: {session.serial_number or '-'}")
            if session.fail_phase:
                panel_lines.append(f"Hata asamasi: {session.fail_phase}")
            for r in session.fail_reasons[:6]:
                panel_lines.append(r[:30])
            panel_lines.append("r = yeni urun")

        elif phase == Phase.FRONT:
            img0, class_counts, _ = run_yolo_throttled(
                model,
                frame,
                img_size,
                float(opt.conf_front),
                float(opt.iou),
                device_str,
                names_list,
                yolo_cache,
                frame_idx,
                infer_every,
            )
            by_name = build_name_to_count(names_list, class_counts)
            session.front_by_name = dict(by_name)

            # cihaz tipi: komponent sayımı tabanlı
            guess = identify_device_from_components(by_name, profiles)
            guess_key, profile = guess.key, guess.profile

            panel_lines.append("1) Komponent sayimi")
            panel_lines.extend(format_raw_front_lines(by_name, profiles)[:7])

            if session.device_key and session.device_key in profiles and session.device_identified:
                profile = profiles[session.device_key]
                guess_key = session.device_key
                panel_lines.append(f"2) Cihaz: {profile.display_name} (manuel)")
            elif guess_key and profile:
                panel_lines.append(f"2) Tahmin: {profile.display_name}")
                panel_lines.append(f"   skor {guess.score:.0%}")
            else:
                panel_lines.append("2) Cihaz: henuz belirlenmedi")
                panel_lines.append("   (sayimlar oturunca)")
                panel_lines.append("1=DM100 2=XIO110")

            if guess_key and profile:
                front_val = validate_section(
                    filter_for_front_id(by_name, profiles),
                    profile.front_components,
                    "front",
                )
                sig = f"{guess_key}|" + "|".join(format_mismatches(front_val.mismatches))
                panel_lines.extend(format_counts_line(by_name, profile, "front")[:6])

                if front_val.ok:
                    stable_count = stable_count + 1 if sig == last_stable_signature else 1
                    last_stable_signature = sig
                    if stable_count >= STABLE_FRAMES_FRONT:
                        session.device_key = guess_key
                        session.device_display = profile.display_name
                        session.device_identified = True
                        session.front_ok = True if not opt.strict else True
                    front_ok_streak += 1
                    if front_ok_streak >= AUTO_STREAK:
                        session.device_key = guess_key
                        session.device_display = profile.display_name
                        session.device_identified = True
                        session.front_ok = True if not opt.strict else True
                        session.front_status = "PASS"
                        phase = Phase.LABEL
                        phase_started_at = time.perf_counter()
                        stable_count = 0
                        last_stable_signature = ""
                        front_ok_streak = 0
                        label_ok_streak = 0
                        continue
                else:
                    stable_count = 0
                    last_stable_signature = sig
                    front_ok_streak = 0

                if opt.strict and not front_val.ok:
                    panel_lines.append("On yuz: sayim hatali")
                elif session.device_identified and front_val.ok:
                    panel_lines.append(">>> Space ile devam")
                elif stable_count > 0 and front_val.ok:
                    panel_lines.append(f"Sabitlik: {stable_count}/{STABLE_FRAMES_FRONT}")

            draw_banner(img0, PHASE_TITLES[phase], PANEL_ACCENT_2)

        elif phase == Phase.LABEL:
            profile = profiles.get(session.device_key) if session.device_key else None
            img0, class_counts, _ = run_yolo_throttled(
                model,
                frame,
                img_size,
                float(opt.conf_label),
                float(opt.iou),
                device_str,
                names_list,
                yolo_cache,
                frame_idx,
                infer_every,
            )
            by_name = build_name_to_count(names_list, class_counts)
            merge_peak_counts(session.label_by_name_peak, by_name)

            if profile:
                label_val = validate_section(session.label_by_name_peak, profile.label_components, "labels")
                label_ok_now = label_val.ok if opt.strict else True
                panel_lines.append(f"Cihaz: {profile.display_name}")
                panel_lines.extend(format_counts_line(session.label_by_name_peak, profile, "labels"))
                if label_ok_now:
                    stable_count += 1
                    panel_lines.append(">>> Space ile devam")
                    label_ok_streak += 1
                    if label_ok_streak >= AUTO_STREAK:
                        session.label_ok = True
                        session.label_status = "PASS"
                        phase = Phase.BARCODE
                        phase_started_at = time.perf_counter()
                        stable_count = 0
                        label_ok_streak = 0
                        front_ok_streak = 0
                        continue
                else:
                    stable_count = 0
                    panel_lines.append("Garanti+Kalite: eksik")
                    label_ok_streak = 0
            else:
                panel_lines.append("Profil yok")
                label_ok_streak = 0
            draw_banner(img0, PHASE_TITLES[phase], (160, 100, 60))

        elif phase == Phase.BARCODE:
            img0 = frame.copy()
            serial = ""
            if HAS_PYZBAR:
                barcodes = pyzbar.decode(frame)
                for obj in barcodes:
                    data = obj.data.decode("utf-8", errors="ignore")
                    draw_barcode(img0, obj, data)
                    if not serial:
                        serial = data.strip()
                brect = union_barcode_rect(barcodes) if barcodes else None
                if brect is not None:
                    cand, _ = detect_label_candidate(frame, brect=brect, debug=label_debug)
                    if cand is not None:
                        session.label_cv_box_seen = True
                        x, y, w, h = cand
                        cv2.rectangle(img0, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(img0, "Etiket", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2, cv2.LINE_AA)
            if serial:
                session.serial_number = serial
                session.barcode_raw = serial
                session.barcode_ok = True
                panel_lines.append(f"SN: {serial[:24]}")
                if session.front_ok and session.label_ok:
                    go_to_result(session, profiles, opt.strict)
                    phase = Phase.RESULT
                    result_start_time = time.perf_counter()
                else:
                    panel_lines.append("Onceki adimlar OK degil")
            else:
                panel_lines.append("Barkod bekleniyor...")
            draw_banner(img0, PHASE_TITLES[phase], PANEL_ACCENT)

        H = img0.shape[0]
        result_col = None
        if phase == Phase.RESULT:
            result_col = PANEL_OK if session.overall == "PASS" else PANEL_DANGER
        panel = build_panel(H, panel_lines, result_col)
        history_panel = build_history_panel(H, DB_PATH, profiles)
        combined = np.concatenate([panel, img0, history_panel], axis=1)
        if screen_size:
            sw, sh = screen_size
            combined = _letterbox_to_screen(combined, sw, sh, bg_color=(0, 0, 0))
        cv2.imshow(win_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            label_debug = not label_debug
        if key == ord("r"):
            session = new_session()
            phase = Phase.FRONT
            phase_started_at = time.perf_counter()
            stable_count = 0
            last_stable_signature = ""
            yolo_cache = YoloCache()
            result_start_time = 0.0
            continue
        if key == ord("1") and "DM100" in profiles:
            session.device_key = "DM100"
            session.device_display = profiles["DM100"].display_name
            session.device_identified = True
        if key == ord("2") and "XIO110" in profiles:
            session.device_key = "XIO110"
            session.device_display = profiles["XIO110"].display_name
            session.device_identified = True
        if key == ord(" "):
            if phase == Phase.FRONT:
                if not (session.device_identified and session.device_key and session.device_key in profiles):
                    print("Cihaz tipi belirlenmedi — sayim otursun veya 1/2 ile secin.")
                else:
                    profile = profiles[session.device_key]
                    if opt.strict:
                        front_val = validate_section(filter_for_front_id(session.front_by_name, profiles), profile.front_components, "front")
                        session.front_ok = front_val.ok
                    else:
                        session.front_ok = True
                    if session.front_ok:
                        session.front_status = "PASS"
                        phase = Phase.LABEL
                        phase_started_at = time.perf_counter()
                        stable_count = 0
            elif phase == Phase.LABEL:
                profile = profiles.get(session.device_key) if session.device_key else None
                if profile:
                    label_val = validate_section(session.label_by_name_peak, profile.label_components, "labels")
                    session.label_ok = label_val.ok if opt.strict else True
                if session.label_ok:
                    session.label_status = "PASS"
                    phase = Phase.BARCODE
                    phase_started_at = time.perf_counter()
                    stable_count = 0
            elif phase == Phase.BARCODE:
                if session.barcode_ok and session.front_ok and session.label_ok:
                    go_to_result(session, profiles, opt.strict)
                    phase = Phase.RESULT
                    result_start_time = time.perf_counter()

        if phase == Phase.RESULT and session.result_saved and result_start_time > 0.0:
            if time.perf_counter() - result_start_time >= RESULT_DISPLAY_SECONDS:
                session = new_session()
                phase = Phase.FRONT
                phase_started_at = time.perf_counter()
                stable_count = 0
                last_stable_signature = ""
                yolo_cache = YoloCache()
                result_start_time = 0.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

