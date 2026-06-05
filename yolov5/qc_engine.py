"""
Aşamalı endüstriyel kalite kontrolü: ön yüz → etiket → barkod → PASS/FAIL + SQLite.

  python qc_engine.py
  python qc_engine.py --weights ../best.pt --strict
  python qc_engine.py --no-strict          # sayım doğrulaması gevşek
  python qc_engine.py --show-fps
  python qc_engine.py --infer-every 1   # her kare YOLO (daha yavas, daha guncel)
  python qc_engine.py --img-size 416    # daha hizli YOLO

Tuşlar: Space=adım onayla, 1=DM100 2=XIO110 | r=yeni urun | q=cikis | d=debug
"""

from __future__ import annotations

import argparse
import ctypes
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
from utils.general import non_max_suppression, scale_boxes  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402
from utils import db as dbm  # noqa: E402

# --- Etiket adayı (OCR2 ile uyumlu) ---
MIN_AREA_FRAC = 0.005
MAX_AREA_FRAC = 0.70
MIN_RECT_SCORE = 0.58
ASPECT_MIN_MAX = (0.8, 10.0)
BORDER_MARGIN = 6
WHITE_S_MAX = 85
WHITE_V_MIN = 165
BARCODE_MARGIN = 6

RESULTS_DIR = YOLOV5_ROOT / "results"
DB_PATH = RESULTS_DIR / "devices.db"

PANEL_W = 360
HISTORY_PANEL_W = 300
HISTORY_ROW_LIMIT = 10
PHASE_LABEL_TR = {"FRONT": "On yuz", "LABEL": "Etiket", "BARCODE": "Barkod", "": ""}
PANEL_BG = (242, 244, 248)  # light neutral background
PANEL_ACCENT = (25, 84, 166)
PANEL_ACCENT_2 = (14, 132, 115)
PANEL_WARN = (0, 140, 220)
PANEL_OK = (40, 160, 80)
PANEL_DANGER = (50, 50, 220)
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
    unit_number: int = 0
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
    pending_fail_phase: str = ""
    pending_fail_reasons: list[str] = field(default_factory=list)
    result_saved: bool = False


@dataclass
class YoloCache:
    """Son inference sonucu — ara karelerde sadece çizim için kullanılır."""

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


def draw_yolo_boxes(img0, boxes: list[tuple[int, int, int, int, float, int]], names_list: list[str]) -> None:
    for x1, y1, x2, y2, conf, cid in boxes:
        cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img0,
            f"{names_list[cid]} {conf:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 0),
            1,
            cv2.LINE_AA,
        )


def run_yolo(
    model,
    frame,
    device,
    img_size: int,
    conf_thres: float,
    iou_thres: float,
    names_list: list[str],
    draw: bool = True,
):
    """Tek kare tam YOLO inference (ağır işlem)."""
    img0 = frame.copy()
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
    boxes: list[tuple[int, int, int, int, float, int]] = []

    if pred[0] is not None and len(pred[0]):
        pred[0][:, :4] = scale_boxes(img.shape[1:], pred[0][:, :4], img0.shape).round()
        for *xyxy, conf, cls in pred[0]:
            cid = int(cls)
            class_counts[cid] += 1
            class_conf_sum[cid] += float(conf)
            x1, y1, x2, y2 = map(int, xyxy)
            boxes.append((x1, y1, x2, y2, float(conf), cid))

    if draw and boxes:
        draw_yolo_boxes(img0, boxes, names_list)

    return img0, class_counts, class_conf_sum, boxes


def run_yolo_throttled(
    model,
    frame,
    device,
    img_size: int,
    conf_thres: float,
    iou_thres: float,
    names_list: list[str],
    cache: YoloCache,
    frame_idx: int,
    infer_every: int,
):
    """infer_every>1 ise ara karelerde sadece önbellek çizilir (FPS artar)."""
    if infer_every <= 1 or frame_idx % infer_every == 0:
        img0, cc, cs, boxes = run_yolo(
            model, frame, device, img_size, conf_thres, iou_thres, names_list, draw=True
        )
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


def new_session(*, unit_number: int = 1) -> InspectionSession:
    return InspectionSession(
        session_id=datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        unit_number=max(1, int(unit_number)),
    )


def format_session_display(session: InspectionSession) -> str:
    """Operator-facing label: serial when scanned, else unit counter (#001)."""
    serial = (session.serial_number or "").strip()
    if serial:
        return serial if len(serial) <= 24 else serial[:21] + "..."
    if session.unit_number > 0:
        return f"#{session.unit_number:03d}"
    return "—"


def finalize_pass_fail(session: InspectionSession, profiles, strict: bool) -> None:
    """Sonuç ekranına geçmeden önce bayrakları ve aşama durumlarını netleştirir."""
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
    elif not session.front_ok:
        reasons.append("on_yuz_tamamlanmadi")
    if not session.label_ok:
        reasons.append("etiket_kontrolu_eksik")
    if not session.barcode_ok or not session.serial_number:
        reasons.append("barkod_okunamadi")

    session.fail_reasons = reasons
    session.front_status = "PASS" if session.front_ok else "FAIL"
    session.label_status = "PASS" if session.label_ok else ("SKIPPED" if not session.front_ok else "FAIL")
    if session.barcode_ok:
        session.barcode_status = "PASS"
    elif session.fail_phase in ("FRONT", "LABEL") and session.pending_fail_phase:
        session.barcode_status = "PENDING"
    else:
        session.barcode_status = "SKIPPED" if session.fail_phase in ("FRONT", "LABEL") else "FAIL"
    session.overall = "PASS" if not reasons else "FAIL"


def defer_fail_to_barcode(
    session: InspectionSession,
    failed_phase: Phase,
    reasons: list[str],
) -> None:
    """On yuz/etiket FAIL: once barkod okutulur, sonra kayit (cihaz SN ile)."""
    session.pending_fail_phase = failed_phase.name
    session.pending_fail_reasons = list(reasons) if reasons else [f"{failed_phase.name.lower()}_fail"]
    session.overall = "PENDING"
    session.result_saved = False

    if failed_phase == Phase.FRONT:
        session.front_ok = False
        session.front_status = "FAIL"
        session.label_status = "SKIPPED"
        session.label_ok = False
    elif failed_phase == Phase.LABEL:
        session.label_ok = False
        session.label_status = "FAIL"
        if not session.front_ok:
            session.front_status = "FAIL"


def complete_fail_after_barcode(
    session: InspectionSession,
    profiles,
    strict: bool,
    db_path: Path,
    *,
    extra_reasons: list[str] | None = None,
) -> None:
    """Barkod okunduktan (veya barkod asamasi bittikten) sonra FAIL kaydi."""
    reasons = list(session.pending_fail_reasons)
    if extra_reasons:
        reasons.extend(extra_reasons)

    fail_phase = session.pending_fail_phase
    if not fail_phase:
        if not session.front_ok:
            fail_phase = "FRONT"
        elif not session.label_ok:
            fail_phase = "LABEL"
        else:
            fail_phase = "BARCODE"

    session.fail_phase = fail_phase
    session.fail_reasons = reasons

    if not session.front_ok:
        session.front_status = "FAIL"
    elif session.front_status == "PENDING":
        session.front_status = "PASS"

    if session.label_status == "SKIPPED":
        pass
    elif not session.label_ok:
        session.label_status = "FAIL"
    elif session.label_status == "PENDING":
        session.label_status = "PASS"

    if session.barcode_ok and session.serial_number:
        session.barcode_status = "PASS"
    else:
        session.barcode_status = "FAIL"
        if "barkod_okunamadi" not in session.fail_reasons:
            session.fail_reasons.append("barkod_okunamadi")

    if strict and profiles and session.device_key:
        profile = profiles.get(session.device_key)
        if profile:
            if fail_phase == "FRONT" and session.front_by_name:
                front_val = validate_section(
                    filter_for_front_id(session.front_by_name, profiles),
                    profile.front_components,
                    "front",
                )
                if not front_val.ok:
                    for m in format_mismatches(front_val.mismatches):
                        if m not in session.fail_reasons:
                            session.fail_reasons.append(m)
            if fail_phase == "LABEL" and session.label_by_name_peak:
                label_val = validate_section(
                    session.label_by_name_peak, profile.label_components, "labels"
                )
                if not label_val.ok:
                    for m in format_mismatches(label_val.mismatches):
                        if m not in session.fail_reasons:
                            session.fail_reasons.append(m)

    session.overall = "FAIL"
    session.pending_fail_phase = ""
    session.pending_fail_reasons = []
    save_inspection_sqlite(db_path, session)
    session.result_saved = True


def inspection_barcode_key(session: InspectionSession) -> str:
    serial = (session.serial_number or session.barcode_raw or "").strip()
    return serial if serial else session.session_id


def save_inspection_sqlite(db_path: Path, session: InspectionSession) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pk = inspection_barcode_key(session)
    try:
        dbm.save_inspection(
            db_path,
            barcode=pk,
            session_id=session.session_id,
            device_type=session.device_key or "belirlenemedi",
            front_status=session.front_status,
            front_components=session.front_by_name,
            label_status=session.label_status,
            label_components=session.label_by_name_peak,
            barcode_status=session.barcode_status,
            overall=session.overall,
            fail_phase=session.fail_phase,
            fail_reasons=session.fail_reasons,
        )
        print(f"SQLite kayit: {pk} -> {session.overall}")
    except Exception as exc:
        print(f"SQLite kayit hatasi: {exc}")


def abort_to_fail(
    session: InspectionSession,
    failed_phase: Phase,
    reasons: list[str],
    db_path: Path,
) -> None:
    session.overall = "FAIL"
    if session.pending_fail_phase and failed_phase == Phase.BARCODE:
        session.fail_phase = session.pending_fail_phase
        session.fail_reasons = list(session.pending_fail_reasons) + list(reasons)
    else:
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
        session.label_status = (
            "SKIPPED" if session.pending_fail_phase == "FRONT" else ("PASS" if session.label_ok else "FAIL")
        )
        session.barcode_status = "FAIL"

    session.pending_fail_phase = ""
    session.pending_fail_reasons = []
    save_inspection_sqlite(db_path, session)
    session.result_saved = True


def go_to_result(
    session: InspectionSession,
    profiles,
    strict: bool,
    db_path: Path,
) -> None:
    if session.overall == "PENDING":
        finalize_pass_fail(session, profiles, strict)
    if not session.result_saved:
        save_inspection_sqlite(db_path, session)
        session.result_saved = True


def draw_banner(img, text: str, color=(40, 40, 40)):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 58), (20, 20, 24), -1)
    cv2.rectangle(overlay, (0, 0), (w, 6), color, -1)
    cv2.rectangle(overlay, (0, 58), (w, 60), (210, 210, 210), -1)
    cv2.addWeighted(overlay, 0.92, img, 0.08, 0, img)
    cv2.putText(
        img,
        text,
        (16, 38),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (250, 250, 250),
        2,
        cv2.LINE_AA,
    )


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
    """PASS/FAIL rozeti — BGR kirmizi/yesil."""
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

    # --- Ust baslik ---
    header_bottom = 76
    draw_round_rect(panel, (12, 10), (PANEL_W - 12, header_bottom), (255, 255, 255), radius=18, thickness=-1)
    cv2.rectangle(panel, (12, 10), (PANEL_W - 12, 16), PANEL_ACCENT, -1)
    cv2.putText(panel, "Kalite Kontrol", (22, 42), cv2.FONT_HERSHEY_DUPLEX, 0.82, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(panel, "Canli uretim hatti", (22, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_MUTED, 1, cv2.LINE_AA)

    # Rozetler: alt alta (ust uste binmesin)
    chip_y = header_bottom + 12
    _, chip_h1 = draw_chip(panel, 16, chip_y, phase, phase_accent, bold=True)
    short_id = _truncate(ident, 16)
    _, chip_h2 = draw_chip(
        panel, 16, chip_y + chip_h1 + 8, f"ID {short_id}", (88, 92, 102), font_scale=0.40
    )
    y = chip_y + chip_h1 + chip_h2 + 16

    # --- Zamanlayici (varsa) ---
    timer_line = next((ln for ln in lines if ln.startswith("Kalan:")), None)
    if timer_line:
        draw_round_rect(panel, (14, y), (PANEL_W - 14, y + 44), (255, 252, 240), radius=14, thickness=-1)
        cv2.rectangle(panel, (14, y), (18, y + 44), PANEL_WARN, -1)
        cv2.putText(panel, timer_line, (26, y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.58, (40, 90, 140), 2, cv2.LINE_AA)
        y += 52

    # --- Asama ozeti karti (FRONT tekrar yazilmaz; yeterli yukseklik) ---
    card_h = 68
    draw_round_rect(panel, (14, y), (PANEL_W - 14, y + card_h), (255, 255, 255), radius=16, thickness=-1)
    draw_round_rect(panel, (14, y), (PANEL_W - 14, y + card_h), (228, 232, 238), radius=16, thickness=1)
    cv2.rectangle(panel, (14, y), (20, y + card_h), phase_accent, -1)
    cv2.putText(panel, phase_title, (26, y + 24), cv2.FONT_HERSHEY_DUPLEX, 0.52, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(panel, phase_sub, (26, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_MUTED, 1, cv2.LINE_AA)
    y += card_h + 14

    # --- Icerik karti ---
    skip_prefixes = ("Asama:", "ID:", "FPS:", "SONUC:", "SN:", "r =", "Hata asamasi:")
    content_lines = [
        ln
        for ln in lines
        if not any(ln.startswith(p) for p in skip_prefixes)
        and not ln.startswith("Kalan:")
    ]

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

    # --- Alt: sonuc / ipuclari ---
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

    cv2.putText(
        panel,
        "Space:onay  d:debug  r:yeni",
        (PAD_X, H - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_MUTED,
        1,
        cv2.LINE_AA,
    )

    return panel


def _truncate(text: str, max_len: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _device_display_name(device_key: str, profiles: dict) -> str:
    if not device_key:
        return "-"
    profile = profiles.get(device_key)
    if profile is not None:
        return getattr(profile, "display_name", None) or str(device_key)
    return device_key


def build_history_panel(H: int, db_path: Path, profiles: dict) -> np.ndarray:
    """Sag panel: SQLite inspections sonuclari."""
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
            cv2.putText(
                panel,
                f"Asama: {phase_tr}",
                (22, card_y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                PANEL_DANGER,
                1,
                cv2.LINE_AA,
            )

        y += row_h

    return panel


def parse_opt(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Aşamalı kalite kontrolü (qc_engine)")
    default_weights = YOLOV5_ROOT.parent / "best.pt"
    p.add_argument("--weights", type=str, default=str(default_weights))
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--device", default="")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--conf-front", type=float, default=0.85, help="ön yüz YOLO conf")
    p.add_argument("--conf-label", type=float, default=0.85, help="etiket yüzeyi YOLO conf")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--no-dshow", action="store_true")
    p.add_argument("--strict", action="store_true", default=True, help="referans sayım doğrulama")
    p.add_argument("--no-strict", action="store_false", dest="strict")
    p.add_argument("--profiles", type=str, default="", help="device_profiles.yaml yolu")
    p.add_argument("--label-debug", action="store_true")
    p.add_argument(
        "--infer-every",
        type=int,
        default=2,
        help="YOLO her N karede bir (1=her kare, 2=~2x FPS). Barkod aşaması etkilenmez.",
    )
    p.add_argument("--show-fps", action="store_true", help="panelde FPS göster")
    p.add_argument("--windowed", action="store_true", help="fullscreen yapma (launcher icin)")
    p.add_argument("--headless", action="store_true", help="imshow/waitKey kullanma (Qt UI icin)")
    return p.parse_args(argv)


def default_opt(**overrides):
    """Qt/embedded usage: returns argparse-like Namespace with qc_engine defaults."""
    opt = parse_opt([])
    for k, v in (overrides or {}).items():
        setattr(opt, k, v)
    return opt


def _get_screen_size_win32() -> tuple[int, int] | None:
    """Win32 üzerinde ekran boyutunu döndürür (piksel)."""
    try:
        user32 = ctypes.windll.user32
        # Windows'ta DPI ölçeklemeden gerçek piksel boyutlarını okumaya yardımcı olur.
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
        w = int(user32.GetSystemMetrics(0))  # SM_CXSCREEN
        h = int(user32.GetSystemMetrics(1))  # SM_CYSCREEN
        return (w, h)
    except Exception:
        return None


def _letterbox_to_screen(
    img: np.ndarray, screen_w: int, screen_h: int, bg_color=(0, 0, 0)
) -> np.ndarray:
    """Görüntüyü ekrana bozmadan sığdırır ve ortalar (letterbox)."""
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


def _scale_to_fit_display(
    img: np.ndarray, max_w: int, max_h: int
) -> tuple[np.ndarray, int, int]:
    """Siyah bant eklemeden görüntüyü ekrana sığdır (windowed mod)."""
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return img, w, h
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    if new_w == w and new_h == h:
        return img, w, h
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return out, new_w, new_h


def _layout_opencv_window(win_name: str, win_w: int, win_h: int, screen_size: tuple[int, int] | None) -> None:
    """Pencereyi içeriğe göre boyutlandır ve ortala."""
    try:
        cv2.resizeWindow(win_name, int(win_w), int(win_h))
        if screen_size:
            sw, sh = screen_size
            x = max(0, (sw - win_w) // 2)
            y = max(0, (sh - win_h) // 2)
            cv2.moveWindow(win_name, int(x), int(y))
    except Exception:
        pass


class OpenCVFramePresenter:
    """OpenCV penceresi: gösterim ölçekleme (siyah bant yok — windowed)."""

    WIN_NAME = "Kalite Kontrol"

    def __init__(self, *, windowed: bool, headless: bool = False) -> None:
        self.windowed = windowed
        self.headless = headless
        self.screen_size = _get_screen_size_win32() if sys.platform == "win32" else None
        self._layout_done = False

    def setup(self) -> None:
        if self.headless:
            return
        cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
        if not self.windowed and self.screen_size:
            try:
                cv2.setWindowProperty(self.WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except Exception:
                pass
            sw, sh = self.screen_size
            try:
                cv2.resizeWindow(self.WIN_NAME, sw, sh)
            except Exception:
                pass

    def present(self, combined: np.ndarray) -> None:
        if self.headless:
            return
        display = combined
        if self.screen_size:
            sw, sh = self.screen_size
            if self.windowed:
                display, win_w, win_h = _scale_to_fit_display(
                    combined, int(sw * 0.92), int(sh * 0.90)
                )
                if not self._layout_done:
                    _layout_opencv_window(self.WIN_NAME, win_w, win_h, self.screen_size)
                    self._layout_done = True
            else:
                display = _letterbox_to_screen(combined, sw, sh, bg_color=PANEL_BG)
        cv2.imshow(self.WIN_NAME, display)

    def destroy(self) -> None:
        if not self.headless:
            cv2.destroyAllWindows()


class QualityControlApp:
    """Aşamalı kalite kontrolü ana uygulaması (OpenCV döngüsü)."""

    def __init__(self, opt) -> None:
        self.opt = opt
        self.label_debug = bool(opt.label_debug)
        self.infer_every = max(1, int(opt.infer_every))

        self.profiles_path = Path(opt.profiles) if opt.profiles else None
        self.profiles: dict = {}
        self.weights_path: Path | None = None
        self.device = None
        self.model = None
        self.names_list: list[str] = []
        self.img_size = int(opt.img_size)
        self.cap: cv2.VideoCapture | None = None
        self.presenter = OpenCVFramePresenter(windowed=bool(opt.windowed), headless=bool(opt.headless))

        self.phase = Phase.FRONT
        self._unit_counter = 0
        self.session = self._create_new_session()
        self.stable_count = 0
        self.last_stable_signature = ""
        self.yolo_cache = YoloCache()
        self.frame_idx = 0
        self.fps_ema = 0.0
        self.loop_t0 = time.perf_counter()
        self.result_start_time = 0.0
        self.phase_started_at = time.perf_counter()
        self._last_class_counts: dict[int, int] = {}

    def run(self) -> None:
        self.initialize(enable_presenter=True)
        print("Space=onay (OK ise) | 1=DM100 2=XIO110 | r=yeni urun | q=cikis | d=debug")
        try:
            self._loop()
        finally:
            self.close()

    def initialize(self, *, enable_presenter: bool = False) -> None:
        """Prepare model/camera/db for either CLI(OpenCV) or Qt embedding."""
        self._validate()
        self._init_db()
        self._load_model()
        self._open_camera()
        if enable_presenter:
            self.presenter.setup()

    def close(self) -> None:
        """Release camera and any OpenCV windows if used."""
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            self.cap = None
        try:
            self.presenter.destroy()
        except Exception:
            pass

    def handle_action(self, action: str) -> None:
        """Qt UI hooks: action in {'space','r','d','1','2'}."""
        a = (action or "").lower().strip()
        if a == "d":
            self.label_debug = not self.label_debug
        elif a == "r":
            self._reset_session()
        elif a == "1":
            self._select_device("DM100")
        elif a == "2":
            self._select_device("XIO110")
        elif a == "space":
            self._on_space()

    def step(self) -> tuple[np.ndarray, list[str], list[dict]] | None:
        """Qt embedding: read 1 frame, run phase logic, return (image, panel_lines, recent_rows)."""
        assert self.cap is not None and self.model is not None and self.device is not None
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.frame_idx += 1
        self._update_fps()
        panel_lines = self._base_panel_lines()
        timed_out = self._apply_timeout(panel_lines)
        img0, panel_lines = self._process_phase(frame, panel_lines, timed_out)
        self._auto_transitions()
        try:
            rows = dbm.list_inspections(DB_PATH, limit=HISTORY_ROW_LIMIT)
        except Exception:
            rows = []
        return img0, panel_lines, rows

    def _validate(self) -> None:
        self.weights_path = Path(self.opt.weights).expanduser().resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {self.weights_path}")
        if not HAS_PYZBAR:
            print("Uyarı: pyzbar yok — barkod aşaması çalışmaz. pip install pyzbar (+ ZBar)")
        self.profiles = load_profiles(self.profiles_path)

    def _init_db(self) -> None:
        try:
            dbm.init_db(DB_PATH)
            dbm.load_profiles_into_db(DB_PATH, self.profiles)
        except Exception:
            pass

    def _load_model(self) -> None:
        t0 = time.perf_counter()
        print("Cihaz seciliyor...")
        self.device = select_device(
            self.opt.device if self.opt.device else ("0" if torch.cuda.is_available() else "cpu")
        )
        print(f"Model yukleniyor ({self.weights_path.name})...")
        self.model = attempt_load(str(self.weights_path), device=self.device)
        self.model.eval()
        names = self.model.names
        self.names_list = (
            [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
        )
        if self.device.type != "cpu":
            self.model.half()
            torch.backends.cudnn.benchmark = True
        print("Isinma (ilk inference)...")
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.img_size, self.img_size, device=self.device)
            if self.device.type != "cpu":
                dummy = dummy.half()
            _ = self.model(dummy)
        print(
            f"Hazir ({time.perf_counter() - t0:.1f}s) | cihaz={self.device} | "
            f"img={self.img_size} | infer_every={self.infer_every}"
        )

    def _open_camera(self) -> None:
        src = self.opt.source
        try:
            cam_id = int(src)
            cap = (
                cv2.VideoCapture(cam_id)
                if self.opt.no_dshow
                else cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            )
        except ValueError:
            cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Kaynak açılamadı: {self.opt.source}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap

    def _loop(self) -> None:
        assert self.cap is not None and self.model is not None and self.device is not None
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.frame_idx += 1
            self._update_fps()
            panel_lines = self._base_panel_lines()
            timed_out = self._apply_timeout(panel_lines)
            img0, panel_lines = self._process_phase(frame, panel_lines, timed_out)
            combined = self._compose_frame(img0, panel_lines)
            self.presenter.present(combined)
            if self._handle_key():
                break
            self._auto_transitions()

    def _update_fps(self) -> None:
        dt = time.perf_counter() - self.loop_t0
        self.loop_t0 = time.perf_counter()
        if dt > 0:
            self.fps_ema = self.fps_ema * 0.9 + (1.0 / dt) * 0.1

    def _base_panel_lines(self) -> list[str]:
        lines = [
            f"Asama: {self.phase.name}",
            f"Unit: {format_session_display(self.session)}",
        ]
        if self.opt.show_fps:
            lines.append(f"FPS: {self.fps_ema:.0f}")
        return lines

    def _apply_timeout(self, panel_lines: list[str]) -> bool:
        if self.phase not in (Phase.FRONT, Phase.LABEL, Phase.BARCODE):
            return False
        limit = PHASE_TIMEOUT_SEC[self.phase]
        elapsed = time.perf_counter() - self.phase_started_at
        panel_lines.append(f"Kalan: {int(max(0.0, limit - elapsed))} sn")
        if elapsed >= limit:
            if self.phase == Phase.FRONT:
                self._defer_fail_to_barcode(Phase.FRONT, [f"{self.phase.name.lower()}_timeout"])
                return False
            if self.phase == Phase.LABEL:
                self._defer_fail_to_barcode(Phase.LABEL, [f"{self.phase.name.lower()}_timeout"])
                return False
            abort_to_fail(self.session, self.phase, [f"{self.phase.name.lower()}_timeout"], DB_PATH)
            self.phase = Phase.RESULT
            self.result_start_time = time.perf_counter()
            return True
        return False

    def _process_phase(
        self, frame: np.ndarray, panel_lines: list[str], timed_out: bool
    ) -> tuple[np.ndarray, list[str]]:
        if timed_out or self.phase == Phase.RESULT:
            return self._phase_result(frame, panel_lines)
        if self.phase == Phase.FRONT:
            return self._phase_front(frame, panel_lines)
        if self.phase == Phase.LABEL:
            return self._phase_label(frame, panel_lines)
        return self._phase_barcode(frame, panel_lines)

    def _phase_result(self, frame: np.ndarray, panel_lines: list[str]) -> tuple[np.ndarray, list[str]]:
        img0 = frame.copy()
        if not self.session.result_saved:
            go_to_result(self.session, self.profiles, self.opt.strict, DB_PATH)
        banner_color = COLOR_PASS_BG if self.session.overall == "PASS" else COLOR_FAIL_BG
        draw_banner(img0, self.session.overall, banner_color)
        panel_lines.extend(
            [
                f"SONUC: {self.session.overall}",
                f"Cihaz: {self.session.device_display or self.session.device_key or '-'}",
                f"SN: {self.session.serial_number or '-'}",
            ]
        )
        if self.session.fail_phase:
            panel_lines.append(f"Hata asamasi: {self.session.fail_phase}")
        for r in self.session.fail_reasons[:6]:
            panel_lines.append(r[:30])
        panel_lines.append("r = yeni urun")
        return img0, panel_lines

    def _phase_front(self, frame: np.ndarray, panel_lines: list[str]) -> tuple[np.ndarray, list[str]]:
        img0, class_counts, _ = run_yolo_throttled(
            self.model,
            frame,
            self.device,
            self.img_size,
            self.opt.conf_front,
            self.opt.iou,
            self.names_list,
            self.yolo_cache,
            self.frame_idx,
            self.infer_every,
        )
        self._last_class_counts = class_counts
        by_name = build_name_to_count(self.names_list, class_counts)
        self.session.front_by_name = dict(by_name)

        guess = None
        guess_key, profile = None, None
        if self.session.device_key and self.session.device_key in self.profiles and self.session.device_identified:
            profile = self.profiles[self.session.device_key]
            guess_key = self.session.device_key
        else:
            guess = identify_device_from_components(by_name, self.profiles)
            guess_key, profile = guess.key, guess.profile

        panel_lines.append("1) Komponent sayimi")
        panel_lines.extend(format_raw_front_lines(by_name, self.profiles)[:7])

        if self.session.device_key and self.session.device_identified and profile:
            panel_lines.append(f"2) Cihaz: {profile.display_name} (manuel)")
        elif guess_key and profile:
            panel_lines.append(f"2) Tahmin: {profile.display_name}")
            if guess is not None:
                panel_lines.append(f"   skor {guess.score:.0%}")
        else:
            panel_lines.extend(
                ["2) Cihaz: henuz belirlenmedi", "   (sayimlar oturunca)", "1=DM100 2=XIO110"]
            )

        if guess_key and profile:
            front_val = validate_section(
                filter_for_front_id(by_name, self.profiles),
                profile.front_components,
                "front",
            )
            sig = f"{guess_key}|" + "|".join(format_mismatches(front_val.mismatches))
            panel_lines.extend(format_counts_line(by_name, profile, "front")[:6])
            if front_val.ok:
                self.stable_count = self.stable_count + 1 if sig == self.last_stable_signature else 1
                self.last_stable_signature = sig
                if self.stable_count >= STABLE_FRAMES_FRONT:
                    self.session.device_key = guess_key
                    self.session.device_display = profile.display_name
                    self.session.device_identified = True
            else:
                self.stable_count = 0
                self.last_stable_signature = sig
            if self.opt.strict and not front_val.ok:
                panel_lines.append("On yuz: sayim hatali")
            elif self.session.device_identified and front_val.ok:
                panel_lines.append(">>> Space ile devam")
            elif self.stable_count > 0 and front_val.ok:
                panel_lines.append(f"Sabitlik: {self.stable_count}/{STABLE_FRAMES_FRONT}")
        else:
            self.stable_count = 0
            self.last_stable_signature = ""

        draw_banner(img0, PHASE_TITLES[self.phase], PANEL_ACCENT_2)
        return img0, panel_lines

    def _phase_label(self, frame: np.ndarray, panel_lines: list[str]) -> tuple[np.ndarray, list[str]]:
        profile = self.profiles.get(self.session.device_key) if self.session.device_key else None
        img0, class_counts, _ = run_yolo_throttled(
            self.model,
            frame,
            self.device,
            self.img_size,
            self.opt.conf_label,
            self.opt.iou,
            self.names_list,
            self.yolo_cache,
            self.frame_idx,
            self.infer_every,
        )
        by_name = build_name_to_count(self.names_list, class_counts)
        merge_peak_counts(self.session.label_by_name_peak, by_name)

        if profile:
            label_val = validate_section(
                self.session.label_by_name_peak, profile.label_components, "labels"
            )
            label_ok_now = label_val.ok
            panel_lines.append(f"Cihaz: {profile.display_name}")
            panel_lines.extend(format_counts_line(self.session.label_by_name_peak, profile, "labels"))
            if label_ok_now:
                self.stable_count += 1
                panel_lines.append(">>> Space ile devam")
            else:
                self.stable_count = 0
                panel_lines.append("Garanti+Kalite: eksik")
        else:
            panel_lines.append("Profil yok")

        draw_banner(img0, PHASE_TITLES[self.phase], (160, 100, 60))
        return img0, panel_lines

    def _phase_barcode(self, frame: np.ndarray, panel_lines: list[str]) -> tuple[np.ndarray, list[str]]:
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
                cand, _ = detect_label_candidate(frame, brect=brect, debug=self.label_debug)
                if cand is not None:
                    self.session.label_cv_box_seen = True
                    x, y, w, h = cand
                    cv2.rectangle(img0, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if self.session.pending_fail_phase == "FRONT":
            panel_lines.append("On yuz FAIL — barkodu okutun (cihaz kaydi)")
        elif self.session.pending_fail_phase == "LABEL":
            panel_lines.append("Etiket FAIL — barkodu okutun (cihaz kaydi)")
        if serial:
            self.session.serial_number = serial
            self.session.barcode_raw = serial
            self.session.barcode_ok = True
            panel_lines.append(f"SN: {serial[:24]}")
            if self.session.pending_fail_phase or not (
                self.session.front_ok and self.session.label_ok
            ):
                self._finalize_fail_after_barcode()
            else:
                go_to_result(self.session, self.profiles, self.opt.strict, DB_PATH)
                self.phase = Phase.RESULT
                self.result_start_time = time.perf_counter()
        else:
            panel_lines.append("Barkod bekleniyor...")
        draw_banner(img0, PHASE_TITLES[self.phase], PANEL_ACCENT)
        return img0, panel_lines

    def _compose_frame(self, img0: np.ndarray, panel_lines: list[str]) -> np.ndarray:
        h = img0.shape[0]
        result_col = None
        if self.phase == Phase.RESULT:
            result_col = (0, 140, 0) if self.session.overall == "PASS" else (0, 0, 200)
        panel = build_panel(h, panel_lines, result_col)
        history_panel = build_history_panel(h, DB_PATH, self.profiles)
        return np.concatenate([panel, img0, history_panel], axis=1)

    def _handle_key(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            try:
                from launcher.win32_helper import Win32Helper

                Win32Helper.allow_parent_foreground()
            except Exception:
                pass
            return True
        if key == ord("d"):
            self.label_debug = not self.label_debug
        elif key == ord("r"):
            self._reset_session()
        elif key == ord("1"):
            self._select_device("DM100")
        elif key == ord("2"):
            self._select_device("XIO110")
        elif key == ord(" "):
            self._on_space()
        return False

    def _select_device(self, key: str) -> None:
        if key not in self.profiles:
            return
        self.session.device_key = key
        self.session.device_display = self.profiles[key].display_name
        self.session.device_identified = True

    def _on_space(self) -> None:
        if self.phase == Phase.FRONT:
            profile = self.profiles.get(self.session.device_key) if self.session.device_key else None
            if not profile or not self.session.device_identified:
                print("Cihaz tipi belirlenmedi — sayim otursun veya 1/2 ile secin.")
                return
            by_name = build_name_to_count(self.names_list, self._last_class_counts)
            self.session.front_by_name = dict(by_name)
            if self.opt.strict:
                front_val = validate_section(
                    filter_for_front_id(by_name, self.profiles),
                    profile.front_components,
                    "front",
                )
                self.session.front_ok = front_val.ok
            else:
                self.session.front_ok = True
            if not self.session.front_ok:
                reasons = (
                    format_mismatches(front_val.mismatches)
                    if self.opt.strict
                    else ["on_yuz_kosullari_saglanmadi"]
                )
                self._defer_fail_to_barcode(Phase.FRONT, reasons)
                print("On yuz FAIL -> barkod okutun")
                return
            self.session.front_status = "PASS"
            self.phase = Phase.LABEL
            self._reset_stable()
            print("On yuz OK -> etiket asamasi")
        elif self.phase == Phase.LABEL:
            profile = self.profiles.get(self.session.device_key)
            if profile:
                label_val = validate_section(
                    self.session.label_by_name_peak, profile.label_components, "labels"
                )
                self.session.label_ok = label_val.ok if self.opt.strict else True
            else:
                self.session.label_ok = False
            if not self.session.label_ok:
                reasons = (
                    format_mismatches(label_val.mismatches)
                    if profile and self.opt.strict
                    else ["etiket_kontrolu_fail"]
                )
                self._defer_fail_to_barcode(Phase.LABEL, reasons)
                print("Etiket FAIL -> barkod okutun")
                return
            self.session.label_status = "PASS"
            self.session.label_surfaces_done += 1
            self.phase = Phase.BARCODE
            self._reset_stable()
            print("Etiket OK -> barkod asamasi")
        elif self.phase == Phase.BARCODE:
            if not self.session.barcode_ok:
                print("Barkod okunamadi — Space ile gecilemez.")
            elif self.session.pending_fail_phase or not (
                self.session.front_ok and self.session.label_ok
            ):
                self._finalize_fail_after_barcode()
            else:
                go_to_result(self.session, self.profiles, self.opt.strict, DB_PATH)
                self.phase = Phase.RESULT
                self.result_start_time = time.perf_counter()
        elif self.phase == Phase.RESULT:
            self._reset_session()

    def _auto_transitions(self) -> None:
        if (
            self.phase == Phase.FRONT
            and self.stable_count >= STABLE_FRAMES_FRONT
            and self.session.device_identified
            and self.session.device_key
        ):
            profile = self.profiles[self.session.device_key]
            if self.opt.strict:
                self.session.front_ok = validate_section(
                    filter_for_front_id(self.session.front_by_name, self.profiles),
                    profile.front_components,
                    "front",
                ).ok
            else:
                self.session.front_ok = True
            if self.session.front_ok:
                self.session.front_status = "PASS"
                self.phase = Phase.LABEL
                self._reset_stable()
                print("Otomatik: on yuz tamam -> etiket asamasi")
        elif self.phase == Phase.LABEL and self.stable_count >= STABLE_FRAMES_LABEL:
            profile = self.profiles.get(self.session.device_key)
            if profile:
                if self.opt.strict:
                    self.session.label_ok = validate_section(
                        self.session.label_by_name_peak, profile.label_components, "labels"
                    ).ok
                else:
                    self.session.label_ok = True
            else:
                self.session.label_ok = False
            if self.session.label_ok:
                self.session.label_status = "PASS"
                self.phase = Phase.BARCODE
                self._reset_stable()
                print("Otomatik: etiket tamam -> barkod asamasi")

        if (
            self.phase == Phase.RESULT
            and self.session.result_saved
            and self.result_start_time > 0.0
            and time.perf_counter() - self.result_start_time >= RESULT_DISPLAY_SECONDS
        ):
            self._reset_session()
            print("Otomatik: yeni urun -> FRONT")

    def _reset_stable(self) -> None:
        self.phase_started_at = time.perf_counter()
        self.stable_count = 0

    def _defer_fail_to_barcode(self, failed_phase: Phase, reasons: list[str]) -> None:
        defer_fail_to_barcode(self.session, failed_phase, reasons)
        self.phase = Phase.BARCODE
        self._reset_stable()
        phase_tr = PHASE_LABEL_TR.get(failed_phase.name, failed_phase.name)
        print(f"{phase_tr} FAIL -> barkod okutun (SN ile kayit)")

    def _finalize_fail_after_barcode(self) -> None:
        complete_fail_after_barcode(
            self.session, self.profiles, self.opt.strict, DB_PATH
        )
        self.phase = Phase.RESULT
        self.result_start_time = time.perf_counter()
        print(f"FAIL kaydedildi: SN={self.session.serial_number or '-'}")

    def _create_new_session(self) -> InspectionSession:
        self._unit_counter += 1
        return new_session(unit_number=self._unit_counter)

    def _reset_session(self) -> None:
        self.session = self._create_new_session()
        self.phase = Phase.FRONT
        self.phase_started_at = time.perf_counter()
        self.stable_count = 0
        self.last_stable_signature = ""
        self.yolo_cache = YoloCache()
        self.result_start_time = 0.0


def main():
    opt = parse_opt()
    try:
        QualityControlApp(opt).run()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except RuntimeError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
