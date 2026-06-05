"""Persistent QC launcher settings (config/app_settings.json)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
SETTINGS_PATH = CONFIG_DIR / "app_settings.json"
DEFAULT_WEIGHTS = (ROOT.parent / "best.pt").resolve()

IMG_SIZE_PRESETS: tuple[tuple[int, str], ...] = (
    (640, "640 — Accurate (default)"),
    (512, "512 — Balanced"),
    (416, "416 — Fast"),
)
ALLOWED_IMG_SIZES = {size for size, _ in IMG_SIZE_PRESETS}


@dataclass
class AppSettings:
    source: str = "0"
    weights: str = str(DEFAULT_WEIGHTS)
    device: str = ""
    img_size: int = 640
    conf_front: float = 0.88
    conf_label: float = 0.88
    iou: float = 0.45
    infer_every: int = 2
    strict: bool = True
    no_dshow: bool = False
    show_fps: bool = False


def default_settings() -> AppSettings:
    return AppSettings()


def load_settings() -> AppSettings:
    if not SETTINGS_PATH.is_file():
        return default_settings()
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        s = default_settings()
        for f in fields(AppSettings):
            if f.name in data:
                setattr(s, f.name, data[f.name])
        return normalize_settings(s)
    except Exception:
        return default_settings()


def normalize_settings(s: AppSettings) -> AppSettings:
    s.source = (s.source or "0").strip()
    s.weights = (s.weights or str(DEFAULT_WEIGHTS)).strip()
    s.device = (s.device or "").strip()
    s.img_size = int(s.img_size)
    if s.img_size not in ALLOWED_IMG_SIZES:
        s.img_size = 640
    s.conf_front = float(min(0.99, max(0.05, s.conf_front)))
    s.conf_label = float(min(0.99, max(0.05, s.conf_label)))
    s.iou = float(min(0.90, max(0.10, s.iou)))
    s.infer_every = int(min(5, max(1, s.infer_every)))
    s.strict = bool(s.strict)
    s.no_dshow = bool(s.no_dshow)
    s.show_fps = bool(s.show_fps)
    return s


def save_settings(s: AppSettings) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    normalized = normalize_settings(s)
    SETTINGS_PATH.write_text(
        json.dumps(asdict(normalized), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_settings(s: AppSettings) -> tuple[bool, str]:
    s = normalize_settings(s)
    weights = Path(s.weights).expanduser()
    if not weights.is_file():
        return False, f"Model weights not found:\n{weights}"
    return True, ""


def settings_to_opt_kwargs(s: AppSettings | None = None) -> dict:
    """Map saved settings to qc_engine `default_opt(**kwargs)` overrides."""
    s = normalize_settings(s or load_settings())
    return {
        "source": s.source,
        "weights": s.weights,
        "device": s.device,
        "img_size": s.img_size,
        "conf_front": s.conf_front,
        "conf_label": s.conf_label,
        "iou": s.iou,
        "infer_every": s.infer_every,
        "strict": s.strict,
        "no_dshow": s.no_dshow,
        "show_fps": s.show_fps,
    }
