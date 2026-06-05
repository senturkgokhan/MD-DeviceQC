"""PySide6 launcher for embedded qc_engine QC.

Home: Start / Test Results / Settings / Exit
Start: in-process camera + YOLO (center video, Qt side panels). Esc = Home.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QKeySequence, QLinearGradient, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import ( 
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QStackedLayout,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from app_settings import (
    AppSettings,
    IMG_SIZE_PRESETS,
    default_settings,
    load_settings,
    save_settings,
    settings_to_opt_kwargs,
    validate_settings,
)
from utils import db as dbm

if TYPE_CHECKING:
    from qc_engine import QualityControlApp

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
ICON_PATH = ASSETS_DIR / "app_icon.png"
DB_PATH = ROOT / "results" / "devices.db"

HOME_WINDOW_SIZE = (500, 600)
APP_VERSION = "1.0"
CAMERA_WINDOW_SIZE = (1320, 620)

FONT_UI = "Segoe UI"
FONT_DISPLAY = "Bahnschrift SemiBold"

PHASE_TITLE_EN = {
    "FRONT": "Front inspection",
    "LABEL": "Label inspection",
    "BARCODE": "Barcode scan",
    "RESULT": "Inspection result",
}

PANEL_LINE_TR_EN = [
    ("Asama:", "Phase:"),
    ("Unit:", "Unit:"),
    ("Kalan:", "Time left:"),
    ("Komponent sayimi", "Component count"),
    ("Komponent:", "Components:"),
    ("Cihaz:", "Device:"),
    ("Tahmin:", "Guess:"),
    ("henuz belirlenmedi", "not determined yet"),
    ("sayimlar oturunca", "when counts stabilize"),
    ("On yuz:", "Front:"),
    ("Garanti+", "Warranty label:"),
    ("Barkod", "Barcode"),
    ("Onceki", "Previous step"),
    ("SONUC:", "Result:"),
    ("SN:", "Serial:"),
    ("Hata asamasi:", "Fail stage:"),
    ("Sabitlik:", "Stability:"),
    (">>>", ">>>"),
    ("On yuz FAIL", "Front FAIL"),
    ("Etiket FAIL", "Label FAIL"),
    ("Barkod bekleniyor", "Waiting for barcode"),
    ("belirlenemedi", "undetermined"),
]


def translate_panel_line(line: str) -> str:
    s = line
    for tr, en in PANEL_LINE_TR_EN:
        s = s.replace(tr, en)
    return s


def parse_panel_state(lines: list) -> tuple[str, str, str, str, str, str]:
    """Returns phase_key, phase_title, timer_text, session_id, fps_text, body_text."""
    phase_key = ""
    timer = ""
    fps = ""
    body: list[str] = []
    session_id = ""

    for raw in lines:
        line = translate_panel_line(str(raw).strip())
        if line.startswith("Phase:"):
            phase_key = line.replace("Phase:", "").strip()
            continue
        if line.startswith("Unit:"):
            session_id = line.replace("Unit:", "").strip()
            continue
        if line.startswith("ID:"):
            session_id = line.replace("ID:", "").strip()
            continue
        if line.startswith("Time left:"):
            timer = line.replace("Time left:", "").strip()
            continue
        if line.startswith("FPS:"):
            fps = line.replace("FPS:", "").strip()
            continue
        body.append(line)

    title = PHASE_TITLE_EN.get(phase_key, phase_key or "Inspection")
    return phase_key, title, timer, session_id, fps, "\n".join(body).strip()


QSS = """
QMainWindow { background: transparent; }
QWidget#Root {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
    stop:0 #0c121c, stop:0.4 #121a2e, stop:1 #0e1424);
}
QFrame#Card, QFrame#HomeCard {
  background: rgba(20, 28, 46, 0.97);
  border: 1px solid rgba(100, 130, 220, 0.32);
  border-radius: 20px;
}
QFrame#LeftPanel {
  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
    stop:0 #1a2a44, stop:1 #141e34);
  border: 1px solid rgba(56, 189, 248, 0.45);
  border-radius: 16px;
}
QFrame#RightPanel {
  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
    stop:0 #222842, stop:1 #181e34);
  border: 1px solid rgba(167, 139, 250, 0.4);
  border-radius: 16px;
}
QFrame#InfoStrip {
  background: rgba(14, 22, 40, 0.88);
  border: 1px solid rgba(71, 85, 105, 0.55);
  border-radius: 12px;
  padding: 4px;
}
QLabel#SessionLabel {
  color: #94a3b8;
  font-size: 11px;
}
QLabel#SessionValue {
  color: #e2e8f0;
  font-size: 12px;
  font-weight: 600;
}
QLabel#FpsValue {
  color: #6ee7b7;
  font-size: 12px;
  font-weight: 700;
}
QTableWidget#HistoryTable {
  background: rgba(12, 16, 30, 0.78);
  border: 1px solid rgba(120, 130, 200, 0.35);
  border-radius: 12px;
  font-size: 12px;
}
QFrame#VideoShell {
  background: #0a0e18;
  border: 1px solid rgba(80, 110, 200, 0.35);
  border-radius: 16px;
}
QFrame#AccentBar {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #2dd4bf);
  border-radius: 2px; min-height: 4px; max-height: 4px;
}
QLabel#Title { color: #f8fafc; padding: 0 8px; }
QLabel#HomeTitle {
  color: #7dd3fc;
  padding: 4px 8px 2px 8px;
  letter-spacing: 0.02em;
}
QLabel#Subtitle { color: rgba(203, 213, 225, 0.78); }
QLabel#HomeSubtitle {
  color: rgba(186, 210, 235, 0.82);
  padding: 0 6px;
}
QLabel#Badge {
  color: #7dd3fc;
  background: rgba(37, 99, 235, 0.18);
  border: 1px solid rgba(59, 130, 246, 0.45);
  border-radius: 12px; padding: 6px 14px;
}
QLabel#PhaseBadge {
  color: #e0f2fe;
  background: rgba(14, 116, 144, 0.35);
  border: 1px solid rgba(45, 212, 191, 0.45);
  border-radius: 10px;
  padding: 8px 12px;
  font-weight: 700;
}
QLabel#TimerLabel {
  color: #fdba74;
  font-weight: 600;
}
QLabel#SectionTitle {
  color: #f1f5f9;
  font-size: 15px;
  font-weight: 700;
}
QLabel#Muted { color: rgba(148, 163, 184, 0.9); font-size: 11px; }
QLabel#HomeShortcuts {
  color: rgba(186, 230, 253, 0.92);
  font-size: 11px;
  font-weight: 600;
  padding: 12px 6px 2px 6px;
  border-top: 1px solid rgba(100, 140, 220, 0.28);
}
QLabel#HomePageCopyright {
  color: rgba(148, 163, 184, 0.48);
  padding: 0 16px 4px 16px;
}
QPlainTextEdit#StatusBox {
  background: rgba(14, 18, 32, 0.68);
  border: 1px solid rgba(80, 100, 160, 0.25);
  border-radius: 12px;
  color: rgba(226, 232, 240, 0.92);
  padding: 10px;
  font-size: 12px;
}
QPushButton#MenuBtn {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.14);
  color: #eef2ff; border-radius: 12px;
  padding: 14px 20px; min-height: 24px;
  font-size: 15px; font-weight: 600;
}
QPushButton#MenuBtn:hover {
  background: rgba(255, 255, 255, 0.12);
  border-color: rgba(147, 197, 253, 0.5);
}
QPushButton#Primary {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #2563eb);
  border: 1px solid rgba(96, 165, 250, 0.9);
  color: #ffffff; border-radius: 12px; padding: 14px 22px;
  font-size: 15px; font-weight: 700;
}
QPushButton#Primary:hover {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #60a5fa, stop:1 #3b82f6);
}
QPushButton#Danger {
  background: rgba(239, 68, 68, 0.16);
  border: 1px solid rgba(248, 113, 113, 0.5);
  color: #fecaca; border-radius: 12px; padding: 14px 20px;
  font-size: 15px; font-weight: 600;
}
QPushButton#GhostBtn {
  background: transparent;
  border: 1px solid rgba(148, 163, 184, 0.35);
  color: rgba(226, 232, 240, 0.9);
  border-radius: 10px;
  padding: 8px 14px;
  font-size: 12px;
}
QPushButton#GhostBtn:hover {
  border-color: rgba(147, 197, 253, 0.55);
  background: rgba(255, 255, 255, 0.06);
}

QDialog#ResultsDialog {
  background-color: #141c30;
  border: 1px solid rgba(120, 150, 255, 0.35);
  border-radius: 18px;
}

QTableWidget {
  background: rgba(16, 20, 36, 0.58);
  border: 1px solid rgba(120, 150, 255, 0.20);
  border-radius: 14px;
  gridline-color: rgba(120, 150, 255, 0.14);
}

QHeaderView::section {
  background: rgba(26, 36, 68, 0.75);
  color: rgba(240, 246, 255, 0.92);
  padding: 8px 10px;
  border: 1px solid rgba(120, 150, 255, 0.18);
}

QTableWidget::item {
  color: rgba(240, 246, 255, 0.92);
  padding: 6px 10px;
  font-size: 12px;
}

QTableWidget::item:selected {
  background: rgba(59, 130, 246, 0.30);
  border: none;
}

QDialog#SettingsDialog {
  background-color: #141c30;
  border: 1px solid rgba(120, 150, 255, 0.35);
  border-radius: 18px;
}
QTabWidget#SettingsTabs::pane {
  border: 1px solid rgba(100, 130, 200, 0.22);
  border-radius: 12px;
  background: rgba(10, 16, 30, 0.55);
  top: -1px;
}
QTabWidget#SettingsTabs QTabBar::tab {
  background: rgba(255, 255, 255, 0.05);
  color: rgba(203, 213, 225, 0.85);
  border: 1px solid rgba(100, 130, 200, 0.2);
  border-bottom: none;
  border-top-left-radius: 10px;
  border-top-right-radius: 10px;
  padding: 10px 18px;
  margin-right: 4px;
  font-weight: 600;
}
QTabWidget#SettingsTabs QTabBar::tab:selected {
  background: rgba(59, 130, 246, 0.22);
  color: #e0f2fe;
  border-color: rgba(96, 165, 250, 0.45);
}
QGroupBox#SettingsGroup {
  color: rgba(186, 210, 235, 0.92);
  font-weight: 600;
  border: 1px solid rgba(100, 130, 200, 0.22);
  border-radius: 12px;
  margin-top: 14px;
  padding: 16px 12px 12px 12px;
}
QGroupBox#SettingsGroup::title {
  subcontrol-origin: margin;
  left: 12px;
  padding: 0 6px;
}
QLabel#SettingsHint {
  color: rgba(148, 163, 184, 0.78);
  font-size: 11px;
}
QLabel#SettingsFieldLabel {
  color: rgba(186, 210, 235, 0.88);
  font-size: 12px;
}
QLineEdit#SettingsInput, QComboBox#SettingsInput, QSpinBox#SettingsInput, QDoubleSpinBox#SettingsInput {
  background: rgba(8, 12, 24, 0.85);
  border: 1px solid rgba(100, 130, 200, 0.28);
  border-radius: 10px;
  color: #eef2ff;
  padding: 8px 10px;
  min-height: 18px;
}
QLineEdit#SettingsInput:focus, QComboBox#SettingsInput:focus, QSpinBox#SettingsInput:focus, QDoubleSpinBox#SettingsInput:focus {
  border-color: rgba(96, 165, 250, 0.65);
}
QCheckBox#SettingsCheck {
  color: rgba(226, 232, 240, 0.92);
  spacing: 8px;
}
QCheckBox#SettingsCheck::indicator {
  width: 18px; height: 18px;
  border-radius: 5px;
  border: 1px solid rgba(100, 130, 200, 0.45);
  background: rgba(8, 12, 24, 0.85);
}
QCheckBox#SettingsCheck::indicator:checked {
  background: rgba(59, 130, 246, 0.85);
  border-color: rgba(147, 197, 253, 0.8);
}
"""


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    panel_ready = Signal(list)
    history_ready = Signal(list)
    error = Signal(str)
    started_ok = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stop = False
        self._app: "QualityControlApp | None" = None
        self._pending_actions: list[str] = []

    def request_stop(self) -> None:
        self._stop = True

    @Slot(str)
    def post_action(self, action: str) -> None:
        self._pending_actions.append(action)

    def run(self) -> None:
        try:
            # qc_engine pulls in torch/YOLO — load only when inspection starts, not at app launch.
            from qc_engine import QualityControlApp, default_opt

            opt = default_opt(
                headless=True,
                windowed=True,
                **settings_to_opt_kwargs(),
            )
            qc = QualityControlApp(opt)
            qc.initialize(enable_presenter=False)
            self._app = qc
            self.started_ok.emit()

            while not self._stop:
                # consume actions
                while self._pending_actions:
                    qc.handle_action(self._pending_actions.pop(0))

                out = qc.step()
                if out is None:
                    break
                img0, panel_lines, rows = out

                # BGR -> RGB QImage
                h, w = img0.shape[:2]
                rgb = img0[:, :, ::-1].copy()
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
                self.frame_ready.emit(qimg)
                self.panel_ready.emit(panel_lines)
                self.history_ready.emit(rows)

                self.msleep(1)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                if self._app is not None:
                    self._app.close()
            except Exception:
                pass
            self._app = None


class CameraPage(QWidget):
    home_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_pixmap: QPixmap | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # --- Left: inspection status ---
        left = QFrame(self)
        left.setObjectName("LeftPanel")
        left.setFixedWidth(278)
        lyt = QVBoxLayout(left)
        lyt.setContentsMargins(16, 16, 16, 16)
        lyt.setSpacing(10)

        hdr = QLabel("Inspection status", left)
        hdr.setObjectName("SectionTitle")
        hdr.setFont(QFont(FONT_DISPLAY, 14, QFont.Weight.Bold))
        lyt.addWidget(hdr)

        self.lbl_phase = QLabel("FRONT", left)
        self.lbl_phase.setObjectName("PhaseBadge")
        self.lbl_phase.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_phase.setFont(QFont(FONT_UI, 12, QFont.Weight.Bold))
        lyt.addWidget(self.lbl_phase)

        self.lbl_phase_title = QLabel("Front inspection", left)
        self.lbl_phase_title.setFont(QFont(FONT_UI, 13, QFont.Weight.DemiBold))
        self.lbl_phase_title.setStyleSheet("color: #f1f5f9; padding: 0 2px;")
        lyt.addWidget(self.lbl_phase_title)

        info = QFrame(left)
        info.setObjectName("InfoStrip")
        info_lyt = QVBoxLayout(info)
        info_lyt.setContentsMargins(12, 10, 12, 10)
        info_lyt.setSpacing(6)

        self.lbl_timer = QLabel("—", info)
        self.lbl_timer.setObjectName("TimerLabel")
        self.lbl_timer.setFont(QFont(FONT_UI, 13, QFont.Weight.Bold))
        info_lyt.addWidget(self.lbl_timer)

        sid_row = QHBoxLayout()
        info_lyt.addLayout(sid_row)
        sid_lbl = QLabel("Unit / SN", info)
        sid_lbl.setObjectName("SessionLabel")
        sid_row.addWidget(sid_lbl)
        self.lbl_session = QLabel("—", info)
        self.lbl_session.setObjectName("SessionValue")
        self.lbl_session.setWordWrap(True)
        sid_row.addWidget(self.lbl_session, 1)

        fps_row = QHBoxLayout()
        info_lyt.addLayout(fps_row)
        fps_lbl = QLabel("FPS", info)
        fps_lbl.setObjectName("SessionLabel")
        fps_row.addWidget(fps_lbl)
        self.lbl_fps = QLabel("—", info)
        self.lbl_fps.setObjectName("FpsValue")
        fps_row.addWidget(self.lbl_fps, 1)
        self.lbl_fps.setVisible(False)
        fps_lbl.setVisible(False)

        self._fps_label = fps_lbl
        lyt.addWidget(info)

        det_lbl = QLabel("Details", left)
        det_lbl.setObjectName("SessionLabel")
        det_lbl.setStyleSheet("color: #7dd3fc; font-weight: 600;")
        lyt.addWidget(det_lbl)

        self.txt_status = QPlainTextEdit(left)
        self.txt_status.setObjectName("StatusBox")
        self.txt_status.setReadOnly(True)
        self.txt_status.setMaximumBlockCount(80)
        self.txt_status.setPlaceholderText("Component counts and device hints…")
        self.txt_status.setMinimumHeight(120)
        self.txt_status.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lyt.addWidget(self.txt_status, 1)

        dev_lbl = QLabel("Quick device select", left)
        dev_lbl.setObjectName("SessionLabel")
        lyt.addWidget(dev_lbl)
        dev_row = QHBoxLayout()
        lyt.addLayout(dev_row)
        self.btn_dev1 = QPushButton("DM100", left)
        self.btn_dev1.setObjectName("MenuBtn")
        self.btn_dev2 = QPushButton("XIO110", left)
        self.btn_dev2.setObjectName("MenuBtn")
        dev_row.addWidget(self.btn_dev1)
        dev_row.addWidget(self.btn_dev2)

        self.lbl_hints = QLabel("Space confirm · 1/2 device · r new unit · q home", left)
        self.lbl_hints.setObjectName("Muted")
        self.lbl_hints.setWordWrap(True)
        lyt.addWidget(self.lbl_hints)

        # --- Center: live feed ---
        center = QVBoxLayout()
        center.setSpacing(12)

        top_bar = QHBoxLayout()
        center.addLayout(top_bar)
        self.btn_home = QPushButton("← Home", self)
        self.btn_home.setObjectName("GhostBtn")
        self.btn_home.clicked.connect(self.home_requested.emit)
        top_bar.addWidget(self.btn_home)
        top_bar.addStretch(1)
        live_lbl = QLabel("Live camera", self)
        live_lbl.setObjectName("Muted")
        top_bar.addWidget(live_lbl)

        video_shell = QFrame(self)
        video_shell.setObjectName("VideoShell")
        vlyt = QVBoxLayout(video_shell)
        vlyt.setContentsMargins(8, 8, 8, 8)
        self.lbl_video = QLabel("Starting camera…", video_shell)
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video.setMinimumSize(480, 300)
        self.lbl_video.setStyleSheet("color: rgba(148, 163, 184, 0.85); background: #080c14;")
        vlyt.addWidget(self.lbl_video, 1)
        center.addWidget(video_shell, 1)

        btns = QHBoxLayout()
        btns.setSpacing(10)
        center.addLayout(btns)
        self.btn_reset = QPushButton("↻  New unit", self)
        self.btn_reset.setObjectName("MenuBtn")
        self.btn_reset.setMinimumHeight(38)
        btns.addStretch(1)
        btns.addWidget(self.btn_reset)
        btns.addStretch(1)

        center_w = QWidget(self)
        center_w.setLayout(center)

        # --- Right: history ---
        right = QFrame(self)
        right.setObjectName("RightPanel")
        right.setFixedWidth(420)
        ryt = QVBoxLayout(right)
        ryt.setContentsMargins(16, 16, 16, 16)
        ryt.setSpacing(10)

        title = QLabel("Recent results", right)
        title.setObjectName("SectionTitle")
        title.setFont(QFont(FONT_DISPLAY, 14, QFont.Weight.Bold))
        ryt.addWidget(title)

        sub = QLabel("PASS / FAIL from database", right)
        sub.setObjectName("Muted")
        ryt.addWidget(sub)

        self.tbl_history = QTableWidget(0, 4, right)
        self.tbl_history.setObjectName("HistoryTable")
        self.tbl_history.setHorizontalHeaderLabels(["Time", "Device", "Barcode", "PASS/FAIL"])
        self.tbl_history.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_history.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_history.setAlternatingRowColors(True)
        self.tbl_history.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        hdr = self.tbl_history.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.tbl_history.setColumnWidth(0, 88)
        self.tbl_history.setColumnWidth(1, 108)
        self.tbl_history.setColumnWidth(3, 78)
        self.tbl_history.verticalHeader().setVisible(False)
        self.tbl_history.setShowGrid(False)
        self.tbl_history.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        ryt.addWidget(self.tbl_history, 1)

        self.lbl_db = QLabel(str(DB_PATH.name), right)
        self.lbl_db.setObjectName("Muted")
        self.lbl_db.setToolTip(str(DB_PATH))
        ryt.addWidget(self.lbl_db)

        root.addWidget(left)
        root.addWidget(center_w, 1)
        root.addWidget(right)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._last_pixmap is not None and not self._last_pixmap.isNull():
            self.lbl_video.setPixmap(
                self._last_pixmap.scaled(
                    self.lbl_video.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    @Slot(QImage)
    def set_frame(self, img: QImage) -> None:
        if img.isNull():
            return
        self._last_pixmap = QPixmap.fromImage(img)
        self.lbl_video.setPixmap(
            self._last_pixmap.scaled(
                self.lbl_video.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    @Slot(list)
    def set_panel_lines(self, lines: list) -> None:
        phase_key, title, timer, session_id, fps, body = parse_panel_state(lines)
        if phase_key:
            self.lbl_phase.setText(phase_key)
        self.lbl_phase_title.setText(title)
        if timer:
            self.lbl_timer.setText(f"Time left: {timer}")
            self.lbl_timer.setVisible(True)
        else:
            self.lbl_timer.setText("—")
            self.lbl_timer.setVisible(False)
        self.lbl_session.setText(session_id or "—")
        if fps:
            self.lbl_fps.setText(fps)
            self.lbl_fps.setVisible(True)
            self._fps_label.setVisible(True)
        else:
            self.lbl_fps.setVisible(False)
            self._fps_label.setVisible(False)
        self.txt_status.setPlainText(body)

    @Slot(list)
    def set_history_rows(self, rows: list) -> None:
        self.tbl_history.setRowCount(0)
        pass_color = QColor("#34d399")
        fail_color = QColor("#f87171")
        pass_bg = QColor(20, 55, 45)
        fail_bg = QColor(55, 28, 32)
        bold = QFont(FONT_UI, 11, QFont.Weight.Bold)

        for r in rows[:12]:
            row = self.tbl_history.rowCount()
            self.tbl_history.insertRow(row)
            device = r.get("device_type") or ""
            if device == "belirlenemedi":
                device = "Undetermined"
            overall = (r.get("overall") or "").upper()
            full_ts = r.get("inspected_at") or ""
            ts = full_ts
            if "T" in ts:
                ts = ts.replace("T", " ")[:19]
            barcode = r.get("barcode") or ""
            vals = [ts, device, barcode, overall]
            tooltips = [full_ts, device, barcode, overall]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                item.setToolTip(tooltips[c])
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                    if c == 3
                    else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                )
                if c == 3:
                    item.setFont(bold)
                    if overall == "PASS":
                        item.setForeground(pass_color)
                        item.setBackground(pass_bg)
                    elif overall:
                        item.setForeground(fail_color)
                        item.setBackground(fail_bg)
                self.tbl_history.setItem(row, c, item)

        self.tbl_history.setColumnWidth(1, 108)
        self.tbl_history.setColumnWidth(3, 78)

class ResultsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, *, limit: int = 200) -> None:
        super().__init__(parent)
        self.setObjectName("ResultsDialog")
        self.setWindowTitle("Test Results")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumSize(900, 560)

        self._limit = limit
        self._build_ui()
        self._reload()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Test Results")
        title.setFont(QFont(FONT_DISPLAY, 18, QFont.Weight.Bold))
        title.setStyleSheet("color: rgba(244, 247, 255, 0.98);")
        layout.addWidget(title)

        subtitle = QLabel("Inspection history from devices.db")
        subtitle.setFont(QFont(FONT_UI, 10))
        subtitle.setStyleSheet("color: rgba(180, 190, 215, 0.72);")
        layout.addWidget(subtitle)

        meta = QLabel(f"Database: {DB_PATH}")
        meta.setFont(QFont(FONT_UI, 9))
        meta.setStyleSheet("color: rgba(180, 190, 215, 0.70);")
        layout.addWidget(meta)

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setObjectName("MenuBtn")
        self.btn_refresh.clicked.connect(self._reload)
        btn_row.addWidget(self.btn_refresh)

        self.btn_reset = QPushButton("Reset Database")
        self.btn_reset.setObjectName("Danger")
        self.btn_reset.clicked.connect(self._on_reset)
        btn_row.addWidget(self.btn_reset)

        btn_row.addStretch(1)

        self.btn_close = QPushButton("Close")
        self.btn_close.setObjectName("MenuBtn")
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_close)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Date", "Device Type", "Barcode", "Overall", "Fail Phase"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        for i, w in enumerate([170, 170, 220, 120, 160]):
            self.table.setColumnWidth(i, w)

        layout.addWidget(self.table, 1)

    def _reload(self) -> None:
        rows = dbm.list_inspections(DB_PATH, limit=self._limit)
        self.table.setRowCount(len(rows))

        pass_color = QColor("#2dd4bf")
        fail_color = QColor("#fca5a5")

        for idx, r in enumerate(rows, start=1):
            row = idx - 1
            overall = (r.get("overall") or "").upper()
            fail_phase = r.get("fail_phase") or ""
            device_type = r.get("device_type") or ""
            if device_type == "belirlenemedi":
                device_type = "Undetermined"
            cells = [
                r.get("inspected_at") or "",
                device_type,
                r.get("barcode") or "",
                overall,
                str(fail_phase) if fail_phase else "",
            ]
            for col, val in enumerate(cells):
                item = QTableWidgetItem(val)
                if overall == "PASS":
                    item.setForeground(pass_color)
                elif overall:
                    item.setForeground(fail_color)
                self.table.setItem(row, col, item)

    def _on_reset(self) -> None:
        answer = QMessageBox.question(
            self,
            "Reset Database",
            "All test results (inspections) will be deleted.\n"
            "Device profile definitions will be kept.\n\n"
            "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        ok, msg = dbm.reset_inspections(DB_PATH)
        if ok:
            self._reload()
            QMessageBox.information(self, "Reset Database", msg)
        else:
            QMessageBox.critical(self, "Reset Database", msg)


class SettingsDialog(QDialog):
    """Inspection / camera / model settings — applied on next Start Inspection."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SettingsDialog")
        self.setWindowTitle("Settings")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.resize(620, 520)
        self.setMinimumSize(560, 480)

        self._settings = load_settings()
        self._build_ui()
        self._load_into_form(self._settings)

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SettingsFieldLabel")
        return lbl

    @staticmethod
    def _hint(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SettingsHint")
        lbl.setWordWrap(True)
        return lbl

    @staticmethod
    def _input_widget(w: QWidget) -> None:
        w.setObjectName("SettingsInput")

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 18)
        root.setSpacing(12)

        title = QLabel("Settings")
        title.setFont(QFont(FONT_DISPLAY, 18, QFont.Weight.Bold))
        title.setStyleSheet("color: rgba(244, 247, 255, 0.98);")
        root.addWidget(title)

        subtitle = QLabel("Configure camera, model, and detection behavior")
        subtitle.setFont(QFont(FONT_UI, 10))
        subtitle.setStyleSheet("color: rgba(180, 190, 215, 0.72);")
        root.addWidget(subtitle)

        root.addWidget(
            self._hint(
                "Changes take effect on the next Start Inspection. "
                "If a session is running, finish or return Home first."
            )
        )

        tabs = QTabWidget()
        tabs.setObjectName("SettingsTabs")
        tabs.addTab(self._build_camera_tab(), "Camera")
        tabs.addTab(self._build_model_tab(), "Model")
        tabs.addTab(self._build_detection_tab(), "Detection")
        root.addWidget(tabs, 1)

        btn_row = QHBoxLayout()
        root.addLayout(btn_row)

        self.btn_defaults = QPushButton("Restore Defaults")
        self.btn_defaults.setObjectName("GhostBtn")
        self.btn_defaults.clicked.connect(self._on_restore_defaults)
        btn_row.addWidget(self.btn_defaults)

        btn_row.addStretch(1)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("MenuBtn")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Save")
        self.btn_save.setObjectName("Primary")
        self.btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(self.btn_save)

    def _build_camera_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(12)

        group = QGroupBox("Video source")
        group.setObjectName("SettingsGroup")
        form = QFormLayout(group)
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.ed_source = QLineEdit()
        self._input_widget(self.ed_source)
        self.ed_source.setPlaceholderText("0 = default webcam, 1 = second camera, or video file path")
        form.addRow(self._field_label("Camera / source"), self.ed_source)

        self.chk_no_dshow = QCheckBox("Disable DirectShow (try if the camera fails on Windows)")
        self.chk_no_dshow.setObjectName("SettingsCheck")
        form.addRow("", self.chk_no_dshow)

        self.chk_show_fps = QCheckBox("Show FPS in the status panel during inspection")
        self.chk_show_fps.setObjectName("SettingsCheck")
        form.addRow("", self.chk_show_fps)

        layout.addWidget(group)
        layout.addWidget(
            self._hint(
                "Use camera index 0 for the built-in or primary USB camera. "
                "Enable DirectShow disable only when OpenCV cannot open the device."
            )
        )
        layout.addStretch(1)
        return page

    def _build_model_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(12)

        weights_group = QGroupBox("YOLO weights")
        weights_group.setObjectName("SettingsGroup")
        wform = QFormLayout(weights_group)
        wform.setSpacing(12)

        weights_row = QHBoxLayout()
        self.ed_weights = QLineEdit()
        self._input_widget(self.ed_weights)
        weights_row.addWidget(self.ed_weights, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.setObjectName("MenuBtn")
        btn_browse.clicked.connect(self._browse_weights)
        weights_row.addWidget(btn_browse)
        wform.addRow(self._field_label("Weights file (.pt)"), weights_row)

        self.cmb_device = QComboBox()
        self._input_widget(self.cmb_device)
        self.cmb_device.addItem("Auto (GPU if available)", "")
        self.cmb_device.addItem("GPU 0", "0")
        self.cmb_device.addItem("CPU", "cpu")
        wform.addRow(self._field_label("Compute device"), self.cmb_device)

        layout.addWidget(weights_group)

        perf_group = QGroupBox("Speed vs accuracy")
        perf_group.setObjectName("SettingsGroup")
        pform = QFormLayout(perf_group)
        pform.setSpacing(12)

        self.cmb_img_size = QComboBox()
        self._input_widget(self.cmb_img_size)
        for size, label in IMG_SIZE_PRESETS:
            self.cmb_img_size.addItem(label, size)
        pform.addRow(self._field_label("Detection input size"), self.cmb_img_size)

        self.cmb_infer_every = QComboBox()
        self._input_widget(self.cmb_infer_every)
        self.cmb_infer_every.addItem("1 — Every frame (most accurate, slowest)", 1)
        self.cmb_infer_every.addItem("2 — Every 2 frames (recommended)", 2)
        self.cmb_infer_every.addItem("3 — Every 3 frames (fastest preview)", 3)
        pform.addRow(self._field_label("YOLO inference rate"), self.cmb_infer_every)

        layout.addWidget(perf_group)
        layout.addWidget(
            self._hint(
                "Smaller input sizes load faster and run smoother. "
                "Use 640 if detection misses small components."
            )
        )
        layout.addStretch(1)
        return page

    def _build_detection_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(12)

        group = QGroupBox("Quality rules")
        group.setObjectName("SettingsGroup")
        form = QFormLayout(group)
        form.setSpacing(12)

        self.chk_strict = QCheckBox("Strict component count validation (recommended for production)")
        self.chk_strict.setObjectName("SettingsCheck")
        form.addRow("", self.chk_strict)

        self.spin_conf_front = QDoubleSpinBox()
        self._input_widget(self.spin_conf_front)
        self.spin_conf_front.setRange(0.50, 0.99)
        self.spin_conf_front.setSingleStep(0.01)
        self.spin_conf_front.setDecimals(2)
        form.addRow(self._field_label("Front confidence threshold"), self.spin_conf_front)

        self.spin_conf_label = QDoubleSpinBox()
        self._input_widget(self.spin_conf_label)
        self.spin_conf_label.setRange(0.50, 0.99)
        self.spin_conf_label.setSingleStep(0.01)
        self.spin_conf_label.setDecimals(2)
        form.addRow(self._field_label("Label confidence threshold"), self.spin_conf_label)

        self.spin_iou = QDoubleSpinBox()
        self._input_widget(self.spin_iou)
        self.spin_iou.setRange(0.20, 0.70)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setDecimals(2)
        form.addRow(self._field_label("IOU (overlap filter)"), self.spin_iou)

        layout.addWidget(group)
        layout.addWidget(
            self._hint(
                "Higher confidence reduces false detections but may miss faint parts. "
                "Strict mode compares detected component counts against device profiles (DM100 / XIO110)."
            )
        )
        layout.addStretch(1)
        return page

    def _load_into_form(self, s: AppSettings) -> None:
        self.ed_source.setText(s.source)
        self.ed_weights.setText(s.weights)
        self.chk_no_dshow.setChecked(s.no_dshow)
        self.chk_show_fps.setChecked(s.show_fps)
        self.chk_strict.setChecked(s.strict)
        self.spin_conf_front.setValue(s.conf_front)
        self.spin_conf_label.setValue(s.conf_label)
        self.spin_iou.setValue(s.iou)

        idx = self.cmb_device.findData(s.device)
        self.cmb_device.setCurrentIndex(idx if idx >= 0 else 0)

        idx = self.cmb_img_size.findData(s.img_size)
        self.cmb_img_size.setCurrentIndex(idx if idx >= 0 else 0)

        idx = self.cmb_infer_every.findData(s.infer_every)
        self.cmb_infer_every.setCurrentIndex(idx if idx >= 0 else 1)

    def _collect_form(self) -> AppSettings:
        return AppSettings(
            source=self.ed_source.text().strip() or "0",
            weights=self.ed_weights.text().strip(),
            device=str(self.cmb_device.currentData() or ""),
            img_size=int(self.cmb_img_size.currentData()),
            conf_front=float(self.spin_conf_front.value()),
            conf_label=float(self.spin_conf_label.value()),
            iou=float(self.spin_iou.value()),
            infer_every=int(self.cmb_infer_every.currentData()),
            strict=self.chk_strict.isChecked(),
            no_dshow=self.chk_no_dshow.isChecked(),
            show_fps=self.chk_show_fps.isChecked(),
        )

    def _browse_weights(self) -> None:
        start = self.ed_weights.text().strip() or str(ROOT.parent)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO weights",
            start,
            "PyTorch weights (*.pt);;All files (*.*)",
        )
        if path:
            self.ed_weights.setText(path)

    def _on_restore_defaults(self) -> None:
        answer = QMessageBox.question(
            self,
            "Restore Defaults",
            "Reset all settings to factory defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._load_into_form(default_settings())

    def _on_save(self) -> None:
        settings = self._collect_form()
        ok, msg = validate_settings(settings)
        if not ok:
            QMessageBox.warning(self, "Settings", msg)
            return
        save_settings(settings)
        self._settings = load_settings()
        self.accept()


class AppIconFactory:
    @staticmethod
    def load() -> QIcon:
        if ICON_PATH.is_file():
            return QIcon(str(ICON_PATH))
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        pix = AppIconFactory.create_pixmap(256)
        pix.save(str(ICON_PATH), "PNG")
        return QIcon(pix)

    @staticmethod
    def create_pixmap(size: int = 256) -> QPixmap:
        pix = QPixmap(size, size)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        margin = size * 0.08
        rect = margin, margin, size - 2 * margin, size - 2 * margin
        grad = QLinearGradient(0, 0, size, size)
        grad.setColorAt(0.0, QColor("#1e3a8a"))
        grad.setColorAt(1.0, QColor("#0f766e"))
        p.setBrush(grad)
        p.setPen(QPen(QColor("#93c5fd"), max(2, size // 64)))
        p.drawRoundedRect(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]), int(size * 0.22), int(size * 0.22))
        box_pen = QPen(QColor("#34d399"), max(2, size // 80))
        p.setPen(box_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for bx, by, bw, bh in ((0.22, 0.24, 0.38, 0.36), (0.52, 0.30, 0.30, 0.28), (0.30, 0.52, 0.42, 0.32)):
            p.drawRoundedRect(int(size * bx), int(size * by), int(size * bw), int(size * bh), 4, 4)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#f8fafc"))
        p.setFont(QFont(FONT_UI, int(size * 0.17), QFont.Weight.Bold))
        p.drawText(pix.rect(), int(Qt.AlignmentFlag.AlignCenter), "QC")
        p.end()
        return pix


class LauncherWindow(QMainWindow):
    def __init__(self, app_icon: QIcon | None = None) -> None:
        super().__init__()
        self.setWindowTitle("YOLO-Based Industrial QC")
        if app_icon is not None:
            self.setWindowIcon(app_icon)
        self.setMinimumSize(480, 520)
        self._set_window_size(*HOME_WINDOW_SIZE)

        self._startup_dialog: QProgressDialog | None = None
        self._stack: QStackedLayout | None = None
        self._home_widget: QWidget | None = None
        self._camera_page: CameraPage | None = None
        self._worker: CameraWorker | None = None
        self._camera_shortcuts: list[QShortcut] = []

        self._build_ui()
        self._center_on_screen()

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)

        host = QWidget(root)
        self._stack = QStackedLayout(host)
        outer.addWidget(host)

        home = QWidget(host)
        home_l = QVBoxLayout(home)
        home_l.setContentsMargins(16, 20, 16, 10)
        home_l.setSpacing(0)
        home_l.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        home_l.addWidget(self._build_menu_card(), alignment=Qt.AlignmentFlag.AlignHCenter)
        home_l.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        home_l.addWidget(self._build_home_page_footer(home), alignment=Qt.AlignmentFlag.AlignHCenter)
        self._home_widget = home
        self._stack.addWidget(home)

        cam = CameraPage(host)
        cam.home_requested.connect(lambda: self._stop_worker(show_home=True))
        self._camera_page = cam
        self._stack.addWidget(cam)
        self._stack.setCurrentWidget(home)

    @staticmethod
    def _font(family: str, size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
        f = QFont(family, size)
        f.setWeight(weight)
        return f

    @staticmethod
    def _style_button(btn: QPushButton, *, primary: bool = False, danger: bool = False) -> None:
        btn.setMinimumHeight(48)
        if primary:
            btn.setObjectName("Primary")
        elif danger:
            btn.setObjectName("Danger")
        else:
            btn.setObjectName("MenuBtn")

    def _build_home_page_footer(self, parent: QWidget) -> QWidget:
        """Copyright only — below the card on the window background."""
        wrap = QWidget(parent)
        wrap.setFixedWidth(420)
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        year = datetime.now().year
        copyright_lbl = QLabel(
            f"© {year} Industrial Quality Control System\n"
            f"All rights reserved  ·  Version {APP_VERSION}\n"
            "YOLOv5-based device inspection software"
        )
        copyright_lbl.setObjectName("HomePageCopyright")
        copyright_lbl.setFont(self._font(FONT_UI, 9))
        copyright_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copyright_lbl.setWordWrap(True)
        layout.addWidget(copyright_lbl)
        return wrap

    def _build_menu_card(self) -> QWidget:
        card = QFrame(self)
        card.setObjectName("HomeCard")
        card.setFixedWidth(360)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(10)

        accent = QFrame(card)
        accent.setObjectName("AccentBar")
        layout.addWidget(accent)
        layout.addSpacing(4)

        title = QLabel("Quality Control", card)
        title.setObjectName("HomeTitle")
        title.setFont(self._font(FONT_DISPLAY, 26, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel(
            "Real-time YOLO inspection for industrial devices — "
            "front panel, labels, and barcodes in one guided workflow.",
            card,
        )
        subtitle.setObjectName("HomeSubtitle")
        subtitle.setFont(self._font(FONT_UI, 11))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        layout.addSpacing(8)

        hint = QLabel("Choose an action to begin", card)
        hint.setObjectName("Muted")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)
        layout.addSpacing(6)

        for text, slot, primary, danger, icon in (
            ("Start Inspection", self._on_start, True, False, "▶"),
            ("Test Results", self._on_results, False, False, "▤"),
            ("Settings", self._on_settings, False, False, "⚙"),
            ("Exit", self._quit_application, False, True, "⏻"),
        ):
            btn = QPushButton(f"  {icon}   {text}", card)
            self._style_button(btn, primary=primary, danger=danger)
            btn.setFont(self._font(FONT_UI, 15, QFont.Weight.DemiBold))
            btn.clicked.connect(slot)
            layout.addWidget(btn)

        layout.addSpacing(4)
        shortcuts = QLabel("q — Home   ·   Esc — Exit", card)
        shortcuts.setObjectName("HomeShortcuts")
        shortcuts.setFont(self._font(FONT_UI, 11, QFont.Weight.DemiBold))
        shortcuts.setAlignment(Qt.AlignmentFlag.AlignCenter)
        shortcuts.setWordWrap(True)
        layout.addWidget(shortcuts)

        return card

    def _set_window_size(self, w: int, h: int) -> None:
        self.resize(w, h)
        self._center_on_screen()

    def _center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        x = geo.x() + max(0, (geo.width() - self.width()) // 2)
        y = geo.y() + max(0, (geo.height() - self.height()) // 2)
        self.move(x, y)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            if self._worker is not None and self._worker.isRunning():
                self._stop_worker(show_home=True)
                return
            self._quit_application()
            return

        if self._worker is not None and self._worker.isRunning():
            k = event.key()
            if k == Qt.Key.Key_Space:
                self._worker.post_action("space")
                return
            if k == Qt.Key.Key_R:
                self._worker.post_action("r")
                return
            if k == Qt.Key.Key_Q or event.text().lower() == "q":
                self._stop_worker(show_home=True)
                return
            if k == Qt.Key.Key_1:
                self._worker.post_action("1")
                return
            if k == Qt.Key.Key_2:
                self._worker.post_action("2")
                return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._hide_startup_dialog()
        self._stop_worker(show_home=False)
        event.accept()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _quit_application(self) -> None:
        self._hide_startup_dialog()
        self._clear_camera_shortcuts()
        self._stop_worker(show_home=False)
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _on_start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Start", "Camera is already running.")
            return
        if self._camera_page is None or self._stack is None:
            return

        self._show_startup_dialog("Loading model and camera…")
        self._set_window_size(*CAMERA_WINDOW_SIZE)
        self._stack.setCurrentWidget(self._camera_page)

        w = CameraWorker(self)
        self._worker = w
        w.frame_ready.connect(self._camera_page.set_frame)
        w.panel_ready.connect(self._camera_page.set_panel_lines)
        w.history_ready.connect(self._camera_page.set_history_rows)
        w.error.connect(self._on_worker_error)
        w.started_ok.connect(self._on_worker_started)

        self._camera_page.btn_reset.clicked.connect(lambda: w.post_action("r"))
        self._camera_page.btn_dev1.clicked.connect(lambda: w.post_action("1"))
        self._camera_page.btn_dev2.clicked.connect(lambda: w.post_action("2"))

        self._setup_camera_shortcuts(w)
        w.start()

    def _setup_camera_shortcuts(self, worker: CameraWorker) -> None:
        self._clear_camera_shortcuts()
        ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut

        def add(key: str, handler) -> None:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(ctx)
            sc.activated.connect(handler)
            self._camera_shortcuts.append(sc)

        add("Q", lambda: self._stop_worker(show_home=True))
        add("R", lambda: worker.post_action("r"))
        add("Space", lambda: worker.post_action("space"))
        add("1", lambda: worker.post_action("1"))
        add("2", lambda: worker.post_action("2"))

    def _clear_camera_shortcuts(self) -> None:
        for sc in self._camera_shortcuts:
            sc.setEnabled(False)
            sc.deleteLater()
        self._camera_shortcuts.clear()

    def _on_worker_started(self) -> None:
        self._hide_startup_dialog()
        self._center_on_screen()

    def _on_worker_error(self, msg: str) -> None:
        self._hide_startup_dialog()
        QMessageBox.critical(self, "Camera", msg)
        self._stop_worker(show_home=True)

    def _on_results(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._stop_worker(show_home=True)
        if self._stack is not None and self._home_widget is not None:
            self._stack.setCurrentWidget(self._home_widget)

        geo = self.geometry()
        self.hide()
        try:
            dlg = ResultsDialog(None, limit=200)
            dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
            dw = max(920, min(geo.width() + 80, 1280))
            dh = max(580, geo.height())
            dlg.resize(dw, dh)
            screen = QApplication.primaryScreen()
            if screen:
                sg = screen.availableGeometry()
                dlg.move(
                    sg.x() + max(0, (sg.width() - dlg.width()) // 2),
                    sg.y() + max(0, (sg.height() - dlg.height()) // 2),
                )
            dlg.exec()
        finally:
            self.show()
            self._set_window_size(*HOME_WINDOW_SIZE)

    def _on_settings(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self,
                "Settings",
                "Return Home to end the current inspection before changing settings.",
            )
            return
        dlg = SettingsDialog(self)
        dlg.exec()

    def _show_home(self) -> None:
        if self._stack is not None and self._home_widget is not None:
            self._stack.setCurrentWidget(self._home_widget)
        self._set_window_size(*HOME_WINDOW_SIZE)

    def _stop_worker(self, *, show_home: bool) -> None:
        self._clear_camera_shortcuts()
        w = self._worker
        self._worker = None
        if w is not None:
            try:
                w.request_stop()
                w.wait(3000)
            except Exception:
                pass
        if show_home:
            self._show_home()

    def _show_startup_dialog(self, message: str = "Loading…") -> None:
        if self._startup_dialog is None:
            dlg = QProgressDialog(message, None, 0, 0, self)
            dlg.setWindowTitle("Starting")
            dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
            dlg.setCancelButton(None)
            dlg.setMinimumDuration(0)
            dlg.setAutoClose(False)
            dlg.setAutoReset(False)
            self._startup_dialog = dlg
        else:
            self._startup_dialog.setLabelText(message)
        self._startup_dialog.show()

    def _hide_startup_dialog(self) -> None:
        if self._startup_dialog is not None:
            self._startup_dialog.hide()


class LauncherApplication:
    def __init__(self, argv: list[str]) -> None:
        self._app = QApplication(argv)
        self._app.setQuitOnLastWindowClosed(False)
        self._app.setStyle("Fusion")
        self._app.setFont(QFont(FONT_UI, 10))
        self._app.setStyleSheet(QSS)
        self._icon = AppIconFactory.load()
        self._app.setWindowIcon(self._icon)
        self._window = LauncherWindow(self._icon)

    def run(self) -> int:
        self._window.show()
        self._app.processEvents()
        return self._app.exec()


def main() -> int:
    return LauncherApplication(sys.argv).run()


if __name__ == "__main__":
    raise SystemExit(main())
