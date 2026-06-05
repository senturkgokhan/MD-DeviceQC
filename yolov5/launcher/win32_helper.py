"""Win32 window helpers for launcher ↔ OpenCV coordination."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
import sys

from PySide6.QtWidgets import QApplication, QWidget

QC_WIN_TITLE = "Kalite Kontrol"

_SW_RESTORE = 9
_SW_SHOW = 5
_HWND_TOPMOST = -1
_HWND_NOTOPMOST = -2
_SWP_NOMOVE = 0x0002
_SWP_NOSIZE = 0x0001
_SWP_SHOWWINDOW = 0x0040


class Win32Helper:
    QC_WIN_TITLE = QC_WIN_TITLE

    @staticmethod
    def _user32():
        return ctypes.windll.user32

    @staticmethod
    def find_window(title: str) -> int | None:
        if sys.platform != "win32":
            return None
        try:
            user32 = Win32Helper._user32()
            user32.FindWindowW.argtypes = [ctypes.wintypes.LPCWSTR, ctypes.wintypes.LPCWSTR]
            user32.FindWindowW.restype = ctypes.wintypes.HWND
            hwnd = user32.FindWindowW(None, title)
            return int(hwnd) if hwnd else None
        except Exception:
            return None

    @staticmethod
    def is_foreground_window(hwnd: int) -> bool:
        if sys.platform != "win32" or not hwnd:
            return False
        try:
            return int(Win32Helper._user32().GetForegroundWindow()) == int(hwnd)
        except Exception:
            return False

    @staticmethod
    def allow_foreground_process(pid: int) -> None:
        """Caller must be foreground; grants `pid` the right to call SetForegroundWindow."""
        if sys.platform != "win32" or pid <= 0:
            return
        try:
            Win32Helper._user32().AllowSetForegroundWindow(ctypes.c_uint(int(pid)))
        except Exception:
            pass

    @staticmethod
    def allow_parent_foreground() -> None:
        """Call from qc_engine on q while OpenCV still has focus."""
        if sys.platform != "win32":
            return
        try:
            Win32Helper.allow_foreground_process(os.getppid())
        except Exception:
            pass

    @staticmethod
    def center_hwnd(hwnd: int) -> None:
        if sys.platform != "win32":
            return
        try:
            user32 = Win32Helper._user32()
            rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(int(hwnd), ctypes.byref(rect))
            w, h = rect.right - rect.left, rect.bottom - rect.top
            sw = int(user32.GetSystemMetrics(0))
            sh = int(user32.GetSystemMetrics(1))
            user32.MoveWindow(int(hwnd), max(0, (sw - w) // 2), max(0, (sh - h) // 2), w, h, True)
        except Exception:
            pass

    @staticmethod
    def bring_to_foreground(hwnd: int) -> bool:
        if sys.platform != "win32" or not hwnd:
            return False
        hwnd = int(hwnd)
        user32 = Win32Helper._user32()
        kernel32 = ctypes.windll.kernel32

        try:
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, _SW_RESTORE)
            else:
                user32.ShowWindow(hwnd, _SW_SHOW)

            try:
                user32.AllowSetForegroundWindow(ctypes.c_uint(0xFFFFFFFF))
            except Exception:
                pass

            fg = int(user32.GetForegroundWindow())
            if fg and fg != hwnd:
                fg_tid = user32.GetWindowThreadProcessId(fg, None)
                target_tid = user32.GetWindowThreadProcessId(hwnd, None)
                this_thread_id = kernel32.GetCurrentThreadId()
                attached = False
                if fg_tid and target_tid:
                    attached = bool(user32.AttachThreadInput(fg_tid, target_tid, True))
                    user32.AttachThreadInput(this_thread_id, target_tid, True)
                try:
                    user32.SetForegroundWindow(hwnd)
                    user32.BringWindowToTop(hwnd)
                finally:
                    if attached:
                        user32.AttachThreadInput(fg_tid, target_tid, False)
                        user32.AttachThreadInput(this_thread_id, target_tid, False)
            else:
                user32.SetForegroundWindow(hwnd)
                user32.BringWindowToTop(hwnd)

            if int(user32.GetForegroundWindow()) != hwnd:
                flags = _SWP_NOMOVE | _SWP_NOSIZE | _SWP_SHOWWINDOW
                user32.SetWindowPos(hwnd, _HWND_TOPMOST, 0, 0, 0, 0, flags)
                user32.SetWindowPos(hwnd, _HWND_NOTOPMOST, 0, 0, 0, 0, flags)
                user32.SetForegroundWindow(hwnd)

            return int(user32.GetForegroundWindow()) == hwnd
        except Exception:
            return False

    @staticmethod
    def focus_hwnd(hwnd: int) -> None:
        Win32Helper.bring_to_foreground(hwnd)

    @staticmethod
    def bring_qt_window_to_front(widget: QWidget) -> bool:
        if sys.platform != "win32":
            widget.raise_()
            widget.activateWindow()
            return widget.isActiveWindow()
        try:
            hwnd = int(widget.winId())
        except Exception:
            hwnd = 0
        widget.raise_()
        widget.activateWindow()
        app = QApplication.instance()
        if app is not None:
            app.setActiveWindow(widget)
        if hwnd:
            return Win32Helper.bring_to_foreground(hwnd)
        return widget.isActiveWindow()

    @staticmethod
    def focus_qt_window(widget: QWidget) -> None:
        Win32Helper.bring_qt_window_to_front(widget)

    @staticmethod
    def post_close(hwnd: int) -> None:
        if sys.platform != "win32":
            return
        try:
            Win32Helper._user32().PostMessageW(int(hwnd), 0x0010, 0, 0)
        except Exception:
            pass
