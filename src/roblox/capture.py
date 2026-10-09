"""Windows Roblox window capture (ENGINEERED interface; no exploits).

Uses only public OS APIs:
  * FindWindow / EnumWindows + GetWindowText to locate 'Roblox' windows
  * GetWindowRect for geometry, SetWindowPos allowed? NO — we never move the
    game window.
  * BitBlt from the window DC via PrintWindow with PW_RENDERFULLCONTENT flag
    (works on DirectX/ANGLE windows such as Roblox in most configs); falls
    back to screen-region BitBlt of the window rect if PrintWindow yields a
    blank frame (occluded/minimised detection).
All GDI calls go through ctypes — zero third-party deps at runtime.

On non-Windows this module imports but raises CaptureUnavailable so the
runtime can run headless synthetic tests instead.
"""
from __future__ import annotations

import sys

import numpy as np


class CaptureError(RuntimeError):
    pass


class CaptureUnavailable(CaptureError):
    pass


def _load_gdi():
    if sys.platform != "win32":
        raise CaptureUnavailable("screen capture requires Windows")
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    SRCCOPY = 0x00CC0020
    PW_CLIENTONLY = 0x1
    PW_RENDERFULLCONTENT = 0x2
    CAPTUREBLT = 0x40000000

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    return ctypes, wintypes, user32, gdi32, SRCCOPY, PW_CLIENTONLY | PW_RENDERFULLCONTENT, CAPTUREBLT, RECT


class RobloxCapture:
    def __init__(self, title_contains=("roblox",), min_area=100_000):
        self.title_contains = [t.lower() for t in title_contains]
        self.min_area = min_area
        self.hwnd = None
        self._api = None

    # ------------------------------------------------------------ find window
    def find_window(self):
        ctypes, wintypes, user32, gdi32, *_ , RECT = _load_gdi()
        found = []

        EnumWindowsProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

        def cb(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value.lower()
            if any(t in title for t in self.title_contains):
                r = RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(r))
                area = (r.right - r.left) * (r.bottom - r.top)
                if area >= self.min_area:
                    found.append((area, hwnd, buf.value))
            return True

        user32.EnumWindows(EnumWindowsProc(cb), 0)
        if not found:
            raise CaptureError("no visible Roblox window found "
                               f"(searched titles containing {self.title_contains})")
        found.sort(reverse=True)
        self.hwnd = found[0][1]
        self.title = found[0][2]
        return self.hwnd

    # ------------------------------------------------------------ grab frame
    def grab(self) -> np.ndarray:
        """Return BGR->RGB uint8 array (H,W,3) of the window client area."""
        ctypes, wintypes, user32, gdi32, SRCCOPY, PW_FLAGS, CAPTUREBLT, RECT = _load_gdi()
        if self.hwnd is None or not user32.IsWindow(self.hwnd):
            self.find_window()
        if not user32.IsWindowVisible(self.hwnd):
            raise CaptureError("Roblox window exists but is not visible (minimised?)")

        r = RECT()
        cr = RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(cr))
        W, H = cr.right, cr.bottom
        if W < 50 or H < 50:
            raise CaptureError(f"suspicious client size {W}x{H}")

        # DIB section for CPU-readable pixels
        hwindc = user32.GetWindowDC(self.hwnd)
        try:
            memdc = gdi32.CreateCompatibleDC(hwindc)
            bmp = gdi32.CreateCompatibleBitmap(hwindc, W, H)
            old = gdi32.SelectObject(memdc, bmp)
            ok = user32.PrintWindow(self.hwnd, memdc, PW_FLAGS)
            if not ok:
                # fallback: bitblt from screen region
                user32.GetWindowRect(self.hwnd, ctypes.byref(r))
                sx, sy = r.left, r.top
                # approximate client offset
                pt = wintypes.POINT(0, 0)
                user32.ClientToScreen(self.hwnd, ctypes.byref(pt))
                sx, sy = pt.x, pt.y
                screen = user32.GetDC(0)
                gdi32.BitBlt(memdc, 0, 0, W, H, screen, sx, sy, SRCCOPY | CAPTUREBLT)
                user32.ReleaseDC(0, screen)

            class BIHDR(ctypes.Structure):
                _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long),
                            ("biHeight", ctypes.c_long), ("biPlanes", wintypes.WORD),
                            ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                            ("biSizeImage", wintypes.DWORD), ("xp", wintypes.LONG),
                            ("yp", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                            ("biClrImportant", wintypes.DWORD)]

            hdr = BIHDR(40, W, -H, 1, 32, 0, 0, 0, 0, 0, 0)
            buf = (ctypes.c_ubyte * (W * H * 4))()
            got = gdi32.GetDIBits(memdc, bmp, 0, H, buf, ctypes.byref(hdr), 0)
            gdi32.SelectObject(memdc, old)
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(memdc)
            if got == 0:
                raise CaptureError("GetDIBits returned 0 rows")
        finally:
            user32.ReleaseDC(self.hwnd, hwindc)

        arr = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
        rgb = arr[:, :, ::-1].copy()   # BGRA -> RGB
        if rgb.mean() < 1.0 and rgb.max() < 4:
            raise CaptureError("captured frame is black — window occluded or "
                               "using exclusive fullscreen; bring it to front")
        return rgb

    def is_valid(self) -> bool:
        if self._api is None:
            try:
                self._api = _load_gdi()
            except CaptureUnavailable:
                return False
        _, _, user32, *_ = self._api
        return self.hwnd is not None and user32.IsWindow(self.hwnd) \
            and user32.IsWindowVisible(self.hwnd)
