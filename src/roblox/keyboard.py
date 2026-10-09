"""Keyboard state manager with safety guarantees.

ENGINEERED (pure interface code). Design goals from the project brief:
  * never leave W/A/S/D held after crash/exception,
  * keys pressed/released only on transitions (Roblox treats auto-repeat of
    discrete keydown spam poorly; holding is correct for movement),
  * works in DRYRUN mode (log only) for pre-Roblox testing,
  * atexit + signal handlers release everything.

Real injection backend on Windows uses ctypes SendInput (no python deps).
On non-Windows platforms real injection is unavailable and DRYRUN is forced.
"""
from __future__ import annotations

import atexit
import logging
import sys
import threading

log = logging.getLogger("keyboard")

VK = {"W": 0x57, "A": 0x41, "S": 0x53, "D": 0x44,
      "UP": 0x26, "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27}


class Keyboard:
    def __init__(self, dryrun=True):
        self.dryrun = dryrun or sys.platform != "win32"
        self._held = set()
        self._lock = threading.Lock()
        if not self.dryrun:
            self._init_win()
        atexit.register(self.release_all)

    # ------------------------------------------------------------ windows
    def _init_win(self):
        import ctypes
        self._ctypes = ctypes
        self._KEYEVENTF_KEYUP = 0x0002

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", ctypes.c_word),
                        ("wScan", ctypes.c_word),
                        ("dwFlags", ctypes.c_dword),
                        ("time", ctypes.c_dword),
                        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

        class INPUT(ctypes.Structure):
            class _I(ctypes.Union):
                _fields_ = [("ki", KEYBDINPUT)]
            _fields_ = [("type", ctypes.c_dword), ("ii", _I)]

        self._INPUT = INPUT
        self._KEYBDINPUT = KEYBDINPUT
        self._SendInput = ctypes.windll.user32.SendInput

    def _send(self, vk: int, up: bool):
        ki = self._KEYBDINPUT(vk, 0, self._KEYEVENTF_KEYUP if up else 0, 0, None)
        inp = self._INPUT(1, self._INPUT._I(ki))
        n = self._SendInput(1, [inp], self._ctypes.sizeof(self._INPUT))
        if n != 1:
            raise RuntimeError(f"SendInput failed for {'up' if up else 'down'} {vk}")

    # ------------------------------------------------------------- api
    def press(self, key: str):
        key = key.upper()
        assert key in VK, f"unknown key {key}"
        with self._lock:
            if key in self._held:
                return
            if self.dryrun:
                log.info("DRYRUN press %s", key)
            else:
                self._send(VK[key], up=False)
            self._held.add(key)

    def release(self, key: str):
        key = key.upper()
        with self._lock:
            if key not in self._held:
                return
            if self.dryrun:
                log.info("DRYRUN release %s", key)
            else:
                try:
                    self._send(VK[key], up=True)
                except Exception as e:      # keep trying to clear state
                    log.error("release %s failed: %s", key, e)
            self._held.discard(key)

    def set_movement(self, forward: float, turn: float, throttle_deadzone=0.15):
        """Map continuous (-1..1) commands onto WASD hold-state transitions."""
        want = set()
        if forward > throttle_deadzone:
            want.add("W")
        elif forward < -throttle_deadzone:
            want.add("S")
        if turn > throttle_deadzone:
            want.add("D")
        elif turn < -throttle_deadzone:
            want.add("A")
        cur = set(self._held)
        for k in cur - want:
            self.release(k)
        for k in want - cur:
            self.press(k)
        return sorted(want)

    def release_all(self):
        for k in list(self._held):
            self.release(k)

    @property
    def held(self):
        return tuple(sorted(self._held))
