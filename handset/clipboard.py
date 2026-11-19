"""Host clipboard access for the PC side of the bridge.

Windows keeps using the Win32 clipboard API. macOS uses ``pbpaste``/``pbcopy``
for the text and ``NSPasteboard.changeCount`` for change detection, so the
"did the user copy something new?" check keeps working on both platforms.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT = 2.0  # seconds


def _is_darwin() -> bool:
    return sys.platform == "darwin"


def _is_windows() -> bool:
    return sys.platform == "win32"


# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------


def _darwin_pbpaste() -> Optional[str]:
    try:
        result = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("pbpaste failed: %s", exc)
        return None
    if result.returncode != 0:
        logger.warning("pbpaste exited %s: %s", result.returncode, result.stderr.decode("utf-8", "ignore").strip())
        return None
    return result.stdout.decode("utf-8", "replace")


def _darwin_pbcopy(text: str) -> bool:
    try:
        result = subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("pbcopy failed: %s", exc)
        return False
    if result.returncode != 0:
        logger.warning("pbcopy exited %s: %s", result.returncode, result.stderr.decode("utf-8", "ignore").strip())
        return False
    return True


def _darwin_clipboard_change_count() -> Optional[int]:
    """Return ``NSPasteboard.changeCount``, or ``None`` when pyobjc is absent.

    The count increments on every copy, even when the copied text is
    identical -- which is exactly what the "copied something new" check
    needs, and what a content hash cannot tell us.
    """
    try:
        import AppKit
    except Exception:
        return None
    try:
        pasteboard = AppKit.NSPasteboard.generalPasteboard()
        return int(pasteboard.changeCount())
    except Exception as exc:
        logger.debug("NSPasteboard.changeCount unavailable: %s", exc)
        return None


def _darwin_fallback_seq() -> int:
    """Content-hash fallback for the sequence number when pyobjc is missing."""
    import hashlib

    text = _darwin_pbpaste() or ""
    digest = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def _windows_clipboard_seq() -> int:
    try:
        import ctypes

        return int(ctypes.windll.user32.GetClipboardSequenceNumber())
    except Exception:
        return 0


def _windows_clipboard_text() -> Optional[str]:
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        user32.GetClipboardData.argtypes = [wintypes.UINT]
        user32.GetClipboardData.restype = wintypes.HANDLE
        user32.CloseClipboard.argtypes = []
        user32.CloseClipboard.restype = wintypes.BOOL
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalUnlock.restype = wintypes.BOOL

        import time

        opened = False
        for _ in range(5):
            if user32.OpenClipboard(0):
                opened = True
                break
            time.sleep(0.015)
        if not opened:
            return None
        try:
            CF_UNICODETEXT = 13
            h_clip = user32.GetClipboardData(CF_UNICODETEXT)
            if not h_clip:
                return None
            p_text = kernel32.GlobalLock(h_clip)
            if not p_text:
                return None
            try:
                return ctypes.c_wchar_p(p_text).value
            finally:
                kernel32.GlobalUnlock(h_clip)
        finally:
            user32.CloseClipboard()
    except Exception as exc:
        logger.warning("Windows clipboard read failed: %s", exc)
        return None


def _windows_set_clipboard_text(text: str) -> bool:
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        user32.EmptyClipboard.argtypes = []
        user32.EmptyClipboard.restype = wintypes.BOOL
        user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        user32.SetClipboardData.restype = wintypes.HANDLE
        user32.CloseClipboard.argtypes = []
        user32.CloseClipboard.restype = wintypes.BOOL

        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = wintypes.LPVOID
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalUnlock.restype = wintypes.BOOL
        kernel32.RtlMoveMemory.argtypes = [wintypes.LPVOID, wintypes.LPCVOID, ctypes.c_size_t]
        kernel32.RtlMoveMemory.restype = None

        import time

        opened = False
        for _ in range(5):
            if user32.OpenClipboard(0):
                opened = True
                break
            time.sleep(0.015)
        if not opened:
            return False
        try:
            if not user32.EmptyClipboard():
                return False

            encoded = text.encode("utf-16le") + b"\x00\x00"
            GMEM_MOVEABLE = 0x0002
            CF_UNICODETEXT = 13

            h_mem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(encoded))
            if not h_mem:
                return False

            p_mem = kernel32.GlobalLock(h_mem)
            if not p_mem:
                return False
            try:
                kernel32.RtlMoveMemory(p_mem, encoded, len(encoded))
            finally:
                kernel32.GlobalUnlock(h_mem)

            if not user32.SetClipboardData(CF_UNICODETEXT, h_mem):
                return False
            return True
        finally:
            user32.CloseClipboard()
    except Exception as exc:
        logger.warning("Windows clipboard write failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_clipboard_seq() -> int:
    """Return a number that changes whenever the host clipboard changes."""
    if _is_darwin():
        count = _darwin_clipboard_change_count()
        if count is not None:
            return count
        return _darwin_fallback_seq()
    if _is_windows():
        return _windows_clipboard_seq()
    return 0


def get_clipboard_text() -> Optional[str]:
    """Return the host clipboard's plain text, or ``None`` when unavailable."""
    if _is_darwin():
        text = _darwin_pbpaste()
        if text is None:
            return None
        return text
    if _is_windows():
        return _windows_clipboard_text()
    return None


def set_clipboard_text(text: str) -> bool:
    """Write ``text`` to the host clipboard. Returns True on success."""
    if not text:
        return False
    if _is_darwin():
        return _darwin_pbcopy(text)
    if _is_windows():
        return _windows_set_clipboard_text(text)
    return False
