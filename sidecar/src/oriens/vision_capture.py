"""仅捕获已验证游戏窗口客户区的 Windows 边界。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol

from PySide6.QtGui import QImage


class CaptureError(RuntimeError):
    """可安全展示的捕获错误，不包含句柄、PID、路径或底层异常。"""


@dataclass(frozen=True, slots=True)
class WindowInfo:
    handle: int
    process_name: str
    title: str
    visible: bool
    minimized: bool
    foreground: bool
    client_width: int
    client_height: int
    screen_x: int = 0
    screen_y: int = 0
    dpi: int = 96


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    image: QImage
    client_width: int
    client_height: int


class WindowLocator(Protocol):
    def locate(self) -> WindowInfo: ...


class ClientFrameBackend(Protocol):
    def capture(self, window: WindowInfo) -> QImage: ...


_ALLOWED_PROCESS_NAMES = frozenset({"isaac-ng.exe"})
_REQUIRED_TITLE_FRAGMENT = "binding of isaac"


def validate_game_window(
    window: WindowInfo, *, require_foreground: bool = True
) -> None:
    """拒绝任何无法同时通过进程、标题和客户区验证的窗口。"""

    if window.process_name.casefold() not in _ALLOWED_PROCESS_NAMES:
        raise CaptureError("未找到可确认的《以撒的结合》游戏窗口。")
    if _REQUIRED_TITLE_FRAGMENT not in window.title.casefold():
        raise CaptureError("未找到可确认的《以撒的结合》游戏窗口。")
    if not window.visible:
        raise CaptureError("游戏窗口当前不可见，无法安全识别。")
    if window.minimized:
        raise CaptureError("请先恢复游戏窗口，再识别当前画面。")
    if require_foreground and not window.foreground:
        raise CaptureError("请先切回游戏窗口，再识别当前画面。")
    if not 64 <= window.client_width <= 16384 or not 64 <= window.client_height <= 16384:
        raise CaptureError("游戏窗口客户区尺寸异常，已停止捕获。")


class GameWindowCapture:
    """组合可注入定位器和客户区捕获器；从不提供桌面回退。"""

    def __init__(
        self,
        locator: WindowLocator,
        backend: ClientFrameBackend,
        *,
        require_foreground: bool = True,
    ) -> None:
        self._locator = locator
        self._backend = backend
        self._require_foreground = require_foreground

    def capture(self) -> CapturedFrame:
        window = self._locator.locate()
        validate_game_window(window, require_foreground=self._require_foreground)
        image = self._backend.capture(window)
        if image.isNull():
            raise CaptureError("游戏画面捕获失败，请稍后重试。")
        if image.width() != window.client_width or image.height() != window.client_height:
            raise CaptureError("捕获结果尺寸异常，已停止识别。")
        return CapturedFrame(image, window.client_width, window.client_height)


class WindowsGameWindowLocator:
    """通过 Win32 枚举并验证 Isaac 客户窗口。"""

    def locate(self) -> WindowInfo:
        if os.name != "nt":
            raise CaptureError("当前系统不支持安全的游戏窗口捕获。")
        try:
            return self._locate_windows()
        except CaptureError:
            raise
        except Exception:
            raise CaptureError("游戏窗口定位失败，请稍后重试。") from None

    @staticmethod
    def _locate_windows() -> WindowInfo:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [
            wintypes.HWND, wintypes.LPWSTR, ctypes.c_int
        ]
        user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND, ctypes.POINTER(wintypes.DWORD)
        ]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        user32.GetClientRect.argtypes = [
            wintypes.HWND, ctypes.POINTER(wintypes.RECT)
        ]
        user32.GetClientRect.restype = wintypes.BOOL
        user32.ClientToScreen.argtypes = [
            wintypes.HWND, ctypes.POINTER(wintypes.POINT)
        ]
        user32.ClientToScreen.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        found: list[WindowInfo] = []
        foreground = int(user32.GetForegroundWindow())

        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @callback_type
        def visit(hwnd, _lparam):
            title_length = int(user32.GetWindowTextLengthW(hwnd))
            if title_length <= 0:
                return True
            title_buffer = ctypes.create_unicode_buffer(title_length + 1)
            user32.GetWindowTextW(hwnd, title_buffer, len(title_buffer))
            title = title_buffer.value
            if _REQUIRED_TITLE_FRAGMENT not in title.casefold():
                return True

            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            process = kernel32.OpenProcess(0x1000, False, pid.value)
            if not process:
                return True
            try:
                size = wintypes.DWORD(32768)
                path_buffer = ctypes.create_unicode_buffer(size.value)
                if not kernel32.QueryFullProcessImageNameW(
                    process, 0, path_buffer, ctypes.byref(size)
                ):
                    return True
                process_name = Path(path_buffer.value).name
            finally:
                kernel32.CloseHandle(process)
            if process_name.casefold() not in _ALLOWED_PROCESS_NAMES:
                return True

            rect = wintypes.RECT()
            if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
                return True
            origin = wintypes.POINT(0, 0)
            user32.ClientToScreen(hwnd, ctypes.byref(origin))
            dpi = int(user32.GetDpiForWindow(hwnd)) if hasattr(user32, "GetDpiForWindow") else 96
            found.append(WindowInfo(
                handle=int(hwnd),
                process_name=process_name,
                title=title,
                visible=bool(user32.IsWindowVisible(hwnd)),
                minimized=bool(user32.IsIconic(hwnd)),
                foreground=int(hwnd) == foreground,
                client_width=int(rect.right - rect.left),
                client_height=int(rect.bottom - rect.top),
                screen_x=int(origin.x),
                screen_y=int(origin.y),
                dpi=dpi or 96,
            ))
            return True

        user32.EnumWindows(visit, 0)
        if not found:
            raise CaptureError("未找到可确认的《以撒的结合》游戏窗口。")
        found.sort(key=lambda item: (not item.foreground, item.minimized, not item.visible))
        return found[0]


class WindowsClientDCBackend:
    """从目标窗口客户 DC 复制固定客户区，不读取桌面或相邻窗口。"""

    def capture(self, window: WindowInfo) -> QImage:
        if os.name != "nt":
            raise CaptureError("当前系统不支持安全的游戏窗口捕获。")
        try:
            return self._capture_windows(window)
        except CaptureError:
            raise
        except Exception:
            raise CaptureError("游戏画面捕获失败，请稍后重试。") from None

    @staticmethod
    def _capture_windows(window: WindowInfo) -> QImage:
        import ctypes
        from ctypes import wintypes

        class BitmapInfoHeader(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        class BitmapInfo(ctypes.Structure):
            _fields_ = [("bmiHeader", BitmapInfoHeader), ("bmiColors", wintypes.DWORD * 3)]

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        user32.GetDC.argtypes = [wintypes.HWND]
        user32.GetDC.restype = wintypes.HDC
        user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        user32.ReleaseDC.restype = ctypes.c_int
        gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
        gdi32.CreateCompatibleDC.restype = wintypes.HDC
        gdi32.CreateDIBSection.argtypes = [
            wintypes.HDC,
            ctypes.POINTER(BitmapInfo),
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        gdi32.CreateDIBSection.restype = wintypes.HBITMAP
        gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        gdi32.SelectObject.restype = wintypes.HGDIOBJ
        gdi32.BitBlt.argtypes = [
            wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD,
        ]
        gdi32.BitBlt.restype = wintypes.BOOL
        gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        gdi32.DeleteObject.restype = wintypes.BOOL
        gdi32.DeleteDC.argtypes = [wintypes.HDC]
        gdi32.DeleteDC.restype = wintypes.BOOL
        width, height = window.client_width, window.client_height
        client_dc = user32.GetDC(window.handle)
        memory_dc = gdi32.CreateCompatibleDC(client_dc)
        bits = ctypes.c_void_p()
        info = BitmapInfo()
        info.bmiHeader.biSize = ctypes.sizeof(BitmapInfoHeader)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = -height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = 0
        bitmap = gdi32.CreateDIBSection(
            client_dc, ctypes.byref(info), 0, ctypes.byref(bits), None, 0
        )
        if not client_dc or not memory_dc or not bitmap or not bits.value:
            if bitmap:
                gdi32.DeleteObject(bitmap)
            if memory_dc:
                gdi32.DeleteDC(memory_dc)
            if client_dc:
                user32.ReleaseDC(window.handle, client_dc)
            raise CaptureError("游戏画面捕获失败，请稍后重试。")
        previous = gdi32.SelectObject(memory_dc, bitmap)
        try:
            # SRCCOPY | CAPTUREBLT；源 DC 已限定为目标客户区，不存在桌面回退。
            if not gdi32.BitBlt(
                memory_dc, 0, 0, width, height, client_dc, 0, 0, 0x40CC0020
            ):
                raise CaptureError("游戏画面捕获失败，请稍后重试。")
            raw = ctypes.string_at(bits.value, width * height * 4)
            return QImage(
                raw, width, height, width * 4, QImage.Format.Format_ARGB32
            ).copy()
        finally:
            gdi32.SelectObject(memory_dc, previous)
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(memory_dc)
            user32.ReleaseDC(window.handle, client_dc)
