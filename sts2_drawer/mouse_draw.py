from __future__ import annotations

import json
import time
import ctypes
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

from .image_trace import Stroke


user32 = ctypes.windll.user32
VK_ESCAPE = 0x1B
VK_F1 = 0x70
VK_F3 = 0x72
VK_F4 = 0x73
VK_F5 = 0x74
VK_F9 = 0x78
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
GA_ROOT = 2
SW_RESTORE = 9
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


user32.WindowFromPoint.argtypes = [POINT]
user32.WindowFromPoint.restype = ctypes.c_void_p
user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
user32.GetAncestor.restype = ctypes.c_void_p
user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
user32.ShowWindow.restype = ctypes.c_bool
user32.BringWindowToTop.argtypes = [ctypes.c_void_p]
user32.BringWindowToTop.restype = ctypes.c_bool
user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
user32.SetForegroundWindow.restype = ctypes.c_bool
user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
user32.GetWindowLongW.restype = ctypes.c_long
user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
user32.SetWindowLongW.restype = ctypes.c_long


@dataclass(frozen=True)
class DrawArea:
    left: int
    top: int
    width: int
    height: int

    @classmethod
    def from_dict(cls, data: dict) -> "DrawArea":
        return cls(
            left=int(data["left"]),
            top=int(data["top"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )

    def to_screen(self, x: int, y: int, canvas_size: Tuple[int, int]) -> Tuple[int, int]:
        canvas_width, canvas_height = canvas_size
        screen_x = self.left + round((x / max(canvas_width - 1, 1)) * self.width)
        screen_y = self.top + round((y / max(canvas_height - 1, 1)) * self.height)
        return screen_x, screen_y


def enable_dpi_awareness() -> None:
    """Keep Tk overlay coordinates aligned with physical mouse coordinates."""

    try:
        shcore = ctypes.windll.shcore
        shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


def load_draw_area(config_path: Path) -> DrawArea:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return DrawArea.from_dict(data["draw_area"])


def save_draw_area(config_path: Path, area: DrawArea) -> None:
    config_path.write_text(
        json.dumps({"draw_area": asdict(area)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def focus_window_under_area(area: DrawArea) -> bool:
    """Focus the top-level window under the center of the calibrated area."""

    point = POINT(area.left + area.width // 2, area.top + area.height // 2)
    window = user32.WindowFromPoint(point)
    if not window:
        return False

    root_window = user32.GetAncestor(window, GA_ROOT) or window
    user32.ShowWindow(root_window, SW_RESTORE)
    user32.BringWindowToTop(root_window)
    return bool(user32.SetForegroundWindow(root_window))


def set_window_click_through(hwnd: int, enabled: bool) -> None:
    """Let mouse clicks pass through an overlay window when enabled."""

    root_window = user32.GetAncestor(hwnd, GA_ROOT) or hwnd
    style = user32.GetWindowLongW(root_window, GWL_EXSTYLE)
    if enabled:
        style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
    else:
        style &= ~WS_EX_TRANSPARENT
        style |= WS_EX_LAYERED
    user32.SetWindowLongW(root_window, GWL_EXSTYLE, style)


def activate_window(hwnd: int) -> None:
    root_window = user32.GetAncestor(hwnd, GA_ROOT) or hwnd
    user32.ShowWindow(root_window, SW_RESTORE)
    user32.BringWindowToTop(root_window)
    user32.SetForegroundWindow(root_window)


def calibrate(config_path: Path) -> None:
    input("Move the mouse to the TOP-LEFT corner of the drawable area, then press Enter...")
    left, top = get_cursor_position()
    input("Move the mouse to the BOTTOM-RIGHT corner of the drawable area, then press Enter...")
    right, bottom = get_cursor_position()

    area = DrawArea(
        left=min(left, right),
        top=min(top, bottom),
        width=abs(right - left),
        height=abs(bottom - top),
    )
    if area.width <= 0 or area.height <= 0:
        raise RuntimeError("Calibration failed: selected area has no width or height.")

    save_draw_area(config_path, area)


def get_cursor_position() -> Tuple[int, int]:
    point = POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def set_cursor_position(x: int, y: int) -> None:
    user32.SetCursorPos(int(x), int(y))


def is_escape_pressed() -> bool:
    return bool(user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)


def is_f1_pressed() -> bool:
    return bool(user32.GetAsyncKeyState(VK_F1) & 0x8000)


def is_f3_pressed() -> bool:
    return bool(user32.GetAsyncKeyState(VK_F3) & 0x8000)


def is_f4_pressed() -> bool:
    return bool(user32.GetAsyncKeyState(VK_F4) & 0x8000)


def is_f5_pressed() -> bool:
    return bool(user32.GetAsyncKeyState(VK_F5) & 0x8000)


def is_f9_pressed() -> bool:
    return bool(user32.GetAsyncKeyState(VK_F9) & 0x8000)


def mouse_down(button: str) -> None:
    down, _ = button_events(button)
    user32.mouse_event(down, 0, 0, 0, 0)


def mouse_up(button: str) -> None:
    _, up = button_events(button)
    user32.mouse_event(up, 0, 0, 0, 0)


def button_events(button: str) -> Tuple[int, int]:
    if button == "left":
        return MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP
    if button == "right":
        return MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP
    if button == "middle":
        return MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP
    raise ValueError(f"Unsupported mouse button: {button}")


class PauseController:
    def __init__(self, on_pause_change: Optional[Callable[[bool], None]] = None) -> None:
        self.paused = False
        self.f5_was_pressed = is_f5_pressed()
        self.f5_down_since: Optional[float] = time.time() if self.f5_was_pressed else None
        self.on_pause_change = on_pause_change

    def tick(self, allow_pause: bool = True) -> None:
        self.check_abort()
        self.update_toggle()
        if allow_pause:
            self.wait_if_paused()

    @staticmethod
    def check_abort() -> None:
        if is_escape_pressed() or is_f9_pressed():
            raise KeyboardInterrupt("Drawing aborted.")

    def update_toggle(self) -> None:
        f5_pressed = is_f5_pressed()
        now = time.time()
        if f5_pressed and not self.f5_was_pressed:
            self.f5_down_since = now
        if f5_pressed and self.f5_down_since is not None and now - self.f5_down_since >= 0.85:
            raise KeyboardInterrupt("Drawing stopped by F5.")
        if not f5_pressed and self.f5_was_pressed:
            self.f5_down_since = None
            self.paused = not self.paused
            if self.on_pause_change is not None:
                self.on_pause_change(self.paused)
            if self.paused:
                print("Drawing paused. Press F5 to continue, or hold F5 to stop.")
            else:
                print("Drawing resumed.")
        self.f5_was_pressed = f5_pressed

    def wait_if_paused(self) -> None:
        while self.paused:
            self.check_abort()
            self.update_toggle()
            time.sleep(0.05)


def sleep_with_controls(duration: float, pause_controller: PauseController) -> None:
    end_time = time.time() + duration
    while time.time() < end_time:
        pause_controller.tick()
        time.sleep(min(0.05, max(0.0, end_time - time.time())))


def move_smoothly(
    start: Tuple[int, int],
    end: Tuple[int, int],
    duration: float,
    pause_controller: Optional[PauseController] = None,
    allow_pause: bool = True,
) -> None:
    if duration <= 0:
        if pause_controller is not None:
            pause_controller.tick(allow_pause=allow_pause)
        set_cursor_position(*end)
        return

    steps = max(1, int(duration / 0.005))
    start_x, start_y = start
    end_x, end_y = end
    for step in range(1, steps + 1):
        if pause_controller is not None:
            pause_controller.tick(allow_pause=allow_pause)
        elif is_escape_pressed() or is_f9_pressed():
            raise KeyboardInterrupt("Drawing aborted.")
        progress = step / steps
        x = round(start_x + (end_x - start_x) * progress)
        y = round(start_y + (end_y - start_y) * progress)
        set_cursor_position(x, y)
        time.sleep(duration / steps)


def draw_strokes(
    strokes: Iterable[Stroke],
    area: DrawArea,
    canvas_size: Tuple[int, int],
    countdown: int,
    move_duration: float,
    stroke_pause: float,
    button: str,
    jump_duration: float = 0.0,
    on_pause_change: Optional[Callable[[bool], None]] = None,
) -> None:
    for remaining in range(countdown, 0, -1):
        print(f"Drawing starts in {remaining}... Hold F5 or press Esc/F9/Ctrl+C to abort.")
        time.sleep(1)

    pause_controller = PauseController(on_pause_change=on_pause_change)
    print("Press F5 to pause/resume drawing. Hold F5, Esc, or F9 to stop.")

    for index, stroke in enumerate(strokes, start=1):
        pause_controller.tick()

        start = area.to_screen(stroke.x1, stroke.y1, canvas_size)
        end = area.to_screen(stroke.x2, stroke.y2, canvas_size)
        current = get_cursor_position()

        move_smoothly(current, start, jump_duration, pause_controller)
        pause_controller.tick()
        mouse_down(button)
        try:
            move_smoothly(start, end, move_duration, pause_controller, allow_pause=False)
        finally:
            mouse_up(button)

        pause_controller.tick()
        if stroke_pause > 0:
            sleep_with_controls(stroke_pause, pause_controller)
        if index % 100 == 0:
            print(f"Drew {index} strokes...")
