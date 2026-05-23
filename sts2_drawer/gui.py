from __future__ import annotations

import ctypes
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

from .image_trace import (
    Stroke,
    analyze_image_parameters,
    load_luminance_canvas,
    load_mask,
    trace_handdrawn_strokes,
    trace_line_reduction_strokes,
    trace_strokes,
)
from .mouse_draw import (
    DrawArea,
    activate_window,
    draw_strokes,
    enable_dpi_awareness,
    focus_window_under_area,
    is_f1_pressed,
    is_f3_pressed,
    is_f4_pressed,
    load_draw_area,
    save_draw_area,
    set_window_click_through,
)


APP_TITLE = "杀戮尖塔 2 地图绘图工具"
GUI_MUTEX_NAME = "Local\\STS2MapDrawerGuiSingleInstance"
MAX_PREVIEW_LINES = 3500
TRANSPARENT_COLOR = "#010203"
ASSET_DIR = Path(__file__).resolve().parent / "assets"
APP_ICON_IMAGE_PATH = ASSET_DIR / "app_icon.png"
APP_ICON_TITLE_PATH = ASSET_DIR / "app_icon_28.png"
APP_ICON_ICO_PATH = ASSET_DIR / "app_icon.ico"
PANEL_BG = "#eaf2f8"
PANEL_RAIL_BG = "#dbe8f1"
PANEL_BORDER = "#adc1d0"
SURFACE_BG = "#f7fbfd"
TEXT_COLOR = "#1b2a3a"
MUTED_TEXT = "#52677a"
WARNING_TEXT = "#8a5a2b"
ACCENT = "#5f7f9f"
ACCENT_DARK = "#355d7e"
ACCENT_SOFT = "#d6e6f1"
BUTTON_BG = "#edf5fa"
BUTTON_ACTIVE_BG = "#d4e4ef"
CHECK_BG = PANEL_BG
PREVIEW_COLOR = "#d17742"
AREA_COLOR = "#5aa7bd"
DEFAULT_MOVE_DURATION = 0.006
FAST_MOVE_DURATION = 0.004
DRAG_EDGE_PADDING = 10
MODE_LABELS = {
    "线条描绘": "lines",
    "手绘素描": "handdrawn",
    "横向填充": "scanline",
}


MODE_REVERSE_LABELS = {value: key for key, value in MODE_LABELS.items()}
PARAMETER_GUIDE_TEXT = (
    "参数说明\n"
    "阈值：决定多暗的像素会被当成要绘制的内容。提高会保留更多线和暗部，降低会删掉弱线和脏点。\n"
    "采样间距：决定取线密度。调小会更细、更慢；调大更简化、更快。\n"
    "最短线段：过滤太短的碎线。调大可以去毛刺，调小能保留小细节。\n"
    "绘制秒数：单笔拖动时长。越小越快，但过小可能丢笔。\n"
    "定位秒数：抬笔后移动到下一笔起点的耗时。一般保持 0，只有丢定位时再提高。\n"
    "停顿秒数：每笔结束后的额外停顿。只有游戏漏识别时再加。\n"
    "线条上限：允许生成的最大笔触数。提高会增加细节，也会增加总绘制时间。\n"
    "反相：把亮部当成主体，适合深色背景浅色图。\n"
    "线条描绘：偏结构线和排线，适合标志、表情和清晰边缘。\n"
    "手绘素描：偏轮廓曲线和内部结构，适合角色、头像、较复杂图像。\n"
    "横向填充：偏块面填充，适合大面积深色区域。"
)


class OverlayApp:
    def __init__(self) -> None:
        enable_dpi_awareness()
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 1.0)
        self.root.configure(bg=TRANSPARENT_COLOR)
        try:
            self.root.attributes("-transparentcolor", TRANSPARENT_COLOR)
        except tk.TclError:
            pass
        self.window_icon_image = self.load_tk_image(APP_ICON_IMAGE_PATH)
        self.app_icon_image = self.load_tk_image(APP_ICON_TITLE_PATH) or self.window_icon_image
        if self.window_icon_image is not None:
            self.root.iconphoto(True, self.window_icon_image)
        elif APP_ICON_ICO_PATH.exists():
            try:
                self.root.iconbitmap(default=str(APP_ICON_ICO_PATH))
            except tk.TclError:
                pass

        self.config_path = Path("config.json")
        self.image_path: Optional[Path] = None
        self.draw_area: Optional[DrawArea] = None
        self.strokes = []
        self.canvas_size = (640, 360)
        self.drag_start: Optional[Tuple[int, int]] = None
        self.area_rect_id: Optional[int] = None
        self.counting_down = False
        self.click_through = False
        self.area_editing = False
        self.preview_visible = True
        self.preview_refresh_after_id: Optional[str] = None
        self.countdown_after_id: Optional[str] = None
        self.preview_worker_running = False
        self.preview_pending = False
        self.preview_request_id = 0
        self.preview_result_queue = queue.Queue()
        self.draw_ui_queue = queue.Queue()
        self.strokes_signature = None
        self.standby = False
        self.f1_was_pressed = False
        self.f3_was_pressed = False
        self.f4_was_pressed = False
        self.drawing = False
        self.paused_by_hotkey = False

        self.status_var = tk.StringVar(value="先点击“调整区域”，在游戏画面上框选可绘制区域。")
        self.image_var = tk.StringVar(value="尚未选择图片")
        self.area_button_var = tk.StringVar(value="调整区域")
        self.preview_button_var = tk.StringVar(value="关闭预览")
        self.topmost_var = tk.BooleanVar(value=True)
        self.threshold_var = tk.IntVar(value=170)
        self.step_var = tk.IntVar(value=4)
        self.min_run_var = tk.IntVar(value=3)
        self.duration_var = tk.DoubleVar(value=DEFAULT_MOVE_DURATION)
        self.jump_var = tk.DoubleVar(value=0.0)
        self.pause_var = tk.DoubleVar(value=0.0)
        self.invert_var = tk.BooleanVar(value=False)
        self.mode_var = tk.StringVar(value="线条描绘")
        self.line_count_var = tk.IntVar(value=500)
        self.analysis_result = ""
        self.guide_expanded = tk.BooleanVar(value=False)
        self.analysis_expanded = tk.BooleanVar(value=True)
        self.next_draw_timing: Optional[Tuple[float, float, float]] = None

        self.canvas = tk.Canvas(self.root, bg=TRANSPARENT_COLOR, highlightthickness=0, cursor="arrow")
        self.canvas.pack(fill="both", expand=True)

        self.panel = self.build_panel()
        self.panel_window_id = self.canvas.create_window(18, 18, anchor="nw", window=self.panel, tags=("panel",))
        self.panel_drag_start: Optional[Tuple[int, int]] = None
        self.panel_drag_origin: Optional[Tuple[float, float]] = None

        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.root.bind("<F4>", lambda _event: self.enter_standby())

        self.load_existing_area()
        self.redraw_area()
        self.root.update_idletasks()
        self.start_hotkey_worker()
        self.root.after(80, self.poll_preview_results)
        self.root.after(50, self.poll_draw_ui_events)
        self.root.after(80, self.focus_frontend)
        self.root.after(1000, self.enforce_front)

    @staticmethod
    def load_tk_image(image_path: Path) -> Optional[tk.PhotoImage]:
        if not image_path.exists():
            return None
        try:
            return tk.PhotoImage(file=str(image_path))
        except tk.TclError:
            return None

    def build_panel(self) -> tk.Frame:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Drawer.TButton",
            padding=(9, 5),
            background=BUTTON_BG,
            foreground=TEXT_COLOR,
            bordercolor=PANEL_BORDER,
            lightcolor=BUTTON_BG,
            darkcolor=PANEL_BORDER,
            focusthickness=1,
            focuscolor=ACCENT_SOFT,
        )
        style.map(
            "Drawer.TButton",
            background=[("pressed", ACCENT_SOFT), ("active", BUTTON_ACTIVE_BG)],
            foreground=[("disabled", "#8192a1"), ("active", TEXT_COLOR)],
            bordercolor=[("active", ACCENT)],
        )

        panel = tk.Frame(
            self.canvas,
            bg=PANEL_BG,
            bd=0,
            padx=12,
            pady=10,
            highlightbackground=PANEL_BORDER,
            highlightthickness=1,
        )
        title_bar = tk.Frame(panel, bg=PANEL_RAIL_BG, padx=6, pady=4)
        title_bar.grid(row=0, column=0, columnspan=8, sticky="ew")
        title_bar.grid_columnconfigure(0, weight=1)

        title_group = tk.Frame(title_bar, bg=PANEL_RAIL_BG)
        title_group.grid(row=0, column=0, sticky="w")
        icon_label = None
        if self.app_icon_image is not None:
            icon_label = tk.Label(title_group, image=self.app_icon_image, bg=PANEL_RAIL_BG, bd=0)
            icon_label.pack(side="left", padx=(0, 7))
        title = tk.Label(
            title_group,
            text=APP_TITLE,
            bg=PANEL_RAIL_BG,
            fg=TEXT_COLOR,
            font=("Microsoft YaHei UI", 13, "bold"),
        )
        title.pack(side="left")
        ttk.Button(title_bar, text="最小化", style="Drawer.TButton", command=self.minimize_frontend).grid(
            row=0, column=1, sticky="e", padx=(8, 4)
        )
        ttk.Button(title_bar, text="关闭", style="Drawer.TButton", command=self.root.destroy).grid(row=0, column=2)

        hint = tk.Label(
            panel,
            text="先调整区域并选择图片，确认橙色预览后，按 F4 后台待命，或直接点击开始绘制。",
            bg=PANEL_BG,
            fg=MUTED_TEXT,
            font=("Microsoft YaHei UI", 9),
            anchor="w",
            justify="left",
        )
        hint.grid(row=1, column=0, columnspan=8, sticky="ew", pady=(0, 2))
        brush_hint = tk.Label(
            panel,
            text="开始前请在游戏里点开画笔。绘制中按 F5 暂停/继续，暂停时本工具会自动显示；长按 F5、Esc 或 F9 停止。",
            bg=PANEL_BG,
            fg=WARNING_TEXT,
            font=("Microsoft YaHei UI", 9),
            anchor="w",
            justify="left",
        )
        brush_hint.grid(row=2, column=0, columnspan=8, sticky="ew", pady=(0, 8))

        buttons = [
            ("选择图片", self.choose_image),
            (self.area_button_var, self.toggle_area_editing),
            ("刷新预览", self.refresh_preview),
            (self.preview_button_var, self.toggle_preview),
            ("开始绘制(F1)", self.start_countdown),
            ("战争迷雾", self.start_fog_countdown),
            ("退回后台(F4)", self.enter_standby),
        ]
        for column, (text, command) in enumerate(buttons):
            options = {"text": text} if isinstance(text, str) else {"textvariable": text}
            ttk.Button(panel, style="Drawer.TButton", command=command, **options).grid(
                row=3,
                column=column,
                sticky="ew",
                padx=(0, 6),
            )

        tk.Label(panel, textvariable=self.image_var, bg=PANEL_BG, fg=TEXT_COLOR, anchor="w").grid(
            row=4, column=0, columnspan=8, sticky="ew", pady=(8, 2)
        )
        ttk.Button(panel, text="评估图片并套用建议", style="Drawer.TButton", command=self.analyze_current_image).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(2, 6)
        )
        tk.Label(
            panel,
            text="本项目完全开源免费，更新欢迎关注 GitHub 的 taping233。",
            bg=PANEL_BG,
            fg=MUTED_TEXT,
            font=("Microsoft YaHei UI", 9),
            anchor="w",
        ).grid(row=6, column=0, columnspan=8, sticky="ew", pady=(0, 4))
        tk.Label(panel, text="参数说明", bg=PANEL_BG, fg=TEXT_COLOR, font=("Microsoft YaHei UI", 10, "bold")).grid(
            row=20, column=0, columnspan=8, sticky="w", pady=(8, 2)
        )
        self.guide_text = tk.Text(
            panel,
            height=11,
            width=72,
            wrap="word",
            bg=SURFACE_BG,
            fg=MUTED_TEXT,
            relief="solid",
            borderwidth=1,
            font=("Microsoft YaHei UI", 9),
        )
        self.guide_text.grid(row=21, column=0, columnspan=8, sticky="ew", pady=(0, 6))
        self.set_readonly_text(self.guide_text, PARAMETER_GUIDE_TEXT)
        ttk.Button(panel, text="展开/收起", style="Drawer.TButton", command=self.toggle_guide_section).grid(
            row=20, column=7, sticky="e", pady=(8, 2)
        )
        self.guide_text.grid_remove()
        tk.Label(panel, text="图片评估与建议", bg=PANEL_BG, fg=TEXT_COLOR, font=("Microsoft YaHei UI", 10, "bold")).grid(
            row=22, column=0, columnspan=8, sticky="w", pady=(2, 2)
        )
        self.analysis_text = tk.Text(
            panel,
            height=9,
            width=72,
            wrap="word",
            bg=SURFACE_BG,
            fg=TEXT_COLOR,
            relief="solid",
            borderwidth=1,
            font=("Microsoft YaHei UI", 9),
        )
        self.analysis_text.grid(row=23, column=0, columnspan=8, sticky="ew", pady=(0, 8))
        ttk.Button(panel, text="展开/收起", style="Drawer.TButton", command=self.toggle_analysis_section).grid(
            row=22, column=7, sticky="e", pady=(2, 2)
        )
        self.set_readonly_text(self.analysis_text, "选择图片后可以评估：程序会分析明暗、边缘和复杂度，并给出建议模式与参数。")

        self.add_spinner(panel, "阈值", self.threshold_var, 0, 255, 10)
        self.add_spinner(panel, "采样间距", self.step_var, 1, 30, 11)
        self.add_spinner(panel, "最短线段", self.min_run_var, 1, 30, 12)
        self.add_spinner(panel, "绘制秒数", self.duration_var, 0.0, 0.3, 13, increment=0.002)
        self.add_spinner(panel, "定位秒数", self.jump_var, 0.0, 0.2, 14, increment=0.005)
        self.add_spinner(panel, "停顿秒数", self.pause_var, 0.0, 0.2, 15, increment=0.005)
        self.add_spinner(panel, "线条上限", self.line_count_var, 20, 5000, 16, increment=20)

        tk.Checkbutton(
            panel,
            text="强制置顶",
            variable=self.topmost_var,
            bg=CHECK_BG,
            fg=TEXT_COLOR,
            selectcolor=ACCENT_SOFT,
            activebackground=CHECK_BG,
            activeforeground=TEXT_COLOR,
            command=self.apply_topmost,
        ).grid(row=17, column=0, sticky="w", pady=(4, 0))
        tk.Checkbutton(
            panel,
            text="反相",
            variable=self.invert_var,
            bg=CHECK_BG,
            fg=TEXT_COLOR,
            selectcolor=ACCENT_SOFT,
            activebackground=CHECK_BG,
            activeforeground=TEXT_COLOR,
            command=self.schedule_preview_refresh,
        ).grid(row=17, column=1, sticky="w", pady=(4, 0))

        tk.Label(panel, text="绘制风格", bg=PANEL_BG, fg=MUTED_TEXT, anchor="w").grid(row=18, column=0, sticky="w", pady=2)
        mode_frame = tk.Frame(panel, bg=PANEL_BG)
        mode_frame.grid(row=18, column=1, columnspan=7, sticky="w", pady=2)
        for label in MODE_LABELS:
            tk.Radiobutton(
                mode_frame,
                text=label,
                value=label,
                variable=self.mode_var,
                indicatoron=False,
                width=10,
                bg=BUTTON_BG,
                fg=TEXT_COLOR,
                selectcolor=ACCENT_SOFT,
                activebackground=BUTTON_ACTIVE_BG,
                activeforeground=TEXT_COLOR,
                command=self.schedule_preview_refresh,
            ).pack(side="left", padx=(0, 4))

        tk.Label(panel, textvariable=self.status_var, bg=PANEL_BG, fg=ACCENT_DARK, anchor="w").grid(
            row=19, column=0, columnspan=8, sticky="ew", pady=(8, 0)
        )

        for column in range(8):
            panel.grid_columnconfigure(column, weight=0)
        panel.grid_columnconfigure(7, weight=1)

        draggable_widgets = [panel, title_bar, title_group, title, hint, brush_hint]
        if icon_label is not None:
            draggable_widgets.append(icon_label)
        self.bind_panel_drag(*draggable_widgets)

        return panel

    def bind_panel_drag(self, *widgets) -> None:
        for widget in widgets:
            widget.bind("<ButtonPress-1>", self.start_panel_drag)
            widget.bind("<B1-Motion>", self.drag_panel)
            widget.bind("<ButtonRelease-1>", self.end_panel_drag)
            try:
                widget.configure(cursor="fleur")
            except tk.TclError:
                pass

    def start_hotkey_worker(self) -> None:
        threading.Thread(target=self.hotkey_loop, daemon=True).start()

    def hotkey_loop(self) -> None:
        while True:
            f1_pressed = is_f1_pressed()
            f3_pressed = is_f3_pressed()
            f4_pressed = is_f4_pressed()

            if f1_pressed and not self.f1_was_pressed and not self.schedule(self.start_from_hotkey):
                break
            if f3_pressed and not self.f3_was_pressed and not self.schedule(
                lambda: self.show_overlay("工具界面已显示，可以继续修改图片、区域和参数。")
            ):
                break
            if f4_pressed and not self.f4_was_pressed and not self.schedule(self.enter_standby):
                break

            self.f1_was_pressed = f1_pressed
            self.f3_was_pressed = f3_pressed
            self.f4_was_pressed = f4_pressed
            time.sleep(0.06)

    def schedule(self, callback) -> bool:
        try:
            self.root.after(0, callback)
            return True
        except (RuntimeError, tk.TclError):
            return False

    def start_panel_drag(self, event) -> None:
        self.panel_drag_start = (event.x_root, event.y_root)
        x, y = self.canvas.coords(self.panel_window_id)
        self.panel_drag_origin = (x, y)
        self.canvas.tag_raise(self.panel_window_id)

    def drag_panel(self, event) -> None:
        if self.panel_drag_start is None or self.panel_drag_origin is None:
            return
        dx = event.x_root - self.panel_drag_start[0]
        dy = event.y_root - self.panel_drag_start[1]
        width = max(1, self.root.winfo_width())
        height = max(1, self.root.winfo_height())
        panel_width = self.panel.winfo_width()
        panel_height = self.panel.winfo_height()
        max_x = max(DRAG_EDGE_PADDING, width - panel_width - DRAG_EDGE_PADDING)
        max_y = max(DRAG_EDGE_PADDING, height - panel_height - DRAG_EDGE_PADDING)
        x = max(DRAG_EDGE_PADDING, min(self.panel_drag_origin[0] + dx, max_x))
        y = max(DRAG_EDGE_PADDING, min(self.panel_drag_origin[1] + dy, max_y))
        self.canvas.coords(self.panel_window_id, round(x), round(y))

    def end_panel_drag(self, _event) -> None:
        self.panel_drag_start = None
        self.panel_drag_origin = None

    def minimize_frontend(self) -> None:
        self.set_area_editing(False)
        self.disable_click_through()
        self.root.iconify()

    def disable_click_through(self) -> None:
        set_window_click_through(self.root.winfo_id(), False)
        self.click_through = False

    def update_canvas_cursor(self) -> None:
        self.canvas.configure(cursor="crosshair" if self.area_editing else "arrow")

    def set_area_editing(self, enabled: bool) -> None:
        if enabled:
            if self.click_through:
                self.disable_click_through()
            self.root.deiconify()
            self.canvas.itemconfigure("panel", state="normal")
            self.root.attributes("-alpha", 0.28)
            self.canvas.configure(bg="#14212b")

        self.area_editing = enabled
        self.area_button_var.set("完成区域" if enabled else "调整区域")
        if not enabled:
            self.root.attributes("-alpha", 1.0)
            self.canvas.configure(bg=TRANSPARENT_COLOR)
        self.update_canvas_cursor()
        if enabled:
            self.status_var.set("区域调整已开启：在游戏画面上拖动矩形，松开后保存。")
        else:
            self.drag_start = None
            self.status_var.set("区域调整已关闭。")

    def toggle_area_editing(self) -> None:
        self.set_area_editing(not self.area_editing)

    def toggle_preview(self) -> None:
        self.preview_visible = not self.preview_visible
        self.preview_button_var.set("关闭预览" if self.preview_visible else "显示预览")
        self.canvas.delete("preview")
        if not self.preview_visible:
            self.status_var.set("预览已关闭，绘制计划仍会保留。")
            return
        if self.strokes:
            self.draw_preview_lines()
            self.status_var.set(f"预览已显示：{len(self.strokes)} 条笔触。")
        elif self.image_path is not None and self.draw_area is not None:
            self.refresh_preview()
        else:
            self.status_var.set("预览已开启。选择图片并设置区域后会显示。")

    def schedule_preview_refresh(self, delay_ms: int = 250) -> None:
        if self.preview_refresh_after_id is not None:
            try:
                self.root.after_cancel(self.preview_refresh_after_id)
            except tk.TclError:
                pass
        self.preview_refresh_after_id = self.root.after(delay_ms, self.run_scheduled_preview_refresh)

    def run_scheduled_preview_refresh(self) -> None:
        self.preview_refresh_after_id = None
        self.refresh_preview()

    def setting_int(self, variable, default: int, lower: int, upper: int) -> int:
        try:
            value = int(variable.get())
        except (ValueError, tk.TclError):
            return default
        return max(lower, min(value, upper))

    def setting_float(self, variable, default: float, lower: float, upper: float) -> float:
        try:
            value = float(variable.get())
        except (ValueError, tk.TclError):
            return default
        return max(lower, min(value, upper))

    def current_preview_request(self):
        if self.image_path is None:
            self.strokes = []
            self.strokes_signature = None
            self.canvas.delete("preview")
            self.status_var.set("请先选择图片。")
            return None
        if self.draw_area is None:
            self.strokes = []
            self.strokes_signature = None
            self.canvas.delete("preview")
            self.status_var.set("请先在游戏画面上框选绘制区域。")
            return None

        canvas_size = self.virtual_canvas_size(self.draw_area)
        mode = MODE_LABELS.get(self.mode_var.get(), "scanline")
        threshold = self.setting_int(self.threshold_var, 170, 0, 255)
        step = self.setting_int(self.step_var, 4, 1, 30)
        min_run = self.setting_int(self.min_run_var, 3, 1, 30)
        line_count = self.setting_int(self.line_count_var, 500, 20, 5000)
        invert = bool(self.invert_var.get())
        signature = (
            str(self.image_path),
            self.image_path.stat().st_mtime_ns if self.image_path.exists() else 0,
            self.draw_area.left,
            self.draw_area.top,
            self.draw_area.width,
            self.draw_area.height,
            canvas_size,
            mode,
            threshold,
            step,
            min_run,
            line_count,
            invert,
        )
        return {
            "image_path": self.image_path,
            "canvas_size": canvas_size,
            "mode": mode,
            "threshold": threshold,
            "step": step,
            "min_run": min_run,
            "line_count": line_count,
            "invert": invert,
            "signature": signature,
        }

    def start_preview_worker(self, request: dict) -> None:
        self.preview_worker_running = True
        threading.Thread(target=self.build_preview_in_background, args=(request,), daemon=True).start()

    def build_preview_in_background(self, request: dict) -> None:
        try:
            strokes = self.build_strokes_for_request(request)
            self.preview_result_queue.put(
                {
                    "ok": True,
                    "request_id": request["request_id"],
                    "signature": request["signature"],
                    "canvas_size": request["canvas_size"],
                    "strokes": strokes,
                }
            )
        except Exception as exc:
            self.preview_result_queue.put(
                {
                    "ok": False,
                    "request_id": request["request_id"],
                    "signature": request["signature"],
                    "error": str(exc),
                }
            )

    @staticmethod
    def build_strokes_for_request(request: dict):
        if request["mode"] in ("lines", "handdrawn"):
            luminance = load_luminance_canvas(request["image_path"], request["canvas_size"], request["invert"])
            if request["mode"] == "handdrawn":
                return trace_handdrawn_strokes(
                    luminance,
                    request["step"],
                    request["threshold"],
                    request["min_run"],
                    request["line_count"],
                )
            return trace_line_reduction_strokes(
                luminance,
                request["step"],
                request["threshold"],
                request["min_run"],
                request["line_count"],
            )
        mask = load_mask(request["image_path"], request["canvas_size"], request["threshold"], request["invert"])
        return trace_strokes(mask, request["step"], request["min_run"], request["mode"])

    def poll_preview_results(self) -> None:
        handled = False
        while True:
            try:
                result = self.preview_result_queue.get_nowait()
            except queue.Empty:
                break
            handled = True
            self.apply_preview_result(result)

        if handled:
            self.preview_worker_running = False
            if self.preview_pending:
                self.preview_pending = False
                self.refresh_preview()

        self.root.after(80, self.poll_preview_results)

    def apply_preview_result(self, result: dict) -> None:
        if result.get("request_id") != self.preview_request_id:
            return
        if not result.get("ok"):
            self.strokes = []
            self.strokes_signature = None
            self.canvas.delete("preview")
            self.status_var.set(f"预览失败：{result.get('error')}")
            return

        self.canvas_size = result["canvas_size"]
        self.strokes = result["strokes"]
        self.strokes_signature = result["signature"]
        self.canvas.delete("preview")
        if self.preview_visible:
            self.draw_preview_lines()
            self.status_var.set(f"已生成 {len(self.strokes)} 条笔触。")
        else:
            self.status_var.set(f"已生成 {len(self.strokes)} 条笔触，预览当前关闭。")

    def enter_standby(self) -> None:
        self.standby = True
        self.counting_down = False
        self.cancel_countdown()
        self.set_area_editing(False)
        self.canvas.delete("countdown")
        if self.draw_area is not None:
            save_draw_area(self.config_path, self.draw_area)
        self.root.withdraw()
        self.status_var.set("已退回后台。按 F3 呼回前端，按 F1 开始绘制。")

    def hide_overlay_for_drawing(self) -> None:
        self.set_area_editing(False)
        self.cancel_countdown()
        self.disable_click_through()
        self.canvas.delete("countdown")
        self.canvas.itemconfigure("panel", state="hidden")
        self.show_drawing_hint()
        set_window_click_through(self.root.winfo_id(), True)
        self.click_through = True
        self.root.update_idletasks()

    def show_drawing_hint(self) -> None:
        self.canvas.delete("drawing_hint")
        self.root.deiconify()
        self.root.attributes("-topmost", True)
        self.root.update_idletasks()
        width = max(1, self.root.winfo_width())
        x2 = width - 18
        x1 = max(18, x2 - 220)
        y1 = 18
        y2 = 76
        self.canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            fill=TEXT_COLOR,
            outline=PREVIEW_COLOR,
            width=2,
            tags=("drawing_hint",),
        )
        self.canvas.create_text(
            (x1 + x2) // 2,
            y1 + 20,
            text="正在绘画",
            fill="#f9fafb",
            font=("Microsoft YaHei UI", 11, "bold"),
            tags=("drawing_hint",),
        )
        self.canvas.create_text(
            (x1 + x2) // 2,
            y1 + 43,
            text="按 F5 暂停并显示完整前端",
            fill="#fdba74",
            font=("Microsoft YaHei UI", 9),
            tags=("drawing_hint",),
        )

    def start_from_hotkey(self) -> None:
        if self.drawing or self.counting_down:
            return
        if not self.prepare_strokes(show_errors=False):
            self.show_overlay("还没有准备好：请先框选区域并选择图片。")
            return
        if self.root.state() != "withdrawn" and not self.standby and not self.click_through:
            self.start_countdown()
            return
        self.next_draw_timing = None
        self.run_draw(restore_overlay=True)

    def show_overlay(self, status: Optional[str] = None) -> None:
        self.standby = False
        self.disable_click_through()
        self.root.deiconify()
        self.canvas.itemconfigure("panel", state="normal")
        self.root.update_idletasks()
        self.set_area_editing(False)
        self.focus_frontend(update_status=False)
        if status:
            self.status_var.set(status)

    def apply_topmost(self) -> None:
        self.root.attributes("-topmost", bool(self.topmost_var.get()))
        if self.topmost_var.get():
            self.root.lift()

    def enforce_front(self) -> None:
        if self.root.state() != "withdrawn" and self.topmost_var.get():
            self.root.attributes("-topmost", True)
            self.root.lift()
        self.root.after(1000, self.enforce_front)

    def focus_frontend(self, update_status: bool = True) -> None:
        self.root.deiconify()
        self.root.update_idletasks()
        self.disable_click_through()
        self.set_area_editing(False)
        self.root.attributes("-topmost", False)
        self.apply_topmost()
        self.root.lift()
        activate_window(self.root.winfo_id())
        self.root.focus_force()
        self.canvas.focus_set()
        if update_status:
            self.status_var.set("前端已获得焦点，可以继续调整。")

    def prepare_strokes(self, show_errors: bool = True) -> bool:
        if self.image_path is None:
            if show_errors:
                messagebox.showwarning("没有图片", "请先选择图片。")
            return False
        if self.draw_area is None:
            if show_errors:
                messagebox.showwarning("没有区域", "请先拖动框选游戏里的绘制区域。")
            return False

        request = self.current_preview_request()
        if request is None:
            return False
        if self.strokes_signature != request["signature"]:
            self.refresh_preview()
            if show_errors:
                messagebox.showwarning("预览生成中", "当前设置的笔触还在生成，请稍等预览完成后再开始绘制。")
            return False
        if not self.strokes:
            if show_errors:
                messagebox.showwarning("没有笔触", "当前参数没有生成可绘制笔触，请调整阈值或反相。")
            return False
        return True

    def add_spinner(
        self,
        panel: tk.Frame,
        label: str,
        variable,
        from_: float,
        to: float,
        row: int,
        increment: float = 1,
    ) -> None:
        tk.Label(panel, text=label, bg=PANEL_BG, fg=MUTED_TEXT, anchor="w").grid(row=row, column=0, sticky="w", pady=2)
        tk.Spinbox(
            panel,
            textvariable=variable,
            from_=from_,
            to=to,
            increment=increment,
            width=8,
            bg=SURFACE_BG,
            fg=TEXT_COLOR,
            buttonbackground=ACCENT_SOFT,
            relief="solid",
            command=self.schedule_preview_refresh,
        ).grid(row=row, column=1, sticky="w", pady=2)

    @staticmethod
    def set_readonly_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def toggle_guide_section(self) -> None:
        expanded = not self.guide_expanded.get()
        self.guide_expanded.set(expanded)
        if expanded:
            self.guide_text.grid()
        else:
            self.guide_text.grid_remove()

    def toggle_analysis_section(self) -> None:
        expanded = not self.analysis_expanded.get()
        self.analysis_expanded.set(expanded)
        if expanded:
            self.analysis_text.grid()
        else:
            self.analysis_text.grid_remove()

    def analyze_current_image(self) -> None:
        if self.image_path is None:
            messagebox.showwarning("没有图片", "请先选择图片。")
            return

        canvas_size = self.virtual_canvas_size(self.draw_area) if self.draw_area is not None else (640, 360)
        assessment = analyze_image_parameters(self.image_path, canvas_size, bool(self.invert_var.get()))
        self.apply_assessment(assessment)
        self.analysis_result = self.format_assessment(assessment)
        self.set_readonly_text(self.analysis_text, self.analysis_result)
        self.status_var.set("已根据图片内容更新建议参数，可直接预览或继续微调。")
        self.schedule_preview_refresh(delay_ms=50)

    def apply_assessment(self, assessment: dict) -> None:
        self.mode_var.set(MODE_REVERSE_LABELS.get(assessment["suggested_mode"], self.mode_var.get()))
        self.threshold_var.set(int(assessment["threshold"]))
        self.step_var.set(int(assessment["step"]))
        self.min_run_var.set(int(assessment["min_run"]))
        self.line_count_var.set(int(assessment["line_count"]))
        self.duration_var.set(float(assessment["move_duration"]))
        self.jump_var.set(float(assessment["jump_duration"]))
        self.pause_var.set(float(assessment["pause_duration"]))

    def format_assessment(self, assessment: dict) -> str:
        complexity = assessment["complexity"]
        if complexity >= 0.7:
            complexity_text = "高"
        elif complexity >= 0.4:
            complexity_text = "中"
        else:
            complexity_text = "低"

        tuning = []
        if assessment["edge_density"] >= 0.14:
            tuning.append("边缘很多，想保留更多细节时把采样间距降到 2-3，并把线条上限提高。")
        else:
            tuning.append("边缘不密，优先提高阈值或改用线条描绘，不要一开始就把线条上限拉满。")
        if assessment["dark_ratio"] >= 0.28:
            tuning.append("暗部比例较高，阈值过高会带来大面积黑块，先小幅下调阈值观察。")
        else:
            tuning.append("暗部不多，阈值太低会丢掉主体，线不够时优先提高阈值。")
        if assessment["line_count"] >= 1800:
            tuning.append("推荐线条上限较高，绘制会变慢。先看预览，再决定是否继续提高。")
        else:
            tuning.append("当前图不需要特别高的笔触数，先从建议值开始更稳。")

        return (
            f"图片评估\n"
            f"亮度均值：{assessment['mean_luminance']:.1f} / 255\n"
            f"对比度：{assessment['contrast_std']:.1f}\n"
            f"暗部占比：{assessment['dark_ratio'] * 100:.1f}%\n"
            f"边缘密度：{assessment['edge_density'] * 100:.1f}%\n"
            f"复杂度：{complexity_text}\n\n"
            f"推荐参数\n"
            f"模式：{MODE_REVERSE_LABELS.get(assessment['suggested_mode'], assessment['suggested_mode'])}\n"
            f"阈值：{assessment['threshold']}\n"
            f"采样间距：{assessment['step']}\n"
            f"最短线段：{assessment['min_run']}\n"
            f"线条上限：{assessment['line_count']}\n"
            f"绘制秒数：{assessment['move_duration']:.3f}\n"
            f"定位秒数：{assessment['jump_duration']:.3f}\n"
            f"停顿秒数：{assessment['pause_duration']:.3f}\n\n"
            f"调参建议\n- " + "\n- ".join(tuning)
        )

    def load_existing_area(self) -> None:
        if not self.config_path.exists():
            return
        try:
            self.draw_area = load_draw_area(self.config_path)
            self.status_var.set("已读取 config.json。需要调整时点击“调整区域”。")
        except (OSError, KeyError, ValueError, TypeError):
            self.status_var.set("无法读取 config.json，请重新框选区域。")

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择要绘制的图片",
            filetypes=(("图片文件", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")),
        )
        if not path:
            return
        self.image_path = Path(path)
        self.image_var.set(self.image_path.name)
        self.analyze_current_image()
        self.schedule_preview_refresh(delay_ms=50)

    def on_drag_start(self, event) -> None:
        if not self.area_editing:
            return
        self.drag_start = (event.x, event.y)
        self.canvas.delete("preview")
        self.status_var.set("松开鼠标后锁定绘制区域。")

    def on_drag_move(self, event) -> None:
        if self.drag_start is None:
            return
        x1, y1 = self.drag_start
        self.draw_selection_rect(x1, y1, event.x, event.y)

    def on_drag_end(self, event) -> None:
        if self.drag_start is None:
            return
        x1, y1 = self.drag_start
        x2, y2 = event.x, event.y
        self.drag_start = None

        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        width = right - left
        height = bottom - top
        if width < 20 or height < 20:
            self.status_var.set("区域太小，请拖一个更大的矩形。")
            return

        self.draw_area = DrawArea(left=left, top=top, width=width, height=height)
        save_draw_area(self.config_path, self.draw_area)
        self.redraw_area()
        self.set_area_editing(False)
        self.status_var.set(f"区域已保存：{width} x {height}。")
        self.schedule_preview_refresh(delay_ms=50)

    def draw_selection_rect(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if self.area_rect_id is None:
            self.area_rect_id = self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline=AREA_COLOR,
                width=3,
                dash=(8, 4),
                tags=("area",),
            )
        else:
            self.canvas.coords(self.area_rect_id, x1, y1, x2, y2)

    def redraw_area(self) -> None:
        self.canvas.delete("area")
        self.area_rect_id = None
        if self.draw_area is None:
            return
        self.area_rect_id = self.canvas.create_rectangle(
            self.draw_area.left,
            self.draw_area.top,
            self.draw_area.left + self.draw_area.width,
            self.draw_area.top + self.draw_area.height,
            outline=AREA_COLOR,
            width=3,
            dash=(8, 4),
            tags=("area",),
        )

    def refresh_preview(self) -> None:
        if self.preview_refresh_after_id is not None:
            try:
                self.root.after_cancel(self.preview_refresh_after_id)
            except tk.TclError:
                pass
            self.preview_refresh_after_id = None

        request = self.current_preview_request()
        if request is None:
            return
        if self.strokes_signature == request["signature"] and self.strokes:
            self.canvas.delete("preview")
            if self.preview_visible:
                self.draw_preview_lines()
                self.status_var.set(f"预览已刷新：{len(self.strokes)} 条笔触。")
            else:
                self.status_var.set(f"笔触已是最新：{len(self.strokes)} 条，预览当前关闭。")
            return

        self.preview_request_id += 1
        request["request_id"] = self.preview_request_id
        self.canvas.delete("preview")
        self.status_var.set("预览生成中，切换模式不会阻塞界面。")

        if self.preview_worker_running:
            self.preview_pending = True
            return
        self.start_preview_worker(request)

    def draw_preview_lines(self) -> None:
        if not self.preview_visible or self.draw_area is None:
            return
        total = len(self.strokes)
        if total == 0:
            return
        stride = max(1, total // MAX_PREVIEW_LINES)
        for stroke in self.strokes[::stride]:
            x1, y1 = self.draw_area.to_screen(stroke.x1, stroke.y1, self.canvas_size)
            x2, y2 = self.draw_area.to_screen(stroke.x2, stroke.y2, self.canvas_size)
            self.canvas.create_line(x1, y1, x2, y2, fill=PREVIEW_COLOR, width=2, tags=("preview",))

    def start_countdown(self) -> None:
        if self.counting_down or self.drawing:
            return
        self.set_area_editing(False)
        if not self.prepare_strokes():
            return

        self.next_draw_timing = None
        save_draw_area(self.config_path, self.draw_area)
        self.standby = False
        self.counting_down = True
        self.status_var.set("工具窗口将收起为小提示。请确认游戏内画笔已点开，绘制中按 F5 暂停。")
        self.show_frontend_countdown(3)

    def start_fog_countdown(self) -> None:
        if self.counting_down or self.drawing:
            return
        self.set_area_editing(False)
        if self.draw_area is None:
            messagebox.showwarning("没有区域", "请先拖动框选要涂黑的游戏区域。")
            return

        self.canvas_size = self.virtual_canvas_size(self.draw_area)
        self.strokes = self.build_fog_strokes(self.canvas_size)
        self.strokes_signature = ("fog", self.draw_area.left, self.draw_area.top, self.draw_area.width, self.draw_area.height)
        self.canvas.delete("preview")
        fog_duration = min(self.setting_float(self.duration_var, DEFAULT_MOVE_DURATION, 0.0, 0.3), FAST_MOVE_DURATION)
        self.next_draw_timing = (fog_duration, 0.0, 0.0)
        save_draw_area(self.config_path, self.draw_area)
        self.standby = False
        self.counting_down = True
        self.status_var.set("战争迷雾即将开始：会把当前框选区域横向涂黑。绘制中按 F5 暂停，长按 F5、Esc 或 F9 停止。")
        self.show_frontend_countdown(3)

    @staticmethod
    def build_fog_strokes(canvas_size: Tuple[int, int]) -> List[Stroke]:
        width, height = canvas_size
        strokes: List[Stroke] = []
        for y in range(height):
            if y % 2 == 0:
                strokes.append(Stroke(0, y, width - 1, y))
            else:
                strokes.append(Stroke(width - 1, y, 0, y))
        return strokes

    def run_draw(self, restore_overlay: bool = True, initial_countdown: int = 0) -> None:
        if self.draw_area is None:
            return

        self.drawing = True
        self.paused_by_hotkey = False
        self.counting_down = False
        self.hide_overlay_for_drawing()
        draw_area = self.draw_area
        strokes = list(self.strokes)
        canvas_size = self.canvas_size
        if self.next_draw_timing is None:
            move_duration = self.setting_float(self.duration_var, DEFAULT_MOVE_DURATION, 0.0, 0.3)
            stroke_pause = self.setting_float(self.pause_var, 0.0, 0.0, 0.2)
            jump_duration = self.setting_float(self.jump_var, 0.0, 0.0, 0.2)
        else:
            move_duration, stroke_pause, jump_duration = self.next_draw_timing
            self.next_draw_timing = None
        threading.Thread(
            target=self.draw_worker,
            args=(
                strokes,
                draw_area,
                canvas_size,
                initial_countdown,
                move_duration,
                stroke_pause,
                jump_duration,
                restore_overlay,
            ),
            daemon=True,
        ).start()

    def draw_worker(
        self,
        strokes,
        draw_area: DrawArea,
        canvas_size: Tuple[int, int],
        initial_countdown: int,
        move_duration: float,
        stroke_pause: float,
        jump_duration: float,
        restore_overlay: bool,
    ) -> None:
        time.sleep(0.35)
        focused = focus_window_under_area(draw_area)
        time.sleep(0.25)
        try:
            draw_strokes(
                strokes=strokes,
                area=draw_area,
                canvas_size=canvas_size,
                countdown=initial_countdown,
                move_duration=move_duration,
                stroke_pause=stroke_pause,
                button="left",
                jump_duration=jump_duration,
                on_pause_change=self.on_drawing_pause_change,
            )
            result = "绘制完成。可以重新框选区域或选择另一张图片。"
        except KeyboardInterrupt:
            result = "已停止。"
        except Exception as exc:
            result = f"绘制失败：{exc}"
        if not focused:
            result += " 提醒：未能确认游戏窗口焦点，建议使用无边框窗口模式。"

        self.draw_ui_queue.put(("finish", result, restore_overlay))

    def finish_draw(self, result: str, restore_overlay: bool = True) -> None:
        self.drawing = False
        self.paused_by_hotkey = False
        self.canvas.delete("drawing_hint")
        self.disable_click_through()
        self.exit_current_preview()
        if restore_overlay:
            self.root.deiconify()
            self.canvas.itemconfigure("panel", state="normal")
            self.root.attributes("-topmost", True)
            self.root.lift()
            activate_window(self.root.winfo_id())
            self.root.focus_force()
            self.canvas.focus_set()
            self.standby = False
        else:
            self.standby = True
        self.status_var.set(f"{result} 已自动关闭本次预览。")
        self.root.update_idletasks()

    def exit_current_preview(self) -> None:
        self.preview_visible = False
        self.preview_button_var.set("显示预览")
        self.canvas.delete("preview")

    def on_drawing_pause_change(self, paused: bool) -> None:
        self.draw_ui_queue.put(("pause", paused))

    def poll_draw_ui_events(self) -> None:
        while True:
            try:
                event = self.draw_ui_queue.get_nowait()
            except queue.Empty:
                break
            if not event:
                continue
            if event[0] == "pause":
                self.apply_pause_ui_state(bool(event[1]))
            elif event[0] == "finish":
                self.finish_draw(str(event[1]), bool(event[2]))
        self.root.after(50, self.poll_draw_ui_events)

    def apply_pause_ui_state(self, paused: bool) -> None:
        self.paused_by_hotkey = paused
        if paused:
            self.canvas.delete("drawing_hint")
            self.disable_click_through()
            self.root.deiconify()
            self.canvas.itemconfigure("panel", state="normal")
            self.root.attributes("-topmost", True)
            self.root.lift()
            activate_window(self.root.winfo_id())
            self.status_var.set("绘制已暂停。按 F5 继续，长按 F5、Esc 或 F9 停止。")
        else:
            self.canvas.itemconfigure("panel", state="hidden")
            self.show_drawing_hint()
            set_window_click_through(self.root.winfo_id(), True)
            self.click_through = True
            self.status_var.set("绘制继续，前端已收起为 F5 暂停提示。")
            self.root.update_idletasks()
            if self.draw_area is not None:
                focus_window_under_area(self.draw_area)
        self.root.update_idletasks()

    def cancel_countdown(self) -> None:
        if self.countdown_after_id is None:
            return
        try:
            self.root.after_cancel(self.countdown_after_id)
        except tk.TclError:
            pass
        self.countdown_after_id = None

    def show_frontend_countdown(self, seconds: int) -> None:
        self.cancel_countdown()
        self.canvas.itemconfigure("panel", state="hidden")
        self.root.deiconify()
        self.root.update_idletasks()
        self.draw_countdown_overlay(seconds)
        self.countdown_after_id = self.root.after(1000, lambda: self.advance_countdown(seconds - 1))

    def advance_countdown(self, seconds: int) -> None:
        self.countdown_after_id = None
        if not self.counting_down:
            self.canvas.delete("countdown")
            self.canvas.itemconfigure("panel", state="normal")
            return
        if seconds <= 0:
            self.canvas.delete("countdown")
            self.run_draw(restore_overlay=True, initial_countdown=0)
            return
        self.draw_countdown_overlay(seconds)
        self.countdown_after_id = self.root.after(1000, lambda: self.advance_countdown(seconds - 1))

    def draw_countdown_overlay(self, seconds: int) -> None:
        self.canvas.delete("countdown")
        width = max(1, self.root.winfo_width())
        height = max(1, self.root.winfo_height())
        center_x = width // 2
        center_y = height // 2
        self.canvas.create_rectangle(
            center_x - 120,
            center_y - 110,
            center_x + 120,
            center_y + 110,
            fill=TEXT_COLOR,
            outline=PREVIEW_COLOR,
            width=3,
            tags=("countdown",),
        )
        self.canvas.create_text(
            center_x,
            center_y - 10,
            text=str(seconds),
            fill="#f9fafb",
            font=("Microsoft YaHei UI", 68, "bold"),
            tags=("countdown",),
        )
        self.canvas.create_text(
            center_x,
            center_y + 52,
            text="即将开始",
            fill="#fdba74",
            font=("Microsoft YaHei UI", 16, "bold"),
            tags=("countdown",),
        )

    @staticmethod
    def virtual_canvas_size(area: DrawArea) -> Tuple[int, int]:
        width = 640
        height = max(1, round(width * area.height / max(area.width, 1)))
        return width, height

    def run(self) -> None:
        self.root.mainloop()


class GuiInstanceLock:
    def __init__(self) -> None:
        self.handle = None

    def acquire(self) -> bool:
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.GetLastError.restype = ctypes.c_ulong
        self.handle = kernel32.CreateMutexW(None, False, GUI_MUTEX_NAME)
        return bool(self.handle) and kernel32.GetLastError() != 183

    def close(self) -> None:
        if not self.handle:
            return
        ctypes.windll.kernel32.CloseHandle(self.handle)
        self.handle = None


def focus_existing_gui() -> bool:
    ctypes.windll.user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    ctypes.windll.user32.FindWindowW.restype = ctypes.c_void_p
    hwnd = ctypes.windll.user32.FindWindowW(None, APP_TITLE)
    if not hwnd:
        return False
    activate_window(hwnd)
    return True


def notify_existing_gui() -> None:
    if focus_existing_gui():
        return
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(APP_TITLE, "工具已经在后台运行。最多只允许一个后台，请按 F3 呼回现有前端。")
    root.destroy()


def run_gui() -> None:
    instance_lock = GuiInstanceLock()
    if not instance_lock.acquire():
        notify_existing_gui()
        return
    try:
        app = OverlayApp()
        app.run()
    finally:
        instance_lock.close()
