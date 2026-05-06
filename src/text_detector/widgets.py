"""UI widgets for the text detection app."""

import tkinter as tk

from .config import THEME


class ThemeableButton:
    """Factory for creating styled buttons with hover effects and tooltips."""

    def __init__(self, master, text: str, command, bg: str, active_bg: str,
                 font: str = "Arial", font_size: int = 11, width: int = 14,
                 side: str = "top", padx: int = 0, pady: int = 4,
                 fill: str = "x", expand: bool = False, tooltip: str = ""):
        self.bg = bg
        self.active_bg = active_bg
        self.normal_fg = THEME.button_fg
        self.command = command
        self.tooltip_text = tooltip
        self.tooltip_window: tk.Toplevel | None = None

        self.btn = tk.Button(
            master,
            text=text,
            command=command,
            font=(font, font_size, "bold"),
            bg=bg,
            fg=self.normal_fg,
            activebackground=active_bg,
            activeforeground=self.normal_fg,
            relief="flat",
            borderwidth=0,
            padx=16,
            pady=8,
            cursor="hand2",
            width=width,
        )
        self.btn.pack(side=side, fill=fill, expand=expand, padx=padx, pady=pady)  # type: ignore[arg-type]

        self.btn.bind("<Enter>", self._on_enter)
        self.btn.bind("<Leave>", self._on_leave)

    def _show_tooltip(self, event=None) -> None:
        if not self.tooltip_text or self.tooltip_window:
            return
        tooltip = tk.Toplevel(self.btn)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
        label = tk.Label(
            tooltip,
            text=self.tooltip_text,
            bg="#1E1E2E",
            fg="#CDD6F4",
            font=("Arial", 9),
            padx=6,
            pady=3,
            borderwidth=0,
            relief="flat",
        )
        label.pack()
        self.tooltip_window = tooltip

    def _hide_tooltip(self, _event=None) -> None:
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def _on_enter(self, event=None) -> None:
        self.btn.config(bg=self.active_bg, fg=THEME.background)
        if self.tooltip_text:
            self._show_tooltip(event)

    def _on_leave(self, _event=None) -> None:
        self.btn.config(bg=self.bg, fg=self.normal_fg)
        self._hide_tooltip()

    def config(self, **kwargs) -> None:
        self.btn.config(**kwargs)


class StatusLED:
    """A small circular status indicator widget."""

    def __init__(self, master, color: str = THEME.status_ready, size: int = 12):
        self.color = color
        self.size = size
        self.canvas = tk.Canvas(
            master, width=size, height=size,
            bg=THEME.background, highlightthickness=0,
        )
        self._draw()

    def _draw(self) -> None:
        r = self.size // 2
        self.canvas.create_oval(1, 1, r * 2 - 1, r * 2 - 1,
                                fill=self.color, outline="")

    def set_color(self, color: str) -> None:
        self.color = color
        self._draw()

    def pack(self, **kwargs) -> None:
        self.canvas.pack(**kwargs)
