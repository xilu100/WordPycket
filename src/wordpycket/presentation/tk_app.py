import threading
import time
import tkinter as tk
from collections.abc import Callable
from typing import Protocol
from tkinter import messagebox, ttk

from wordpycket.application.services import WordService
from wordpycket.application.study_session import StudyCardState, StudySessionController
from wordpycket.domain.entities import WordEntry


APP_BG = "#eef3fb"
GLASS_FILL = "#fbfdff"
GLASS_BORDER = "#d8e4f2"
GLASS_HIGHLIGHT = "#ffffff"
TEXT_PRIMARY = "#172033"
TEXT_SECONDARY = "#596579"
ACCENT = "#0a84ff"
ACCENT_HOVER = "#006edb"
DANGER = "#ff3b30"
DANGER_HOVER = "#d92d24"
BLUR_SIGMA = 24


class GlassPanel(tk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        padding: int = 18,
        radius: int = 24,
        blur_sigma: int = BLUR_SIGMA,
        fill: str = GLASS_FILL,
        border: str = GLASS_BORDER,
    ) -> None:
        super().__init__(parent, bg=APP_BG, highlightthickness=0)
        self._padding = padding
        self._radius = radius
        self._blur_sigma = max(10, min(40, blur_sigma))
        self._fill = fill
        self._border = border
        self._canvas = tk.Canvas(self, bg=APP_BG, bd=0, highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self.body = tk.Frame(self._canvas, bg=fill, bd=0, highlightthickness=0)
        self._window_id = self._canvas.create_window(
            padding,
            padding,
            anchor=tk.NW,
            window=self.body,
        )
        self._canvas.bind("<Configure>", self._redraw)

    def _redraw(self, event: tk.Event) -> None:
        width = max(1, event.width)
        height = max(1, event.height)
        inner_width = max(1, width - self._padding * 2)
        inner_height = max(1, height - self._padding * 2)

        self._canvas.delete("glass")
        self._draw_blur_shadow(width, height)
        self._rounded_rect(1, 1, width - 5, height - 7, self._radius, self._fill, self._border, "glass")
        self._canvas.create_line(
            self._radius,
            2,
            width - self._radius,
            2,
            fill=GLASS_HIGHLIGHT,
            width=2,
            tags="glass",
        )
        self._canvas.tag_lower("glass", self._window_id)
        self._canvas.coords(self._window_id, self._padding, self._padding)
        self._canvas.itemconfigure(self._window_id, width=inner_width, height=inner_height)

    def _draw_blur_shadow(self, width: int, height: int) -> None:
        layers = max(5, min(12, self._blur_sigma // 3))
        colors = ["#d7e1ef", "#dce6f2", "#e1e9f4", "#e5edf6", "#e9f0f8", "#edf3fa"]
        for index in range(layers, 0, -1):
            spread = int(index * self._blur_sigma / layers)
            offset = int(spread * 0.35)
            color = colors[min(len(colors) - 1, layers - index)]
            self._rounded_rect(
                4 + spread // 5,
                5 + offset,
                width - 4 - spread // 5,
                height - 2,
                self._radius + spread // 2,
                color,
                "",
                "glass",
            )

    def _rounded_rect(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        radius: int,
        fill: str,
        outline: str,
        tags: str,
    ) -> None:
        radius = min(radius, max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
        points = [
            x1 + radius,
            y1,
            x2 - radius,
            y1,
            x2,
            y1,
            x2,
            y1 + radius,
            x2,
            y2 - radius,
            x2,
            y2,
            x2 - radius,
            y2,
            x1 + radius,
            y2,
            x1,
            y2,
            x1,
            y2 - radius,
            x1,
            y1 + radius,
            x1,
            y1,
        ]
        self._canvas.create_polygon(
            points,
            smooth=True,
            splinesteps=20,
            fill=fill,
            outline=outline,
            tags=tags,
        )


class GlassButton(tk.Canvas):
    def __init__(
        self,
        parent: tk.Widget,
        text: str,
        command: Callable[[], None] | None = None,
        variant: str = "default",
        height: int = 38,
        radius: int = 19,
        state: str = tk.NORMAL,
    ) -> None:
        requested_width = max(96, len(text) * 15 + 34)
        super().__init__(
            parent,
            width=requested_width,
            height=height,
            bg=self._parent_bg(parent),
            bd=0,
            highlightthickness=0,
            cursor="hand2" if state != tk.DISABLED else "",
        )
        self._text = text
        self._command = command
        self._variant = variant
        self._height = height
        self._radius = radius
        self._state = state
        self._hovered = False
        self._pressed = False
        self._text_id: int | None = None
        self.bind("<Configure>", lambda _event: self._draw())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    @staticmethod
    def _parent_bg(parent: tk.Widget) -> str:
        try:
            return str(parent.cget("bg"))
        except tk.TclError:
            return GLASS_FILL

    def configure(self, cnf=None, **kwargs):  # type: ignore[override]
        if cnf:
            kwargs.update(cnf)
        custom_keys = {"text", "state", "command"}
        custom = {key: kwargs.pop(key) for key in list(kwargs) if key in custom_keys}
        if "text" in custom:
            self._text = custom["text"]
        if "state" in custom:
            self._state = custom["state"]
            super().configure(cursor="hand2" if self._state != tk.DISABLED else "")
        if "command" in custom:
            self._command = custom["command"]
        result = super().configure(**kwargs) if kwargs else None
        if custom:
            self._draw()
        return result

    config = configure

    def cget(self, key: str):  # type: ignore[override]
        if key == "text":
            return self._text
        if key == "state":
            return self._state
        return super().cget(key)

    def _palette(self) -> tuple[str, str, str]:
        if self._state == tk.DISABLED:
            return "#edf1f7", "#9aa6b5", "#dce4ee"
        if self._variant == "primary":
            fill = ACCENT_HOVER if self._hovered else ACCENT
            return fill, "#ffffff", "#77baff"
        if self._variant == "danger":
            fill = "#ffe2df" if self._hovered else "#fff0ef"
            return fill, DANGER_HOVER if self._hovered else DANGER, "#ffd2ce"
        fill = "#e2efff" if self._hovered else "#eef5ff"
        return fill, TEXT_PRIMARY, "#d8e6f6"

    def _draw(self) -> None:
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height() or self._height)
        y_shift = 1 if self._pressed and self._state != tk.DISABLED else 0
        fill, text_fill, outline = self._palette()
        self.delete("all")
        self._rounded_rect(2, 4 + y_shift, width - 2, height - 1 + y_shift, self._radius, "#d7e2f0", "")
        self._rounded_rect(1, 1 + y_shift, width - 3, height - 4 + y_shift, self._radius, fill, outline)
        self.create_line(
            self._radius,
            2 + y_shift,
            width - self._radius,
            2 + y_shift,
            fill="#ffffff",
            width=1,
        )
        self._text_id = self.create_text(
            width // 2,
            height // 2 - 1 + y_shift,
            text=self._text,
            fill=text_fill,
            font=("Microsoft YaHei UI", 10, "bold"),
        )

    def _rounded_rect(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        radius: int,
        fill: str,
        outline: str,
    ) -> None:
        radius = min(radius, max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
        points = [
            x1 + radius,
            y1,
            x2 - radius,
            y1,
            x2,
            y1,
            x2,
            y1 + radius,
            x2,
            y2 - radius,
            x2,
            y2,
            x2 - radius,
            y2,
            x1 + radius,
            y2,
            x1,
            y2,
            x1,
            y2 - radius,
            x1,
            y1 + radius,
            x1,
            y1,
        ]
        self.create_polygon(points, smooth=True, splinesteps=20, fill=fill, outline=outline)

    def _on_enter(self, _event: tk.Event) -> None:
        if self._state == tk.DISABLED:
            return
        self._hovered = True
        self._draw()

    def _on_leave(self, _event: tk.Event) -> None:
        self._hovered = False
        self._pressed = False
        self._draw()

    def _on_press(self, _event: tk.Event) -> None:
        if self._state == tk.DISABLED:
            return
        self._pressed = True
        self._draw()

    def _on_release(self, event: tk.Event) -> None:
        if self._state == tk.DISABLED:
            return
        was_pressed = self._pressed
        self._pressed = False
        self._draw()
        if was_pressed and 0 <= event.x <= self.winfo_width() and 0 <= event.y <= self.winfo_height():
            if self._command is not None:
                self._command()


class ExampleGenerator(Protocol):
    def generate(self, entry: WordEntry, scope: str = ""): ...

    def correct_entry(self, entry: WordEntry, scope: str = ""): ...

    def generate_many(self, entries: list[WordEntry], scope: str = "", progress=None, control=None): ...

    def correct_many(self, entries: list[WordEntry], scope: str = "", progress=None, control=None): ...


class WordPycketApp:
    def __init__(
        self,
        service: WordService,
        reset_entries_loader: Callable[[], list[WordEntry]],
        example_generator: ExampleGenerator | None = None,
    ) -> None:
        self._service = service
        self._reset_entries_loader = reset_entries_loader
        self._example_generator = example_generator
        self._is_generating_example = False
        self._is_correcting_entry = False
        self._batch_state = "idle"
        self._selected_id: str | None = None
        self._selected_ids: list[str] = []
        self._mode: str | None = None
        self._study_session = StudySessionController(service)

        self._root = tk.Tk()
        self._root.title("WordPycket")
        self._root.geometry("1220x680")
        self._root.minsize(980, 600)
        self._root.configure(bg=APP_BG)

        self._search_var = tk.StringVar()
        self._ai_scope_var = tk.StringVar(value="人工智能相关的翻译")

        self._configure_style()
        self._show_home()

    def run(self) -> None:
        self._root.mainloop()

    def _button(
        self,
        parent: tk.Widget,
        text: str,
        command: Callable[[], None] | None = None,
        variant: str = "default",
        state: str = tk.NORMAL,
    ) -> GlassButton:
        return GlassButton(parent, text=text, command=command, variant=variant, state=state)

    def _configure_style(self) -> None:
        style = ttk.Style(self._root)
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10))
        style.configure("App.TFrame", background=APP_BG)
        style.configure("Glass.TFrame", background=GLASS_FILL)
        style.configure("Toolbar.TFrame", background=GLASS_FILL)
        style.configure("TLabel", background=GLASS_FILL, foreground=TEXT_PRIMARY)
        style.configure("App.TLabel", background=APP_BG, foreground=TEXT_PRIMARY)
        style.configure("Treeview", rowheight=30, background="#ffffff", fieldbackground="#ffffff", borderwidth=0)
        style.configure(
            "Treeview.Heading",
            background="#eef4fc",
            foreground=TEXT_SECONDARY,
            relief=tk.FLAT,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map("Treeview", background=[("selected", "#d8ebff")], foreground=[("selected", TEXT_PRIMARY)])
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 18, "bold"), foreground=TEXT_PRIMARY)
        style.configure("Word.TLabel", font=("Segoe UI", 36, "bold"), foreground=TEXT_PRIMARY)
        style.configure("Meaning.TLabel", font=("Microsoft YaHei UI", 18), foreground=TEXT_PRIMARY)
        style.configure("Meta.TLabel", foreground=TEXT_SECONDARY)
        style.configure("HomeTitle.TLabel", background=APP_BG, foreground=TEXT_PRIMARY, font=("Segoe UI", 28, "bold"))
        style.configure("Subtitle.TLabel", background=APP_BG, foreground=TEXT_SECONDARY)
        style.configure("ButtonPlaceholder.TLabel", padding=(0, 4), background=GLASS_FILL)
        style.configure(
            "TButton",
            padding=(16, 8),
            borderwidth=0,
            focusthickness=0,
            background="#eef5ff",
            foreground=TEXT_PRIMARY,
            relief=tk.FLAT,
        )
        style.map(
            "TButton",
            background=[("active", "#e2efff"), ("disabled", "#edf1f7")],
            foreground=[("disabled", "#9aa6b5")],
        )
        style.configure("Primary.TButton", background=ACCENT, foreground="#ffffff")
        style.map("Primary.TButton", background=[("active", ACCENT_HOVER), ("disabled", "#a8cdf4")])
        style.configure("Danger.TButton", background="#fff0ef", foreground=DANGER)
        style.map("Danger.TButton", background=[("active", "#ffe2df")], foreground=[("active", DANGER_HOVER)])
        style.configure("TEntry", padding=(10, 7), fieldbackground="#ffffff", bordercolor=GLASS_BORDER)
        style.configure("TProgressbar", troughcolor="#e8eef7", background=ACCENT, borderwidth=0, thickness=8)

    def _show_home(self) -> None:
        self._study_session.leave_active_session()
        self._study_session.reset()

        self._mode = None
        self._selected_id = None
        self._selected_ids = []
        self._clear_root()

        container = ttk.Frame(self._root, padding=32, style="App.TFrame")
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure((0, 1, 2), weight=1)
        container.rowconfigure(2, weight=1)
        container.rowconfigure(3, weight=0)

        ttk.Label(container, text="WordPycket", style="HomeTitle.TLabel").grid(
            row=0,
            column=0,
            columnspan=3,
            sticky=tk.W,
            pady=(0, 8),
        )
        ttk.Label(container, text="选择要进入的功能", style="Subtitle.TLabel").grid(
            row=1,
            column=0,
            columnspan=3,
            sticky=tk.W,
            pady=(0, 24),
        )

        counts = self._pool_counts()
        learning_panel = GlassPanel(container, padding=20)
        learning_panel.grid(row=2, column=0, sticky=tk.NSEW, padx=(0, 12))
        learning = learning_panel.body
        learning.columnconfigure(0, weight=1)
        ttk.Label(learning, text="学习", style="Title.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(
            learning,
            text=f"{counts['learning']} 个单词",
            style="Meta.TLabel",
        ).grid(row=1, column=0, sticky=tk.W, pady=(8, 18))
        self._button(learning, text="进入学习", variant="primary", command=lambda: self._show_mode("learning")).grid(
            row=2,
            column=0,
            sticky=tk.EW,
        )

        review_panel = GlassPanel(container, padding=20)
        review_panel.grid(row=2, column=1, sticky=tk.NSEW, padx=12)
        review = review_panel.body
        review.columnconfigure(0, weight=1)
        ttk.Label(review, text="复习", style="Title.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(
            review,
            text=f"{counts['review']} 个单词",
            style="Meta.TLabel",
        ).grid(row=1, column=0, sticky=tk.W, pady=(8, 18))
        self._button(review, text="进入复习", variant="primary", command=lambda: self._show_mode("review")).grid(
            row=2,
            column=0,
            sticky=tk.EW,
        )

        words_panel = GlassPanel(container, padding=20)
        words_panel.grid(row=2, column=2, sticky=tk.NSEW, padx=(12, 0))
        words = words_panel.body
        words.columnconfigure(0, weight=1)
        ttk.Label(words, text="词表", style="Title.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(
            words,
            text=f"{counts['total']} 个单词",
            style="Meta.TLabel",
        ).grid(row=1, column=0, sticky=tk.W, pady=(8, 18))
        self._button(words, text="查看词表", variant="primary", command=self._show_word_list).grid(
            row=2,
            column=0,
            sticky=tk.EW,
        )

        reset_area = ttk.Frame(container, style="App.TFrame")
        reset_area.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=(24, 0))
        reset_area.columnconfigure(0, weight=1)
        self._button(reset_area, text="重置学习进度", variant="danger", command=self._confirm_reset_progress).grid(
            row=0,
            column=1,
            sticky=tk.E,
        )

    def _confirm_reset_progress(self) -> None:
        imported_count = self._service.replace_words(self._reset_entries_loader())
        self._study_session.clear_last_session()
        messagebox.showinfo(
            "已重置",
            f"已从 CSV 重新导入 {imported_count} 个词条，原有学习记录已清空。",
            parent=self._root,
        )
        self._show_home()

    def _show_mode(self, mode: str) -> None:
        self._mode = mode
        self._selected_id = None
        self._selected_ids = []
        self._search_var.set("")
        self._clear_root()
        self._build_layout()
        self._render_study_card(self._study_session.begin(mode))  # type: ignore[arg-type]

    def _clear_root(self) -> None:
        for child in self._root.winfo_children():
            child.destroy()
        for name in (
            "_tree",
            "_count_label",
            "_batch_progress",
            "_batch_status_label",
            "_pause_batch_button",
            "_stop_batch_button",
            "_actions_frame",
            "_navigation_frame",
            "_unknown_button",
            "_unknown_placeholder",
            "_known_button",
            "_known_placeholder",
            "_definitely_known_button",
            "_definitely_known_placeholder",
            "_previous_button",
            "_previous_placeholder",
            "_next_button",
            "_next_placeholder",
            "_supplement_example_button",
            "_correct_entry_button",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _pool_counts(self) -> dict[str, int]:
        return self._study_session.pool_counts()

    def _build_layout(self) -> None:
        container = ttk.Frame(self._root, padding=24, style="App.TFrame")
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        title_bar = ttk.Frame(container, style="App.TFrame")
        title_bar.grid(row=0, column=0, sticky=tk.EW, pady=(0, 14))
        title_bar.columnconfigure(0, weight=1)
        title_text = "学习" if self._mode == "learning" else "复习"
        ttk.Label(title_bar, text=title_text, style="HomeTitle.TLabel").grid(row=0, column=0, sticky=tk.W)
        self._button(title_bar, text="返回主页", command=self._show_home).grid(row=0, column=1, sticky=tk.E)

        review_panel = GlassPanel(container, padding=0, radius=32)
        review_panel.grid(row=1, column=0, sticky=tk.N)
        review_panel.configure(width=640, height=552)
        review_panel.grid_propagate(False)
        review = review_panel.body
        review.columnconfigure(0, weight=1)

        content = tk.Frame(review, bg=GLASS_FILL, bd=0, highlightthickness=0)
        content.place(x=28, y=24, width=584, height=312)
        content.columnconfigure(0, weight=1)

        self._word_label = ttk.Label(content, text="", style="Word.TLabel", anchor=tk.CENTER)
        self._word_label.grid(row=0, column=0, sticky=tk.EW, pady=(8, 12))

        self._meaning_label = ttk.Label(content, text="", style="Meaning.TLabel", wraplength=584, anchor=tk.CENTER)
        self._meaning_label.grid(row=1, column=0, sticky=tk.EW, pady=(0, 12))

        self._forms_label = ttk.Label(content, text="", style="Meta.TLabel", wraplength=584, anchor=tk.CENTER)
        self._forms_label.grid(row=2, column=0, sticky=tk.EW, pady=(0, 16))

        self._example_label = ttk.Label(content, text="", style="Meta.TLabel", wraplength=584, anchor=tk.CENTER)
        self._example_label.grid(row=3, column=0, sticky=tk.EW, pady=(0, 8))

        self._example_cn_label = ttk.Label(content, text="", style="Meta.TLabel", wraplength=584, anchor=tk.CENTER)
        self._example_cn_label.grid(row=4, column=0, sticky=tk.EW, pady=(0, 16))

        controls = tk.Frame(review, bg=GLASS_FILL, bd=0, highlightthickness=0)
        self._actions_frame = controls
        controls.place(x=28, y=352, width=584, height=120)
        self._unknown_button = self._button(controls, text="不会", variant="danger", command=self._mark_unknown)
        self._unknown_button.place(x=0, y=0, width=288, height=32)
        self._unknown_placeholder = ttk.Label(controls, text=" ", style="ButtonPlaceholder.TLabel")
        self._unknown_placeholder.place(x=0, y=0, width=288, height=32)
        self._unknown_placeholder.place_forget()
        self._known_button = self._button(controls, text="会", variant="primary", command=self._mark_known)
        self._known_button.place(x=296, y=0, width=288, height=32)
        self._known_placeholder = ttk.Label(controls, text=" ", style="ButtonPlaceholder.TLabel")
        self._known_placeholder.place(x=296, y=0, width=288, height=32)
        self._known_placeholder.place_forget()
        self._definitely_known_button = GlassButton(
            controls,
            text="绝对会",
            variant="primary",
            command=self._mark_definitely_known,
        )
        self._definitely_known_button.place(x=148, y=88, width=288, height=32)
        self._definitely_known_placeholder = ttk.Label(controls, text=" ", style="ButtonPlaceholder.TLabel")
        self._definitely_known_placeholder.place(x=148, y=88, width=288, height=32)
        self._definitely_known_placeholder.place_forget()
        self._navigation_frame = controls
        self._previous_button = self._button(controls, text="上一个", command=self._show_previous_word)
        self._previous_button.place(x=0, y=44, width=288, height=32)
        self._previous_placeholder = ttk.Label(controls, text=" ", style="ButtonPlaceholder.TLabel")
        self._previous_placeholder.place(x=0, y=44, width=288, height=32)
        self._previous_placeholder.place_forget()
        self._next_button = self._button(controls, text="下一个", command=self._continue_from_history)
        self._next_button.place(x=296, y=44, width=288, height=32)
        self._next_placeholder = ttk.Label(controls, text=" ", style="ButtonPlaceholder.TLabel")
        self._next_placeholder.place(x=296, y=44, width=288, height=32)
        self._next_placeholder.place_forget()

        self._review_meta_label = ttk.Label(review, text="", style="Meta.TLabel", wraplength=584, anchor=tk.CENTER)
        self._review_meta_label.place(x=28, y=486, width=584, height=38)

    def _show_word_list(self) -> None:
        self._mode = "words"
        self._selected_id = None
        self._selected_ids = []
        self._study_session.reset()
        self._clear_root()

        container = ttk.Frame(self._root, padding=24, style="App.TFrame")
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        title_bar = ttk.Frame(container, style="App.TFrame")
        title_bar.grid(row=0, column=0, sticky=tk.EW, pady=(0, 14))
        title_bar.columnconfigure(0, weight=1)
        ttk.Label(title_bar, text="词表", style="HomeTitle.TLabel").grid(row=0, column=0, sticky=tk.W)
        self._button(title_bar, text="返回主页", command=self._show_home).grid(row=0, column=1, sticky=tk.E)

        list_panel = GlassPanel(container, padding=18, radius=28)
        list_panel.grid(row=1, column=0, sticky=tk.NSEW)
        list_area = list_panel.body
        list_area.columnconfigure(0, weight=1)
        list_area.rowconfigure(2, weight=1)

        search = ttk.Frame(list_area, style="Toolbar.TFrame")
        search.grid(row=0, column=0, sticky=tk.EW, pady=(0, 10))
        search.columnconfigure(1, weight=1)
        ttk.Label(search, text="搜索").grid(row=0, column=0, padx=(0, 8))
        search_entry = ttk.Entry(search, textvariable=self._search_var)
        search_entry.grid(row=0, column=1, sticky=tk.EW)
        search_entry.bind("<KeyRelease>", lambda _event: self._refresh_words(reload_current=False))

        scope = ttk.Frame(list_area, style="Toolbar.TFrame")
        scope.grid(row=1, column=0, sticky=tk.EW, pady=(0, 10))
        scope.columnconfigure(1, weight=1)
        ttk.Label(scope, text="AI范围").grid(row=0, column=0, padx=(0, 8))
        ttk.Entry(scope, textvariable=self._ai_scope_var).grid(row=0, column=1, sticky=tk.EW)

        columns = (
            "index",
            "word",
            "meaning",
            "forms",
            "example_sentence",
            "example_sentence_cn",
            "frequency",
            "status",
            "stats",
        )
        self._tree = ttk.Treeview(list_area, columns=columns, show="headings", selectmode="extended")
        headings = {
            "index": "#",
            "word": "单词",
            "meaning": "释义",
            "forms": "词形",
            "example_sentence": "例句",
            "example_sentence_cn": "例句中文",
            "frequency": "频率",
            "status": "状态",
            "stats": "复习",
        }
        for column, text in headings.items():
            self._tree.heading(column, text=text)

        self._tree.column("index", width=56, minwidth=48, anchor=tk.E)
        self._tree.column("word", width=150, minwidth=120)
        self._tree.column("meaning", width=230, minwidth=160)
        self._tree.column("forms", width=190, minwidth=140)
        self._tree.column("example_sentence", width=220, minwidth=160)
        self._tree.column("example_sentence_cn", width=180, minwidth=140)
        self._tree.column("frequency", width=70, minwidth=60, anchor=tk.E)
        self._tree.column("status", width=90, minwidth=80)
        self._tree.column("stats", width=90, minwidth=80)
        self._tree.grid(row=2, column=0, sticky=tk.NSEW)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)
        self._tree.bind("<Double-1>", self._edit_word_from_tree)
        self._tree.bind("<Control-a>", self._select_all_visible_words)
        self._tree.bind("<Control-A>", self._select_all_visible_words)

        scrollbar = ttk.Scrollbar(list_area, orient=tk.VERTICAL, command=self._tree.yview)
        scrollbar.grid(row=2, column=1, sticky=tk.NS)
        self._tree.configure(yscrollcommand=scrollbar.set)

        footer = ttk.Frame(list_area, style="Toolbar.TFrame")
        footer.grid(row=3, column=0, sticky=tk.EW, pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        self._count_label = ttk.Label(footer, text="")
        self._count_label.grid(row=0, column=0, sticky=tk.W)
        self._supplement_example_button = self._button(
            footer,
            text="智能补充选中",
            command=self._supplement_selected_example,
        )
        self._supplement_example_button.grid(row=0, column=1, padx=(0, 8))
        self._correct_entry_button = self._button(
            footer,
            text="智能修正选中",
            command=self._correct_selected_entry,
        )
        self._correct_entry_button.grid(row=0, column=2, padx=(0, 8))
        self._pause_batch_button = self._button(
            footer,
            text="暂停",
            command=self._toggle_batch_pause,
            state=tk.DISABLED,
        )
        self._pause_batch_button.grid(row=0, column=3, padx=(0, 8))
        self._stop_batch_button = self._button(
            footer,
            text="停止",
            command=self._stop_batch,
            state=tk.DISABLED,
        )
        self._stop_batch_button.grid(row=0, column=4, padx=(0, 8))
        self._button(footer, text="删除选中", variant="danger", command=self._delete_selected).grid(row=0, column=5)
        self._batch_progress = ttk.Progressbar(
            footer,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=100,
        )
        self._batch_progress.grid(row=1, column=0, columnspan=6, sticky=tk.EW, pady=(8, 0))
        self._batch_status_label = ttk.Label(footer, text="", style="Meta.TLabel")
        self._batch_status_label.grid(row=2, column=0, columnspan=6, sticky=tk.W, pady=(4, 0))
        self._refresh_words(reload_current=False)

    def _reload_review_entries(self, selected_id: str | None = None) -> None:
        self._render_study_card(self._study_session.reload(selected_id))

    def _render_study_card(self, state: StudyCardState) -> None:
        self._set_review_controls(
            history_view=state.history_view,
            has_entry=state.has_entry,
            can_show_previous=state.can_show_previous,
        )
        self._word_label.configure(text=state.word_text)
        self._meaning_label.configure(text=state.meaning_text)
        self._forms_label.configure(text=state.forms_text)
        self._example_label.configure(text=state.example_text)
        self._example_cn_label.configure(text=state.example_cn_text)
        self._review_meta_label.configure(text=state.meta_text)

    def _show_previous_word(self) -> None:
        state = self._study_session.show_previous_word()
        if state is not None:
            self._render_study_card(state)

    def _continue_from_history(self) -> None:
        self._render_study_card(self._study_session.continue_from_history())

    def _mark_known(self) -> None:
        self._mark_current("known")

    def _mark_definitely_known(self) -> None:
        self._mark_current("definitely_known")

    def _mark_unknown(self) -> None:
        self._mark_current("unknown")

    def _mark_current(self, result: str) -> None:
        state = self._study_session.mark_current(result)  # type: ignore[arg-type]
        if state is None:
            messagebox.showinfo("暂无词条", "当前没有可复习的词条。", parent=self._root)
            return
        self._render_study_card(state)

    def _delete_selected(self) -> None:
        selected_ids = self._selected_entry_ids()
        if not selected_ids:
            messagebox.showinfo("未选择词条", "请先在列表中选择一个词条。", parent=self._root)
            return

        for entry_id in selected_ids:
            self._service.delete_word(entry_id)
        self._selected_id = None
        self._selected_ids = []
        self._refresh_words(reload_current=self._mode != "words")
        if self._mode != "words":
            self._reload_review_entries()

    def _correct_selected_entry(self) -> None:
        if self._is_correcting_entry:
            return

        selected_ids = self._selected_entry_ids()
        if not selected_ids:
            messagebox.showinfo("未选择词条", "请先在列表中选择一个词条。", parent=self._root)
            return

        if self._example_generator is None:
            messagebox.showinfo("无法智能修正", "未配置本地模型生成器。", parent=self._root)
            return

        entries = self._entries_by_id(selected_ids)
        if not entries:
            messagebox.showinfo("智能修正失败", "当前词条不存在。", parent=self._root)
            return

        scope = self._ai_scope_var.get().strip()
        self._is_correcting_entry = True
        self._set_batch_running()
        self._set_correct_entry_button(text=f"修正中 0/{len(entries)}", state=tk.DISABLED)
        self._set_batch_progress(
            action="修正",
            done=0,
            total=len(entries),
            workers=0,
            elapsed_seconds=0.0,
        )
        thread = threading.Thread(
            target=self._correct_entry_worker,
            args=(entries, scope),
            daemon=True,
        )
        thread.start()

    def _correct_entry_worker(self, entries: list[WordEntry], scope: str) -> None:
        updated_ids: list[str] = []
        errors: list[str] = []
        total = len(entries)
        started_at = time.monotonic()

        def progress(done: int, count: int, workers: int) -> None:
            elapsed = time.monotonic() - started_at
            self._root.after(
                0,
                lambda done=done, count=count, workers=workers, elapsed=elapsed: (
                    self._set_correct_entry_button(
                        text=f"修正中 {done}/{count} 并行 {workers}",
                        state=tk.DISABLED,
                    ),
                    self._set_batch_progress(
                        action="修正",
                        done=done,
                        total=count,
                        workers=workers,
                        elapsed_seconds=elapsed,
                    ),
                ),
            )

        try:
            if hasattr(self._example_generator, "correct_many"):
                results, errors, _workers = self._example_generator.correct_many(
                    entries,
                    scope=scope,
                    progress=progress,
                    control=self._batch_control_state,
                )
            else:
                results = []
                for index, entry in enumerate(entries, start=1):
                    if not self._wait_for_batch_resume():
                        break
                    progress(index - 1, total, 1)
                    results.append((entry, self._example_generator.correct_entry(entry, scope)))
                progress(total, total, 1)
        except Exception as error:
            message = str(error)
            self._root.after(0, lambda: self._on_correct_entry_failed(message))
            return

        for entry, corrected in results:
            try:
                updated = self._service.update_text(
                    entry.id,
                    corrected.corrected_word,
                    entry.meaning,
                    entry.forms,
                )
                if updated is not None:
                    updated_ids.append(updated.id)
            except Exception as error:
                errors.append(f"{entry.word}: {error}")
        self._root.after(
            0,
            lambda: self._on_correct_entries_finished(updated_ids, errors, total),
        )

    def _on_correct_entries_finished(
        self,
        updated_ids: list[str],
        errors: list[str],
        total: int,
    ) -> None:
        self._is_correcting_entry = False
        self._set_batch_idle()
        self._set_correct_entry_button(text="智能修正选中", state=tk.NORMAL)
        self._finish_batch_progress("修正", len(updated_ids), total, len(errors))
        self._selected_ids = updated_ids
        self._selected_id = updated_ids[0] if updated_ids else None
        self._refresh_words(reload_current=False)
        self._restore_tree_selection(updated_ids)

        message = f"已修正 {len(updated_ids)} / {total} 条。"
        if errors:
            message = f"{message}\n失败 {len(errors)} 条：\n" + "\n".join(errors[:5])
        messagebox.showinfo("智能修正完成", message, parent=self._root)

    def _on_correct_entry_failed(self, message: str) -> None:
        self._is_correcting_entry = False
        self._set_batch_idle()
        self._set_correct_entry_button(text="智能修正选中", state=tk.NORMAL)
        self._clear_batch_progress("修正失败")
        messagebox.showerror("智能修正失败", message, parent=self._root)

    def _on_select(self, _event: tk.Event) -> None:
        selected = list(self._tree.selection())
        self._selected_ids = selected
        self._selected_id = selected[0] if selected else None
        if hasattr(self, "_count_label"):
            total = len(self._visible_word_entries())
            visible = len(self._tree.get_children())
            selected_text = f" | 已选 {len(selected)} 条" if selected else ""
            self._count_label.configure(text=f"显示 {visible} / 共 {total} 条{selected_text}")
        if self._selected_id and self._mode != "words":
            self._reload_review_entries(self._selected_id)

    def _select_all_visible_words(self, _event: tk.Event) -> str:
        if not hasattr(self, "_tree"):
            return "break"

        visible_ids = list(self._tree.get_children())
        self._tree.selection_set(visible_ids)
        self._selected_ids = visible_ids
        self._selected_id = visible_ids[0] if visible_ids else None
        if visible_ids:
            self._tree.focus(visible_ids[0])
        if hasattr(self, "_count_label"):
            total = len(self._visible_word_entries())
            self._count_label.configure(
                text=f"显示 {len(visible_ids)} / 共 {total} 条 | 已选 {len(visible_ids)} 条"
            )
        return "break"

    def _edit_word_from_tree(self, event: tk.Event) -> str:
        if not hasattr(self, "_tree"):
            return "break"

        row_id = self._tree.identify_row(event.y)
        if not row_id:
            return "break"

        entry = self._service.get_word(row_id)
        if entry is None:
            messagebox.showinfo("编辑失败", "当前词条不存在。", parent=self._root)
            return "break"

        self._show_word_editor(entry)
        return "break"

    def _show_word_editor(self, entry: WordEntry) -> None:
        dialog = tk.Toplevel(self._root)
        dialog.title(f"编辑词条 - {entry.word}")
        dialog.transient(self._root)
        dialog.grab_set()
        dialog.geometry("640x460")
        dialog.minsize(560, 420)
        dialog.configure(bg=APP_BG)

        panel = GlassPanel(dialog, padding=20, radius=28)
        panel.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)
        form = panel.body
        form.columnconfigure(1, weight=1)
        form.rowconfigure(4, weight=1)
        form.rowconfigure(5, weight=1)

        word_var = tk.StringVar(value=entry.word)
        meaning_var = tk.StringVar(value=entry.meaning)
        forms_var = tk.StringVar(value=entry.forms)

        ttk.Label(form, text="单词").grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=(0, 10))
        word_entry = ttk.Entry(form, textvariable=word_var)
        word_entry.grid(row=0, column=1, sticky=tk.EW, pady=(0, 10))

        ttk.Label(form, text="释义").grid(row=1, column=0, sticky=tk.W, padx=(0, 8), pady=(0, 10))
        ttk.Entry(form, textvariable=meaning_var).grid(row=1, column=1, sticky=tk.EW, pady=(0, 10))

        ttk.Label(form, text="词形").grid(row=2, column=0, sticky=tk.W, padx=(0, 8), pady=(0, 10))
        ttk.Entry(form, textvariable=forms_var).grid(row=2, column=1, sticky=tk.EW, pady=(0, 10))

        ttk.Label(form, text="例句").grid(row=3, column=0, sticky=tk.NW, padx=(0, 8), pady=(0, 10))
        example_text = tk.Text(
            form,
            height=5,
            wrap=tk.WORD,
            bg="#ffffff",
            fg=TEXT_PRIMARY,
            insertbackground=ACCENT,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=GLASS_BORDER,
            highlightcolor=ACCENT,
        )
        example_text.grid(row=3, column=1, sticky=tk.NSEW, pady=(0, 10))
        example_text.insert("1.0", entry.example_sentence)

        ttk.Label(form, text="例句中文").grid(row=4, column=0, sticky=tk.NW, padx=(0, 8), pady=(0, 10))
        example_cn_text = tk.Text(
            form,
            height=5,
            wrap=tk.WORD,
            bg="#ffffff",
            fg=TEXT_PRIMARY,
            insertbackground=ACCENT,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=GLASS_BORDER,
            highlightcolor=ACCENT,
        )
        example_cn_text.grid(row=4, column=1, sticky=tk.NSEW, pady=(0, 10))
        example_cn_text.insert("1.0", entry.example_sentence_cn)

        actions = ttk.Frame(form, style="Toolbar.TFrame")
        actions.grid(row=5, column=0, columnspan=2, sticky=tk.E, pady=(8, 0))

        def save() -> None:
            word = word_var.get().strip()
            meaning = meaning_var.get().strip()
            forms = forms_var.get().strip()
            example_sentence = example_text.get("1.0", tk.END).strip()
            example_sentence_cn = example_cn_text.get("1.0", tk.END).strip()

            if not word:
                messagebox.showinfo("无法保存", "单词不能为空。", parent=dialog)
                return
            if not meaning:
                messagebox.showinfo("无法保存", "释义不能为空。", parent=dialog)
                return

            try:
                updated = self._service.update_text(entry.id, word, meaning, forms)
                if updated is None:
                    messagebox.showinfo("保存失败", "当前词条不存在。", parent=dialog)
                    return
                updated = self._service.update_examples(
                    entry.id,
                    example_sentence,
                    example_sentence_cn,
                )
            except Exception as error:
                messagebox.showerror("保存失败", str(error), parent=dialog)
                return

            self._selected_id = updated.id if updated else entry.id
            self._selected_ids = [self._selected_id]
            self._refresh_words(reload_current=False)
            self._restore_tree_selection(self._selected_ids)
            dialog.destroy()

        self._button(actions, text="取消", command=dialog.destroy).grid(row=0, column=0, padx=(0, 8))
        self._button(actions, text="保存", variant="primary", command=save).grid(row=0, column=1)

        word_entry.focus_set()
        dialog.bind("<Control-s>", lambda _event: (save(), "break"))
        dialog.bind("<Escape>", lambda _event: (dialog.destroy(), "break"))

    def _refresh_words(self, reload_current: bool = True) -> None:
        if not hasattr(self, "_tree"):
            return

        for item in self._tree.get_children():
            self._tree.delete(item)

        entries = self._visible_word_entries(self._search_var.get())
        for entry in entries:
            self._insert_entry(entry)

        total = len(self._visible_word_entries())
        selected_count = len(self._selected_entry_ids())
        selected_text = f" | 已选 {selected_count} 条" if selected_count else ""
        self._count_label.configure(text=f"显示 {len(entries)} / 共 {total} 条{selected_text}")
        if reload_current and self._mode != "words":
            current = self._study_session.current_entry
            self._reload_review_entries(current.id if current else None)

    def _set_review_controls(self, history_view: bool, has_entry: bool, can_show_previous: bool) -> None:
        controls = (
            self._unknown_button,
            self._unknown_placeholder,
            self._known_button,
            self._known_placeholder,
            self._definitely_known_button,
            self._definitely_known_placeholder,
            self._previous_button,
            self._previous_placeholder,
            self._next_button,
            self._next_placeholder,
        )
        for control in controls:
            control.place_forget()

        if not has_entry:
            self._place_action_control(self._unknown_placeholder, 0)
            self._place_action_control(self._known_placeholder, 1)
            self._place_action_control(self._definitely_known_placeholder, 2)
            if can_show_previous:
                self._place_navigation_control(self._previous_button, 0)
            else:
                self._place_navigation_control(self._previous_placeholder, 0)
            self._place_navigation_control(self._next_placeholder, 1)
            return

        if history_view:
            self._place_action_control(self._unknown_placeholder, 0)
            self._place_action_control(self._known_placeholder, 1)
            self._place_action_control(self._definitely_known_placeholder, 2)
        else:
            self._place_action_control(self._unknown_button, 0)
            self._place_action_control(self._known_button, 1)
            if self._mode == "learning":
                self._place_action_control(self._definitely_known_button, 2)
            else:
                self._place_action_control(self._definitely_known_placeholder, 2)

        if can_show_previous:
            self._place_navigation_control(self._previous_button, 0)
        else:
            self._place_navigation_control(self._previous_placeholder, 0)

        if history_view:
            self._place_navigation_control(self._next_button, 1)
        else:
            self._place_navigation_control(self._next_placeholder, 1)

    @staticmethod
    def _place_action_control(control: tk.Widget, slot: int) -> None:
        positions = {
            0: (0, 0, 288, 32),
            1: (296, 0, 288, 32),
            2: (148, 88, 288, 32),
        }
        x, y, width, height = positions[slot]
        control.place(x=x, y=y, width=width, height=height)

    @staticmethod
    def _place_navigation_control(control: tk.Widget, slot: int) -> None:
        control.place(x=slot * 296, y=44, width=288, height=32)

    def _supplement_selected_example(self) -> None:
        if self._is_generating_example:
            return

        selected_ids = self._selected_entry_ids()
        if not selected_ids:
            messagebox.showinfo("未选择词条", "请先在列表中选择一个词条。", parent=self._root)
            return

        if self._example_generator is None:
            messagebox.showinfo("无法智能补充", "未配置本地模型生成器。", parent=self._root)
            return

        entries = self._entries_by_id(selected_ids)
        if not entries:
            messagebox.showinfo("智能补充失败", "当前词条不存在。", parent=self._root)
            return

        scope = self._ai_scope_var.get().strip()
        self._is_generating_example = True
        self._set_batch_running()
        self._set_supplement_example_button(text=f"补充中 0/{len(entries)}", state=tk.DISABLED)
        self._set_batch_progress(
            action="补充",
            done=0,
            total=len(entries),
            workers=0,
            elapsed_seconds=0.0,
        )

        thread = threading.Thread(
            target=self._generate_example_worker,
            args=(entries, scope),
            daemon=True,
        )
        thread.start()

    def _generate_example_worker(self, entries: list[WordEntry], scope: str) -> None:
        updated_ids: list[str] = []
        errors: list[str] = []
        total = len(entries)
        started_at = time.monotonic()

        def progress(done: int, count: int, workers: int) -> None:
            elapsed = time.monotonic() - started_at
            self._root.after(
                0,
                lambda done=done, count=count, workers=workers, elapsed=elapsed: (
                    self._set_supplement_example_button(
                        text=f"补充中 {done}/{count} 并行 {workers}",
                        state=tk.DISABLED,
                    ),
                    self._set_batch_progress(
                        action="补充",
                        done=done,
                        total=count,
                        workers=workers,
                        elapsed_seconds=elapsed,
                    ),
                ),
            )

        try:
            if hasattr(self._example_generator, "generate_many"):
                results, errors, _workers = self._example_generator.generate_many(
                    entries,
                    scope=scope,
                    progress=progress,
                    control=self._batch_control_state,
                )
            else:
                results = []
                for index, entry in enumerate(entries, start=1):
                    if not self._wait_for_batch_resume():
                        break
                    progress(index - 1, total, 1)
                    results.append((entry, self._example_generator.generate(entry, scope)))
                progress(total, total, 1)
        except Exception as error:
            message = str(error)
            self._root.after(0, lambda: self._on_generate_example_failed(message))
            return

        for entry, generated in results:
            try:
                updated = self._service.update_examples(
                    entry.id,
                    generated.example_sentence,
                    generated.example_sentence_cn,
                )
                if updated is not None:
                    updated_ids.append(updated.id)
            except Exception as error:
                errors.append(f"{entry.word}: {error}")
        self._root.after(
            0,
            lambda: self._on_generate_examples_finished(updated_ids, errors, total),
        )

    def _on_generate_examples_finished(
        self,
        updated_ids: list[str],
        errors: list[str],
        total: int,
    ) -> None:
        self._is_generating_example = False
        self._set_batch_idle()
        self._set_supplement_example_button(text="智能补充选中", state=tk.NORMAL)
        self._finish_batch_progress("补充", len(updated_ids), total, len(errors))
        self._selected_ids = updated_ids
        self._selected_id = updated_ids[0] if updated_ids else None
        self._refresh_words(reload_current=False)
        self._restore_tree_selection(updated_ids)

        message = f"已补充 {len(updated_ids)} / {total} 条。"
        if errors:
            message = f"{message}\n失败 {len(errors)} 条：\n" + "\n".join(errors[:5])
        messagebox.showinfo("智能补充完成", message, parent=self._root)

    def _on_generate_example_failed(self, message: str) -> None:
        self._is_generating_example = False
        self._set_batch_idle()
        self._set_supplement_example_button(text="智能补充选中", state=tk.NORMAL)
        self._clear_batch_progress("补充失败")
        messagebox.showerror("智能补充失败", message, parent=self._root)

    def _set_supplement_example_button(self, text: str, state: str) -> None:
        if hasattr(self, "_supplement_example_button"):
            self._supplement_example_button.configure(text=text, state=state)

    def _set_correct_entry_button(self, text: str, state: str) -> None:
        if hasattr(self, "_correct_entry_button"):
            self._correct_entry_button.configure(text=text, state=state)

    def _set_batch_running(self) -> None:
        self._batch_state = "running"
        if hasattr(self, "_pause_batch_button"):
            self._pause_batch_button.configure(text="暂停", state=tk.NORMAL)
        if hasattr(self, "_stop_batch_button"):
            self._stop_batch_button.configure(state=tk.NORMAL)

    def _set_batch_idle(self) -> None:
        self._batch_state = "idle"
        if hasattr(self, "_pause_batch_button"):
            self._pause_batch_button.configure(text="暂停", state=tk.DISABLED)
        if hasattr(self, "_stop_batch_button"):
            self._stop_batch_button.configure(state=tk.DISABLED)

    def _toggle_batch_pause(self) -> None:
        if self._batch_state == "running":
            self._batch_state = "paused"
            if hasattr(self, "_pause_batch_button"):
                self._pause_batch_button.configure(text="继续")
            if hasattr(self, "_batch_status_label"):
                current_text = self._batch_status_label.cget("text")
                self._batch_status_label.configure(text=f"{current_text} | 已暂停")
            return

        if self._batch_state == "paused":
            self._batch_state = "running"
            if hasattr(self, "_pause_batch_button"):
                self._pause_batch_button.configure(text="暂停")

    def _stop_batch(self) -> None:
        if self._batch_state in {"running", "paused"}:
            self._batch_state = "stopped"
            if hasattr(self, "_pause_batch_button"):
                self._pause_batch_button.configure(text="暂停", state=tk.DISABLED)
            if hasattr(self, "_stop_batch_button"):
                self._stop_batch_button.configure(state=tk.DISABLED)
            if hasattr(self, "_batch_status_label"):
                current_text = self._batch_status_label.cget("text")
                self._batch_status_label.configure(text=f"{current_text} | 正在停止")

    def _batch_control_state(self) -> str:
        return self._batch_state

    def _wait_for_batch_resume(self) -> bool:
        while self._batch_state == "paused":
            time.sleep(0.2)
        return self._batch_state != "stopped"

    def _set_batch_progress(
        self,
        action: str,
        done: int,
        total: int,
        workers: int,
        elapsed_seconds: float,
    ) -> None:
        if not hasattr(self, "_batch_progress") or not hasattr(self, "_batch_status_label"):
            return

        percent = (done / total * 100) if total else 0
        self._batch_progress.configure(value=percent)
        if done <= 0:
            eta_text = "预估剩余：估算中"
        else:
            average_seconds = elapsed_seconds / done
            remaining_seconds = max(0.0, average_seconds * (total - done))
            eta_text = f"预估剩余：{self._format_duration(remaining_seconds)}"

        worker_text = f"并行 {workers}" if workers else "准备中"
        self._batch_status_label.configure(
            text=(
                f"{action}进度：{done} / {total} "
                f"({percent:.0f}%) | {worker_text} | {eta_text}"
            )
        )

    def _finish_batch_progress(
        self,
        action: str,
        success_count: int,
        total: int,
        error_count: int,
    ) -> None:
        if not hasattr(self, "_batch_progress") or not hasattr(self, "_batch_status_label"):
            return

        self._batch_progress.configure(value=100 if total else 0)
        self._batch_status_label.configure(
            text=f"{action}完成：成功 {success_count} / {total}，失败 {error_count}。"
        )

    def _clear_batch_progress(self, message: str = "") -> None:
        if hasattr(self, "_batch_progress"):
            self._batch_progress.configure(value=0)
        if hasattr(self, "_batch_status_label"):
            self._batch_status_label.configure(text=message)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        minutes, remaining_seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}小时{minutes}分"
        if minutes:
            return f"{minutes}分{remaining_seconds}秒"
        return f"{remaining_seconds}秒"

    def _selected_entry_ids(self) -> list[str]:
        if hasattr(self, "_tree"):
            selected = list(self._tree.selection())
            if selected:
                return selected
        return list(self._selected_ids)

    def _entries_by_id(self, entry_ids: list[str]) -> list[WordEntry]:
        entries: list[WordEntry] = []
        for entry_id in entry_ids:
            entry = self._service.get_word(entry_id)
            if entry is not None:
                entries.append(entry)
        return entries

    def _restore_tree_selection(self, entry_ids: list[str]) -> None:
        if not hasattr(self, "_tree") or not entry_ids:
            return

        existing_ids = [
            entry_id
            for entry_id in entry_ids
            if self._tree.exists(entry_id)
        ]
        if not existing_ids:
            return

        self._tree.selection_set(existing_ids)
        self._tree.focus(existing_ids[0])
        self._tree.see(existing_ids[0])
        self._selected_ids = existing_ids
        self._selected_id = existing_ids[0]

    def _visible_word_entries(self, query: str = "") -> list[WordEntry]:
        if self._mode in {"learning", "review"}:
            return self._study_session.mode_entries(query)
        return self._service.list_words(query)

    def _insert_entry(self, entry: WordEntry) -> None:
        self._tree.insert(
            "",
            tk.END,
            iid=entry.id,
            values=(
                entry.source_index,
                entry.word,
                entry.meaning,
                entry.forms,
                entry.example_sentence,
                entry.example_sentence_cn,
                entry.frequency,
                entry.status,
                f"{entry.correct_count}/{entry.wrong_count}",
            ),
        )
