import tkinter as tk
from tkinter import messagebox, ttk

from wordpycket.application.services import WordService
from wordpycket.domain.entities import WordEntry


class WordPycketApp:
    def __init__(self, service: WordService) -> None:
        self._service = service
        self._selected_id: str | None = None
        self._current_review_entry: WordEntry | None = None
        self._review_entries: list[WordEntry] = []
        self._review_index = 0

        self._root = tk.Tk()
        self._root.title("WordPycket")
        self._root.geometry("1040x640")
        self._root.minsize(920, 560)

        self._search_var = tk.StringVar()

        self._configure_style()
        self._build_layout()
        self._refresh_words()
        self._reload_review_entries()

    def run(self) -> None:
        self._root.mainloop()

    def _configure_style(self) -> None:
        style = ttk.Style(self._root)
        style.theme_use("clam")
        style.configure("Treeview", rowheight=28)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Word.TLabel", font=("Segoe UI", 34, "bold"))
        style.configure("Meaning.TLabel", font=("Microsoft YaHei UI", 18))
        style.configure("Meta.TLabel", foreground="#555555")

    def _build_layout(self) -> None:
        container = ttk.Frame(self._root, padding=18)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=0)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(1, weight=1)

        title = ttk.Label(container, text="WordPycket", style="Title.TLabel")
        title.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 14))

        review = ttk.LabelFrame(container, text="复习", padding=18)
        review.grid(row=1, column=0, sticky=tk.NS, padx=(0, 16))
        review.columnconfigure(0, weight=1)

        self._word_label = ttk.Label(review, text="", style="Word.TLabel", wraplength=280)
        self._word_label.grid(row=0, column=0, sticky=tk.EW, pady=(8, 12))

        self._meaning_label = ttk.Label(review, text="", style="Meaning.TLabel", wraplength=280)
        self._meaning_label.grid(row=1, column=0, sticky=tk.EW, pady=(0, 12))

        self._forms_label = ttk.Label(review, text="", style="Meta.TLabel", wraplength=280)
        self._forms_label.grid(row=2, column=0, sticky=tk.EW, pady=(0, 16))

        actions = ttk.Frame(review)
        actions.grid(row=3, column=0, sticky=tk.EW)
        actions.columnconfigure((0, 1), weight=1)
        ttk.Button(actions, text="不会", command=self._mark_unknown).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(actions, text="会", command=self._mark_known).grid(row=0, column=1, sticky=tk.EW, padx=(8, 0))

        navigation = ttk.Frame(review)
        navigation.grid(row=4, column=0, sticky=tk.EW, pady=(8, 0))
        navigation.columnconfigure((0, 1), weight=1)
        ttk.Button(navigation, text="上一个", command=self._show_previous_word).grid(
            row=0,
            column=0,
            sticky=tk.EW,
        )
        ttk.Button(navigation, text="下一个", command=self._show_next_word).grid(
            row=0,
            column=1,
            sticky=tk.EW,
            padx=(8, 0),
        )

        self._review_meta_label = ttk.Label(review, text="", style="Meta.TLabel", wraplength=280)
        self._review_meta_label.grid(row=5, column=0, sticky=tk.EW, pady=(18, 0))

        list_area = ttk.Frame(container)
        list_area.grid(row=1, column=1, sticky=tk.NSEW)
        list_area.columnconfigure(0, weight=1)
        list_area.rowconfigure(1, weight=1)

        search = ttk.Frame(list_area)
        search.grid(row=0, column=0, sticky=tk.EW, pady=(0, 10))
        search.columnconfigure(1, weight=1)
        ttk.Label(search, text="搜索").grid(row=0, column=0, padx=(0, 8))
        search_entry = ttk.Entry(search, textvariable=self._search_var)
        search_entry.grid(row=0, column=1, sticky=tk.EW)
        search_entry.bind("<KeyRelease>", lambda _event: self._refresh_words())

        columns = ("index", "word", "meaning", "frequency", "status", "stats")
        self._tree = ttk.Treeview(list_area, columns=columns, show="headings", selectmode="browse")
        headings = {
            "index": "#",
            "word": "单词",
            "meaning": "释义",
            "frequency": "频率",
            "status": "状态",
            "stats": "复习",
        }
        for column, text in headings.items():
            self._tree.heading(column, text=text)

        self._tree.column("index", width=56, minwidth=48, anchor=tk.E)
        self._tree.column("word", width=150, minwidth=120)
        self._tree.column("meaning", width=230, minwidth=160)
        self._tree.column("frequency", width=70, minwidth=60, anchor=tk.E)
        self._tree.column("status", width=90, minwidth=80)
        self._tree.column("stats", width=90, minwidth=80)
        self._tree.grid(row=1, column=0, sticky=tk.NSEW)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        scrollbar = ttk.Scrollbar(list_area, orient=tk.VERTICAL, command=self._tree.yview)
        scrollbar.grid(row=1, column=1, sticky=tk.NS)
        self._tree.configure(yscrollcommand=scrollbar.set)

        footer = ttk.Frame(list_area)
        footer.grid(row=2, column=0, sticky=tk.EW, pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        self._count_label = ttk.Label(footer, text="")
        self._count_label.grid(row=0, column=0, sticky=tk.W)
        ttk.Button(footer, text="删除选中", command=self._delete_selected).grid(row=0, column=1)

    def _reload_review_entries(self, selected_id: str | None = None) -> None:
        self._review_entries = self._service.list_words(self._search_var.get())
        if not self._review_entries:
            self._show_empty_review_card()
            return

        if selected_id is not None:
            for index, entry in enumerate(self._review_entries):
                if entry.id == selected_id:
                    self._review_index = index
                    break
            else:
                self._review_index = min(self._review_index, len(self._review_entries) - 1)
        else:
            self._review_index = min(self._review_index, len(self._review_entries) - 1)

        self._show_review_entry(self._review_entries[self._review_index])

    def _show_empty_review_card(self) -> None:
        self._current_review_entry = None
        self._word_label.configure(text="暂无词条")
        self._meaning_label.configure(text="")
        self._forms_label.configure(text="")
        self._review_meta_label.configure(text="请确认 CSV 已放在 input/word_frequency.csv。")

    def _show_review_entry(self, entry: WordEntry) -> None:
        self._current_review_entry = entry
        self._word_label.configure(text=entry.word)
        self._meaning_label.configure(text=entry.meaning)
        self._forms_label.configure(text=f"词形: {entry.forms}" if entry.forms else "")
        self._review_meta_label.configure(
            text=(
                f"{self._review_index + 1} / {len(self._review_entries)} | "
                f"频率 {entry.frequency} | {entry.status} | "
                f"会 {entry.correct_count} / 不会 {entry.wrong_count}"
            )
        )

    def _show_previous_word(self) -> None:
        if not self._review_entries:
            return

        self._review_index = (self._review_index - 1) % len(self._review_entries)
        self._show_review_entry(self._review_entries[self._review_index])

    def _show_next_word(self) -> None:
        if not self._review_entries:
            return

        self._review_index = (self._review_index + 1) % len(self._review_entries)
        self._show_review_entry(self._review_entries[self._review_index])

    def _mark_known(self) -> None:
        self._mark_current(known=True)

    def _mark_unknown(self) -> None:
        self._mark_current(known=False)

    def _mark_current(self, known: bool) -> None:
        if self._current_review_entry is None:
            messagebox.showinfo("暂无词条", "当前没有可复习的词条。", parent=self._root)
            return

        if known:
            self._service.mark_known(self._current_review_entry.id)
        else:
            self._service.mark_unknown(self._current_review_entry.id)

        self._refresh_words()
        self._reload_review_entries(self._current_review_entry.id)

    def _delete_selected(self) -> None:
        if not self._selected_id:
            messagebox.showinfo("未选择词条", "请先在列表中选择一个词条。", parent=self._root)
            return

        self._service.delete_word(self._selected_id)
        self._selected_id = None
        self._refresh_words()
        self._reload_review_entries()

    def _on_select(self, _event: tk.Event) -> None:
        selected = self._tree.selection()
        self._selected_id = selected[0] if selected else None
        if self._selected_id:
            self._reload_review_entries(self._selected_id)

    def _refresh_words(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)

        entries = self._service.list_words(self._search_var.get())
        for entry in entries:
            self._insert_entry(entry)

        total = len(self._service.list_words())
        self._count_label.configure(text=f"显示 {len(entries)} / 共 {total} 条")
        self._reload_review_entries(self._current_review_entry.id if self._current_review_entry else None)

    def _insert_entry(self, entry: WordEntry) -> None:
        self._tree.insert(
            "",
            tk.END,
            iid=entry.id,
            values=(
                entry.source_index,
                entry.word,
                entry.meaning,
                entry.frequency,
                entry.status,
                f"{entry.correct_count}/{entry.wrong_count}",
            ),
        )
