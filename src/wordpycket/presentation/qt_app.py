import os
import sys
import time
from collections.abc import Callable
from typing import Protocol

from PySide6.QtCore import QObject, QItemSelectionModel, QProcess, QProcessEnvironment, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from wordpycket.application.services import WordService
from wordpycket.application.study_session import StudyCardState, StudySessionController
from wordpycket.domain.entities import WordEntry


def _font_config() -> dict[str, str]:
    if sys.platform == "darwin":
        return {
            "app_font": "SF Pro Text",
            "ui_stack": '"SF Pro Text", ".AppleSystemUIFont", "PingFang SC"',
            "display_stack": '"SF Pro Display", ".AppleSystemUIFont", "PingFang SC"',
            "word_stack": '"SF Pro Display", ".AppleSystemUIFont", "PingFang SC"',
            "icon_stack": '"SF Pro Display", ".AppleSystemUIFont", "PingFang SC"',
        }
    return {
        "app_font": "Segoe UI",
        "ui_stack": '"Segoe UI", "Microsoft YaHei UI"',
        "display_stack": '"Segoe UI", "Microsoft YaHei UI"',
        "word_stack": '"Segoe UI", "Microsoft YaHei UI"',
        "icon_stack": '"Segoe UI Symbol", "Segoe UI", "Microsoft YaHei UI"',
    }


class ExampleGenerator(Protocol):
    def generate(self, entry: WordEntry, scope: str = ""): ...

    def generate_isolated(self, entry: WordEntry, scope: str = ""): ...

    def correct_entry(self, entry: WordEntry, scope: str = ""): ...

    def correct_entry_isolated(self, entry: WordEntry, scope: str = ""): ...

    def generate_many(self, entries: list[WordEntry], scope: str = "", progress=None, control=None): ...

    def correct_many(self, entries: list[WordEntry], scope: str = "", progress=None, control=None): ...

    def uses_user_model(self) -> bool: ...

    def model_status(self): ...

    def ensure_model_available(self): ...

    def device_status(self): ...

    def check_model_runtime(self): ...

    def recommended_process_parallelism(self) -> int: ...

    def isolated_command(self) -> list[str]: ...

    def isolated_environment(self) -> dict[str, str]: ...

    def isolated_payload(self, action: str, entry: WordEntry, scope: str) -> str: ...

    def parse_isolated_result(self, stdout: str, stderr: str, returncode: int) -> dict: ...


class BatchWorker(QObject):
    progress_changed = Signal(str, int, int, int, float)
    finished = Signal(str, list, list, int)
    failed = Signal(str, str)

    def __init__(
        self,
        action: str,
        entries: list[WordEntry],
        scope: str,
        generator: ExampleGenerator,
        control: Callable[[], str],
    ) -> None:
        super().__init__()
        self._action = action
        self._entries = entries
        self._scope = scope
        self._generator = generator
        self._control = control

    def run(self) -> None:
        updates: list[tuple[str, str, str, str]] = []
        errors: list[str] = []
        total = len(self._entries)
        started_at = time.monotonic()

        def progress(done: int, count: int, workers: int) -> None:
            self.progress_changed.emit(self._action, done, count, workers, time.monotonic() - started_at)

        try:
            if self._action == "补充":
                results, errors, _workers = self._generate(progress)
                for entry, generated in results:
                    updates.append(
                        (
                            entry.id,
                            generated.example_sentence,
                            generated.example_sentence_cn,
                            "",
                        )
                    )
            else:
                results, errors, _workers = self._correct(progress)
                for entry, corrected in results:
                    updates.append(
                        (
                            entry.id,
                            corrected.corrected_word,
                            entry.meaning,
                            entry.forms,
                        )
                    )
        except Exception as error:
            self.failed.emit(self._action, str(error))
            return

        self.finished.emit(self._action, updates, errors, total)

    def _generate(self, progress):
        results = []
        errors = []
        total = len(self._entries)
        for index, entry in enumerate(self._entries, start=1):
            if not self._wait_for_resume():
                break
            progress(index - 1, total, 1)
            try:
                if hasattr(self._generator, "generate_isolated"):
                    generated = self._generator.generate_isolated(entry, self._scope)
                else:
                    generated = self._generator.generate(entry, self._scope)
                results.append((entry, generated))
            except Exception as error:
                errors.append(f"{entry.word}: {error}")
        progress(total, total, 1)
        return results, errors, 1

    def _correct(self, progress):
        results = []
        errors = []
        total = len(self._entries)
        for index, entry in enumerate(self._entries, start=1):
            if not self._wait_for_resume():
                break
            progress(index - 1, total, 1)
            try:
                if hasattr(self._generator, "correct_entry_isolated"):
                    corrected = self._generator.correct_entry_isolated(entry, self._scope)
                else:
                    corrected = self._generator.correct_entry(entry, self._scope)
                results.append((entry, corrected))
            except Exception as error:
                errors.append(f"{entry.word}: {error}")
        progress(total, total, 1)
        return results, errors, 1

    def _wait_for_resume(self) -> bool:
        while self._control() == "paused":
            QThread.msleep(200)
        return self._control() != "stopped"


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
        self._study_session = StudySessionController(service)
        self._selected_id: str | None = None
        self._selected_ids: list[str] = []
        self._mode: str | None = None
        self._batch_state = "idle"
        self._batch_thread: QThread | None = None
        self._batch_worker: BatchWorker | None = None
        self._batch_processes: dict[QProcess, WordEntry] = {}
        self._batch_handled_processes: set[QProcess] = set()
        self._batch_action = ""
        self._batch_scope = ""
        self._batch_entries: list[WordEntry] = []
        self._batch_index = 0
        self._batch_completed_count = 0
        self._batch_finished = False
        self._batch_started_at = 0.0
        self._batch_updated_ids: list[str] = []
        self._batch_errors: list[str] = []
        self._pending_batch_message: tuple[str, str, str] | None = None
        self._user_model_warning_shown = False

        self._app = QApplication.instance() or QApplication(sys.argv)
        self._window = QMainWindow()
        self._window.setWindowTitle("WordPycket")
        self._window.resize(1220, 720)
        self._window.setMinimumSize(980, 620)
        self._apply_style()

        self._search_input: QLineEdit | None = None
        self._scope_input: QLineEdit | None = None
        self._table: QTableWidget | None = None
        self._count_label: QLabel | None = None
        self._progress: QProgressBar | None = None
        self._batch_status_label: QLabel | None = None
        self._supplement_button: QPushButton | None = None
        self._correct_button: QPushButton | None = None
        self._pause_button: QPushButton | None = None
        self._stop_button: QPushButton | None = None
        self._unknown_button: QPushButton | None = None
        self._known_button: QPushButton | None = None
        self._definitely_known_button: QPushButton | None = None
        self._previous_button: QPushButton | None = None
        self._next_button: QPushButton | None = None
        self._word_label: QLabel | None = None
        self._meaning_label: QLabel | None = None
        self._forms_label: QLabel | None = None
        self._example_label: QLabel | None = None
        self._example_cn_label: QLabel | None = None
        self._review_meta_label: QLabel | None = None
        self._study_card: QFrame | None = None
        self._reveal_translation = False
        self._current_card_state: StudyCardState | None = None

        self._show_home()

    def run(self) -> None:
        self._window.show()
        self._app.exec()

    def _apply_style(self) -> None:
        fonts = _font_config()
        self._app.setFont(QFont(fonts["app_font"], 10))
        self._app.setStyleSheet(
            f"""
            QWidget {{
                background: transparent;
                color: #172033;
                font-family: {fonts["ui_stack"]};
                font-size: 14px;
            }}
            QWidget#appSurface {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #f7fbff,
                    stop: 0.45 #edf4ff,
                    stop: 1 #f8fbff
                );
            }}
            QFrame#glassPanel {{
                background: rgba(255, 255, 255, 172);
                border: 1px solid rgba(255, 255, 255, 210);
                border-radius: 30px;
            }}
            QLabel {{
                background: transparent;
            }}
            QLabel#homeTitle {{
                font-family: {fonts["display_stack"]};
                font-size: 32px;
                font-weight: 700;
            }}
            QLabel#title {{
                font-size: 22px;
                font-weight: 700;
            }}
            QLabel#word {{
                font-family: {fonts["word_stack"]};
                font-size: 48px;
                font-weight: 800;
            }}
            QLabel#meaning {{
                font-size: 22px;
            }}
            QLabel#meta {{
                color: #596579;
            }}
            QLabel#formLabel {{
                color: #596579;
                font-weight: 700;
            }}
            QPushButton {{
                background: rgba(255, 255, 255, 154);
                border: 1px solid rgba(255, 255, 255, 196);
                border-radius: 20px;
                padding: 9px 18px;
                font-weight: 700;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background: rgba(255, 255, 255, 210);
            }}
            QPushButton:pressed {{
                padding-top: 10px;
                padding-bottom: 8px;
                background: rgba(232, 242, 255, 220);
            }}
            QPushButton:disabled {{
                color: #9aa6b5;
                background: rgba(237, 241, 247, 120);
            }}
            QPushButton[variant="primary"] {{
                color: white;
                background: rgba(10, 132, 255, 218);
                border-color: rgba(145, 202, 255, 210);
            }}
            QPushButton[variant="primary"]:hover {{
                background: rgba(0, 110, 219, 230);
            }}
            QPushButton[variant="danger"] {{
                color: #ff3b30;
                background: rgba(255, 240, 239, 172);
                border-color: rgba(255, 210, 206, 210);
            }}
            QPushButton[variant="danger"]:hover {{
                color: #d92d24;
                background: rgba(255, 226, 223, 220);
            }}
            QPushButton[variant="icon"] {{
                border-radius: 18px;
                min-width: 36px;
                max-width: 36px;
                min-height: 36px;
                max-height: 36px;
                padding: 0 0 3px 0;
                font-family: {fonts["icon_stack"]};
                font-size: 24px;
                font-weight: 400;
            }}
            QPushButton[variant="icon"]:pressed {{
                padding: 0 0 3px 0;
            }}
            QLineEdit, QPlainTextEdit {{
                background: rgba(255, 255, 255, 178);
                border: 1px solid rgba(255, 255, 255, 210);
                border-radius: 16px;
                padding: 8px 10px;
                selection-background-color: #d8ebff;
            }}
            QMessageBox {{
                background: #f7fbff;
                color: #172033;
            }}
            QMessageBox QLabel {{
                background: transparent;
                color: #172033;
            }}
            QMessageBox QPushButton {{
                background: #eef5ff;
                color: #172033;
                border: 1px solid #d8e4f2;
                border-radius: 12px;
                padding: 7px 18px;
                min-width: 72px;
                min-height: 22px;
            }}
            QMessageBox QPushButton:hover {{
                background: #e2efff;
            }}
            QMessageBox QPushButton:pressed {{
                background: #d8ebff;
            }}
            QTableWidget {{
                background: rgba(255, 255, 255, 172);
                border: 1px solid rgba(255, 255, 255, 210);
                border-radius: 22px;
                gridline-color: rgba(225, 233, 244, 120);
                alternate-background-color: rgba(247, 250, 255, 118);
                selection-background-color: rgba(212, 226, 244, 178);
                selection-color: #172033;
            }}
            QTableWidget::item {{
                border: none;
                padding: 2px 6px;
            }}
            QTableWidget::item:hover {{
                background: rgba(235, 243, 253, 130);
            }}
            QTableWidget::item:selected {{
                background: rgba(212, 226, 244, 178);
                color: #172033;
                border-top: 1px solid rgba(169, 194, 224, 100);
                border-bottom: 1px solid rgba(169, 194, 224, 100);
            }}
            QTableWidget::item:selected:active,
            QTableWidget::item:selected:!active {{
                background: rgba(212, 226, 244, 178);
                color: #172033;
            }}
            QTableWidget::item:focus {{
                outline: none;
            }}
            QHeaderView::section {{
                background: rgba(238, 244, 252, 185);
                color: #596579;
                border: none;
                border-bottom: 1px solid rgba(216, 228, 242, 170);
                padding: 8px;
                font-weight: 700;
            }}
            QScrollBar:vertical {{
                background: rgba(255, 255, 255, 80);
                border: none;
                border-radius: 7px;
                width: 14px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(89, 101, 121, 95);
                border-radius: 6px;
                min-height: 34px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: rgba(89, 101, 121, 130);
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: transparent;
                border: none;
                height: 0;
                width: 0;
            }}
            QScrollBar:horizontal {{
                background: rgba(255, 255, 255, 80);
                border: none;
                border-radius: 7px;
                height: 14px;
                margin: 2px;
            }}
            QScrollBar::handle:horizontal {{
                background: rgba(89, 101, 121, 95);
                border-radius: 6px;
                min-width: 34px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: rgba(89, 101, 121, 130);
            }}
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {{
                background: transparent;
                border: none;
                height: 0;
                width: 0;
            }}
            QProgressBar {{
                background: rgba(232, 238, 247, 170);
                border: none;
                border-radius: 5px;
                height: 10px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: rgba(10, 132, 255, 220);
                border-radius: 5px;
            }}
            """
        )

    def _panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("glassPanel")
        effect = QGraphicsDropShadowEffect(panel)
        effect.setBlurRadius(42)
        effect.setOffset(0, 14)
        effect.setColor(QColor(78, 106, 145, 62))
        panel.setGraphicsEffect(effect)
        return panel

    def _button(self, text: str, callback: Callable[[], None], variant: str = "") -> QPushButton:
        button = QPushButton(text)
        if variant:
            button.setProperty("variant", variant)
        button.clicked.connect(callback)
        return button

    def _meta_label(self, text: str = "") -> QLabel:
        label = QLabel(text)
        label.setObjectName("meta")
        return label

    def _form_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("formLabel")
        return label

    def _set_page(self, widget: QWidget) -> None:
        self._window.setCentralWidget(widget)

    def _show_home(self) -> None:
        self._study_session.leave_active_session()
        self._study_session.reset()
        self._mode = None
        self._selected_id = None
        self._selected_ids = []

        root = QWidget()
        root.setObjectName("appSurface")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(32, 30, 32, 30)
        layout.setSpacing(18)

        title = QLabel("WordPycket")
        title.setObjectName("homeTitle")
        subtitle = self._meta_label("选择要进入的功能")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        cards = QHBoxLayout()
        cards.setSpacing(24)
        counts = self._study_session.pool_counts()
        cards.addWidget(self._home_card("学习", f"{counts['learning']} 个单词", "进入学习", lambda: self._show_mode("learning")))
        cards.addWidget(self._home_card("复习", f"{counts['review']} 个单词", "进入复习", lambda: self._show_mode("review")))
        cards.addWidget(self._home_card("词表", f"{counts['total']} 个单词", "查看词表", self._show_word_list))
        layout.addLayout(cards, 1)
        layout.addWidget(self._model_status_panel())

        footer = QHBoxLayout()
        footer.addStretch()
        footer.addWidget(self._button("重置学习进度", self._confirm_reset_progress, "danger"))
        layout.addLayout(footer)
        self._set_page(root)

    def _home_card(self, title: str, count: str, action: str, callback: Callable[[], None]) -> QFrame:
        card = self._panel()
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        title_label = QLabel(title)
        title_label.setObjectName("title")
        layout.addWidget(title_label)
        layout.addWidget(self._meta_label(count))
        layout.addStretch()
        layout.addWidget(self._button(action, callback, "primary"))
        return card

    def _model_status_panel(self) -> QFrame:
        panel = self._panel()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        text = QLabel(self._model_status_text())
        text.setObjectName("meta")
        text.setWordWrap(True)
        layout.addWidget(text, 1)
        layout.addWidget(self._button("检查模型", self._check_model, "primary"))
        return panel

    def _model_status_text(self) -> str:
        if self._example_generator is None:
            return "智能模型：未配置。学习和复习可正常使用，智能补充/修正不可用。"
        if not hasattr(self._example_generator, "model_status"):
            return "智能模型：当前生成器不支持模型状态检查。"
        try:
            status = self._example_generator.model_status()
        except Exception as error:
            return f"智能模型：配置需要处理。{error}"
        path = getattr(status, "path", None)
        if path is None:
            return (
                "智能模型：未找到。可点击检查模型下载默认 Hugging Face 模型；不影响普通学习和复习。"
                f"\n{self._device_status_text()}"
            )
        if getattr(status, "is_user_model", False):
            model_text = f"智能模型：使用自带模型 {path.name}。不保证完全兼容。"
        else:
            model_text = f"智能模型：使用默认模型 {path.name}。"
        return f"{model_text}\n{self._device_status_text()}"

    def _device_status_text(self) -> str:
        if self._example_generator is None:
            return "Device：未配置。"
        if not hasattr(self._example_generator, "device_status"):
            return "Device：当前生成器不支持设备检查。"
        try:
            status = self._example_generator.device_status()
        except Exception as error:
            return f"Device：检查失败。{error}"
        detected = self._device_label(getattr(status, "detected", "cpu"))
        selected = getattr(status, "selected", None)
        error = getattr(status, "error", "")
        if error:
            return f"Device：检测到 {detected}，当前不可用。{error}"
        return f"Device：检测到 {detected}，将使用 {self._device_label(selected)}。"

    @staticmethod
    def _device_label(device: str | None) -> str:
        labels = {
            "cuda": "CUDA",
            "mps": "Metal",
            "cpu": "CPU",
            "auto": "Auto",
            None: "未知",
        }
        return labels.get(device, str(device).upper())

    def _check_model(self) -> None:
        if self._example_generator is None:
            QMessageBox.information(self._window, "模型检查", "未配置本地模型生成器。")
            return
        if not hasattr(self._example_generator, "check_model_runtime"):
            QMessageBox.information(self._window, "模型检查", "当前生成器不支持模型检查。")
            return
        try:
            result = self._example_generator.check_model_runtime()
        except Exception as error:
            QMessageBox.critical(self._window, "模型检查失败", str(error))
            self._show_home()
            return
        status = result.model
        device = result.device
        path = getattr(status, "path", None)
        device_line = f"\nDevice：{self._device_label(getattr(device, 'selected', None))}"
        smoke_line = "\n最小执行测试：通过。"
        if path is None:
            QMessageBox.information(self._window, "模型检查", "未找到可用模型。")
        elif getattr(status, "is_user_model", False):
            QMessageBox.warning(
                self._window,
                "模型检查",
                f"当前使用自带模型：{path.name}"
                f"{device_line}{smoke_line}\n不保证提示词格式、JSON 输出稳定性和 llama.cpp 兼容性。",
            )
        elif getattr(status, "downloaded", False):
            QMessageBox.information(
                self._window,
                "模型检查",
                f"默认模型已下载：{path.name}{device_line}{smoke_line}",
            )
        else:
            QMessageBox.information(
                self._window,
                "模型检查",
                f"默认模型已就绪：{path.name}{device_line}{smoke_line}",
            )
        self._show_home()

    def _confirm_reset_progress(self) -> None:
        confirmation_text = "我确认重置学习进度"
        text, accepted = QInputDialog.getText(
            self._window,
            "确认重置",
            f"此操作会删除所有学习记录和例句。\n请输入“{confirmation_text}”以继续：",
        )
        if not accepted:
            return
        if text.strip() != confirmation_text:
            QMessageBox.information(self._window, "未重置", "确认文本不匹配，学习进度未重置。")
            return
        imported_count = self._service.replace_words(self._reset_entries_loader())
        self._study_session.clear_last_session()
        QMessageBox.information(
            self._window,
            "已重置",
            f"已从 CSV 重新导入 {imported_count} 个词条，原有学习记录已清空。",
        )
        self._show_home()

    def _show_mode(self, mode: str) -> None:
        self._mode = mode
        self._selected_id = None
        self._selected_ids = []

        root = QWidget()
        root.setObjectName("appSurface")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(32, 28, 32, 32)
        layout.setSpacing(18)

        title_bar = QHBoxLayout()
        title = QLabel("学习" if mode == "learning" else "复习")
        title.setObjectName("homeTitle")
        title_bar.addWidget(title)
        title_bar.addStretch()
        title_bar.addWidget(self._button("返回主页", self._show_home))
        layout.addLayout(title_bar)

        layout.addStretch()
        card = self._panel()
        self._study_card = card
        card.mousePressEvent = lambda event: self._toggle_translation_reveal()
        card.setFixedSize(860, 560)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 22, 30, 28)
        card_layout.setSpacing(8)

        self._word_label = QLabel("")
        self._word_label.setObjectName("word")
        self._word_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._word_label.setFixedHeight(74)
        self._meaning_label = QLabel("")
        self._meaning_label.setObjectName("meaning")
        self._meaning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._meaning_label.setFixedHeight(34)
        self._forms_label = self._meta_label()
        self._forms_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._forms_label.setFixedHeight(24)
        self._example_label = self._meta_label()
        self._example_label.setWordWrap(False)
        self._example_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._example_label.setFixedHeight(40)
        self._example_cn_label = self._meta_label()
        self._example_cn_label.setWordWrap(False)
        self._example_cn_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._example_cn_label.setFixedHeight(34)
        self._review_meta_label = self._meta_label()
        self._review_meta_label.setWordWrap(False)
        self._review_meta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._review_meta_label.setFixedHeight(30)

        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        self._previous_button = self._button("‹", self._show_previous_word, "icon")
        self._previous_button.setToolTip("上一个")
        self._previous_button.setFlat(False)
        self._next_button = self._button("›", self._continue_from_history, "icon")
        self._next_button.setToolTip("下一个")
        self._next_button.setFlat(False)
        nav_row.addWidget(self._previous_button, 0, Qt.AlignmentFlag.AlignLeft)
        nav_row.addStretch()
        nav_row.addWidget(self._next_button, 0, Qt.AlignmentFlag.AlignRight)
        card_layout.addLayout(nav_row)
        card_layout.addWidget(self._word_label)
        card_layout.addWidget(self._meaning_label)
        card_layout.addWidget(self._forms_label)
        card_layout.addWidget(self._example_label)
        card_layout.addWidget(self._example_cn_label)
        card_layout.addStretch()

        actions = QGridLayout()
        actions.setHorizontalSpacing(8)
        actions.setVerticalSpacing(8)
        self._unknown_button = self._button("不会", self._mark_unknown, "danger")
        self._known_button = self._button("会", self._mark_known, "primary")
        self._definitely_known_button = self._button("绝对会", self._mark_definitely_known, "primary")
        actions.addWidget(self._unknown_button, 0, 0)
        actions.addWidget(self._known_button, 0, 1)
        actions.addWidget(self._definitely_known_button, 1, 0, 1, 2)
        card_layout.addLayout(actions)
        card_layout.addWidget(self._review_meta_label)

        center = QHBoxLayout()
        center.addStretch()
        center.addWidget(card)
        center.addStretch()
        layout.addLayout(center)
        layout.addStretch()
        self._set_page(root)
        self._render_study_card(self._study_session.begin(mode))  # type: ignore[arg-type]

    def _render_study_card(self, state: StudyCardState) -> None:
        self._current_card_state = state
        self._reveal_translation = False
        self._set_single_line_text(self._word_label, state.word_text, 780)
        self._set_single_line_text(self._forms_label, state.forms_text, 760)
        self._set_single_line_text(self._example_label, state.example_text, 780)
        self._update_translation_labels()
        self._set_single_line_text(self._review_meta_label, state.meta_text, 780)
        has_entry = state.has_entry
        history_view = state.history_view
        show_actions = has_entry and not history_view
        self._unknown_button.setVisible(show_actions)
        self._known_button.setVisible(show_actions)
        self._definitely_known_button.setVisible(show_actions and self._mode == "learning")
        self._previous_button.setVisible(state.can_show_previous)
        self._next_button.setVisible(history_view)

    def _set_translation_revealed(self, revealed: bool) -> None:
        if self._reveal_translation == revealed:
            return
        self._reveal_translation = revealed
        self._update_translation_labels()

    def _toggle_translation_reveal(self) -> None:
        self._set_translation_revealed(not self._reveal_translation)

    def _update_translation_labels(self) -> None:
        if self._meaning_label is None or self._example_cn_label is None:
            return
        state = self._current_card_state
        if state is None or not state.has_entry:
            self._set_single_line_text(self._meaning_label, state.meaning_text if state else "", 760)
            self._set_single_line_text(self._example_cn_label, state.example_cn_text if state else "", 780)
            return
        if self._reveal_translation:
            self._set_single_line_text(self._meaning_label, state.meaning_text, 760)
            self._set_single_line_text(self._example_cn_label, state.example_cn_text, 780)
            return
        self._set_single_line_text(self._meaning_label, "", 760)
        self._set_single_line_text(self._example_cn_label, "", 780)

    def _set_single_line_text(self, label: QLabel, text: str, max_width: int) -> None:
        label.setToolTip(text)
        metrics = label.fontMetrics()
        label.setText(metrics.elidedText(text, Qt.TextElideMode.ElideRight, max_width))

    def _mark_known(self) -> None:
        self._mark_current("known")

    def _mark_unknown(self) -> None:
        self._mark_current("unknown")

    def _mark_definitely_known(self) -> None:
        self._mark_current("definitely_known")

    def _mark_current(self, result: str) -> None:
        state = self._study_session.mark_current(result)  # type: ignore[arg-type]
        if state is None:
            QMessageBox.information(self._window, "暂无词条", "当前没有可复习的词条。")
            return
        self._render_study_card(state)

    def _show_previous_word(self) -> None:
        state = self._study_session.show_previous_word()
        if state is not None:
            self._render_study_card(state)

    def _continue_from_history(self) -> None:
        self._render_study_card(self._study_session.continue_from_history())

    def _show_word_list(self) -> None:
        self._mode = "words"
        self._study_session.reset()
        self._selected_id = None
        self._selected_ids = []

        root = QWidget()
        root.setObjectName("appSurface")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(32, 28, 32, 32)
        layout.setSpacing(16)

        title_bar = QHBoxLayout()
        title = QLabel("词表")
        title.setObjectName("homeTitle")
        title_bar.addWidget(title)
        title_bar.addStretch()
        title_bar.addWidget(self._button("返回主页", self._show_home))
        layout.addLayout(title_bar)

        panel = self._panel()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(10)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("搜索"))
        self._search_input = QLineEdit()
        self._search_input.textChanged.connect(lambda _text: self._refresh_words(False))
        search_row.addWidget(self._search_input, 1)
        panel_layout.addLayout(search_row)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("AI范围"))
        self._scope_input = QLineEdit("人工智能相关的翻译")
        scope_row.addWidget(self._scope_input, 1)
        panel_layout.addLayout(scope_row)

        self._table = QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels(["#", "单词", "释义", "词形", "例句", "例句中文", "频率", "状态", "复习"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for column, width in enumerate([56, 150, 230, 190, 220, 180, 70, 90, 90]):
            self._table.setColumnWidth(column, width)
        self._table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self._table.itemDoubleClicked.connect(lambda item: self._edit_selected_word(item.row()))
        panel_layout.addWidget(self._table, 1)

        footer = QVBoxLayout()
        action_row = QHBoxLayout()
        self._count_label = self._meta_label()
        action_row.addWidget(self._count_label, 1)
        self._supplement_button = self._button("智能补充选中", self._supplement_selected_example)
        self._correct_button = self._button("智能修正选中", self._correct_selected_entry)
        self._pause_button = self._button("暂停", self._toggle_batch_pause)
        self._stop_button = self._button("停止", self._stop_batch)
        delete_button = self._button("删除选中", self._delete_selected, "danger")
        action_row.addWidget(self._supplement_button)
        action_row.addWidget(self._correct_button)
        action_row.addWidget(self._pause_button)
        action_row.addWidget(self._stop_button)
        action_row.addWidget(delete_button)
        footer.addLayout(action_row)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._batch_status_label = self._meta_label()
        footer.addWidget(self._progress)
        footer.addWidget(self._batch_status_label)
        panel_layout.addLayout(footer)
        layout.addWidget(panel, 1)
        self._set_page(root)
        self._set_batch_idle()
        self._refresh_words(False)

    def _visible_word_entries(self, query: str = "") -> list[WordEntry]:
        if self._mode in {"learning", "review"}:
            return self._study_session.mode_entries(query)
        return self._service.list_words(query)

    def _refresh_words(self, reload_current: bool = True) -> None:
        if self._table is None:
            return
        selected_ids = self._selected_entry_ids()
        self._table.blockSignals(True)
        try:
            self._table.setRowCount(0)
            query = self._search_input.text() if self._search_input is not None else ""
            entries = self._visible_word_entries(query)
            self._table.setRowCount(len(entries))
            for row, entry in enumerate(entries):
                self._insert_entry(row, entry)
        finally:
            self._table.blockSignals(False)
        if selected_ids:
            self._restore_table_selection(selected_ids)
        else:
            self._selected_ids = []
            self._selected_id = None
        self._update_count_label()
        if reload_current and self._mode != "words":
            current = self._study_session.current_entry
            self._render_study_card(self._study_session.reload(current.id if current else None))

    def _insert_entry(self, row: int, entry: WordEntry) -> None:
        values = [
            entry.source_index,
            entry.word,
            entry.meaning,
            entry.forms,
            entry.example_sentence,
            entry.example_sentence_cn,
            entry.frequency,
            entry.status,
            f"{entry.correct_count}/{entry.wrong_count}",
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setData(Qt.ItemDataRole.UserRole, entry.id)
            item.setToolTip(str(value))
            if column in {0, 6}:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            else:
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            if column == 7:
                if entry.status == "学习池":
                    item.setForeground(QColor("#0a84ff"))
                elif entry.status == "复习池":
                    item.setForeground(QColor("#b15f00"))
                else:
                    item.setForeground(QColor("#596579"))
            self._table.setItem(row, column, item)

    def _on_table_selection_changed(self) -> None:
        self._selected_ids = self._selected_entry_ids()
        self._selected_id = self._selected_ids[0] if self._selected_ids else None
        self._update_count_label()

    def _selected_entry_ids(self) -> list[str]:
        if self._table is None:
            return list(self._selected_ids)
        ids = []
        for index in self._table.selectionModel().selectedRows():
            item = self._table.item(index.row(), 0)
            if item is not None:
                ids.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return ids

    def _update_count_label(self) -> None:
        if self._count_label is None or self._table is None:
            return
        total = len(self._visible_word_entries())
        selected = len(self._selected_entry_ids())
        selected_text = f" | 已选 {selected} 条" if selected else ""
        self._count_label.setText(f"显示 {self._table.rowCount()} / 共 {total} 条{selected_text}")

    def _entries_by_id(self, entry_ids: list[str]) -> list[WordEntry]:
        entries = []
        for entry_id in entry_ids:
            entry = self._service.get_word(entry_id)
            if entry is not None:
                entries.append(entry)
        return entries

    def _restore_table_selection(self, entry_ids: list[str]) -> None:
        if self._table is None:
            return
        self._table.clearSelection()
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return
        first_row = None
        targets = set(entry_ids)
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and str(item.data(Qt.ItemDataRole.UserRole)) in targets:
                self._table.setCurrentCell(row, 0)
                selection_model.select(
                    self._table.model().index(row, 0),
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
                if first_row is None:
                    first_row = row
        if first_row is not None:
            self._table.scrollToItem(self._table.item(first_row, 0))
        self._selected_ids = self._selected_entry_ids()
        self._selected_id = self._selected_ids[0] if self._selected_ids else None

    def _delete_selected(self) -> None:
        selected_ids = self._selected_entry_ids()
        if not selected_ids:
            QMessageBox.information(self._window, "未选择词条", "请先在列表中选择一个词条。")
            return
        for entry_id in selected_ids:
            self._service.delete_word(entry_id)
        self._selected_id = None
        self._selected_ids = []
        self._refresh_words(self._mode != "words")

    def _edit_selected_word(self, row: int) -> None:
        item = self._table.item(row, 0)
        if item is None:
            return
        entry = self._service.get_word(str(item.data(Qt.ItemDataRole.UserRole)))
        if entry is None:
            QMessageBox.information(self._window, "编辑失败", "当前词条不存在。")
            return
        self._show_word_editor(entry)

    def _show_word_editor(self, entry: WordEntry) -> None:
        dialog = QDialog(self._window)
        dialog.setWindowTitle(f"编辑词条 - {entry.word}")
        dialog.setObjectName("appSurface")
        dialog.resize(700, 560)
        dialog.setMinimumSize(620, 500)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("编辑词条")
        title.setObjectName("homeTitle")
        subtitle = self._meta_label(entry.word)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        panel = self._panel()
        form = QGridLayout(panel)
        form.setContentsMargins(24, 24, 24, 24)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        form.setColumnStretch(1, 1)

        word_input = QLineEdit(entry.word)
        meaning_input = QLineEdit(entry.meaning)
        forms_input = QLineEdit(entry.forms)
        example_input = QPlainTextEdit(entry.example_sentence)
        example_cn_input = QPlainTextEdit(entry.example_sentence_cn)
        example_input.setMinimumHeight(96)
        example_cn_input.setMinimumHeight(96)
        form.addWidget(self._form_label("单词"), 0, 0)
        form.addWidget(word_input, 0, 1)
        form.addWidget(self._form_label("释义"), 1, 0)
        form.addWidget(meaning_input, 1, 1)
        form.addWidget(self._form_label("词形"), 2, 0)
        form.addWidget(forms_input, 2, 1)
        form.addWidget(self._form_label("例句"), 3, 0, Qt.AlignmentFlag.AlignTop)
        form.addWidget(example_input, 3, 1)
        form.addWidget(self._form_label("例句中文"), 4, 0, Qt.AlignmentFlag.AlignTop)
        form.addWidget(example_cn_input, 4, 1)

        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self._button("取消", dialog.reject))

        def save() -> None:
            word = word_input.text().strip()
            meaning = meaning_input.text().strip()
            forms = forms_input.text().strip()
            if not word:
                QMessageBox.information(dialog, "无法保存", "单词不能为空。")
                return
            if not meaning:
                QMessageBox.information(dialog, "无法保存", "释义不能为空。")
                return
            try:
                updated = self._service.update_text(entry.id, word, meaning, forms)
                if updated is None:
                    QMessageBox.information(dialog, "保存失败", "当前词条不存在。")
                    return
                updated = self._service.update_examples(
                    entry.id,
                    example_input.toPlainText().strip(),
                    example_cn_input.toPlainText().strip(),
                )
            except Exception as error:
                QMessageBox.critical(dialog, "保存失败", str(error))
                return
            self._selected_id = updated.id if updated else entry.id
            self._selected_ids = [self._selected_id]
            self._refresh_words(False)
            self._restore_table_selection(self._selected_ids)
            dialog.accept()

        actions.addWidget(self._button("保存", save, "primary"))
        form.addLayout(actions, 5, 0, 1, 2)
        layout.addWidget(panel)
        dialog.exec()

    def _supplement_selected_example(self) -> None:
        self._start_batch("补充")

    def _correct_selected_entry(self) -> None:
        self._start_batch("修正")

    def _start_batch(self, action: str) -> None:
        if self._batch_state != "idle":
            return
        selected_ids = self._selected_entry_ids()
        if not selected_ids:
            QMessageBox.information(self._window, "未选择词条", "请先在列表中选择一个词条。")
            return
        if self._example_generator is None:
            QMessageBox.information(self._window, f"无法智能{action}", "未配置本地模型生成器。")
            return
        entries = self._entries_by_id(selected_ids)
        if not entries:
            QMessageBox.information(self._window, f"智能{action}失败", "当前词条不存在。")
            return
        if not self._can_run_isolated_batch():
            QMessageBox.information(
                self._window,
                f"无法智能{action}",
                "当前生成器不支持隔离子进程，请使用 LocalLlmExampleGenerator。",
            )
            return
        self._warn_if_using_user_model()

        scope = self._scope_input.text().strip() if self._scope_input is not None else ""
        self._set_batch_running()
        if action == "补充":
            self._supplement_button.setText(f"补充中 0/{len(entries)}")
            self._supplement_button.setEnabled(False)
        else:
            self._correct_button.setText(f"修正中 0/{len(entries)}")
            self._correct_button.setEnabled(False)
        self._set_batch_progress(action, 0, len(entries), 0, 0.0)

        self._batch_action = action
        self._batch_scope = scope
        self._batch_entries = entries
        self._batch_index = 0
        self._batch_completed_count = 0
        self._batch_finished = False
        self._batch_started_at = time.monotonic()
        self._batch_updated_ids = []
        self._batch_errors = []
        self._batch_processes = {}
        self._batch_handled_processes = set()
        self._pump_batch_processes()

    def _can_run_isolated_batch(self) -> bool:
        return (
            self._example_generator is not None
            and hasattr(self._example_generator, "isolated_command")
            and hasattr(self._example_generator, "isolated_environment")
            and hasattr(self._example_generator, "isolated_payload")
            and hasattr(self._example_generator, "parse_isolated_result")
        )

    def _warn_if_using_user_model(self) -> None:
        if self._user_model_warning_shown or self._example_generator is None:
            return
        if not hasattr(self._example_generator, "uses_user_model"):
            return
        try:
            uses_user_model = self._example_generator.uses_user_model()
        except Exception:
            return
        if not uses_user_model:
            return
        self._user_model_warning_shown = True
        QMessageBox.warning(
            self._window,
            "使用自带模型",
            "检测到 model 目录中已有用户提供的 GGUF 模型，将优先使用该模型。\n"
            "不同模型的提示词格式、JSON 输出稳定性和 llama.cpp 兼容性可能不同，"
            "软件不保证自带模型完全兼容。",
        )

    def _pump_batch_processes(self) -> None:
        if self._batch_finished:
            return
        if self._batch_state == "stopped":
            if not self._batch_processes:
                self._finish_isolated_batch()
            return
        if self._batch_state == "paused":
            return
        if self._batch_index >= len(self._batch_entries) and not self._batch_processes:
            self._finish_isolated_batch()
            return
        if self._example_generator is None:
            self._batch_errors.append("生成器已不可用。")
            self._finish_isolated_batch()
            return

        while (
            self._batch_state == "running"
            and self._batch_index < len(self._batch_entries)
            and len(self._batch_processes) < self._batch_parallel_limit()
        ):
            self._start_one_batch_process(self._batch_entries[self._batch_index])
            self._batch_index += 1

    def _start_one_batch_process(self, entry: WordEntry) -> None:
        self._on_batch_progress(
            self._batch_action,
            self._batch_completed_count,
            len(self._batch_entries),
            len(self._batch_processes),
            time.monotonic() - self._batch_started_at,
        )

        try:
            llm_action = "generate" if self._batch_action == "补充" else "correct"
            command = self._example_generator.isolated_command()
            payload = self._example_generator.isolated_payload(llm_action, entry, self._batch_scope)
            env_values = self._example_generator.isolated_environment()
        except Exception as error:
            self._batch_errors.append(f"{entry.word}: {error}")
            self._batch_completed_count += 1
            QTimer.singleShot(0, self._pump_batch_processes)
            return

        process = QProcess(self._window)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        environment = QProcessEnvironment.systemEnvironment()
        for key, value in env_values.items():
            environment.insert(key, value)
        process.setProcessEnvironment(environment)
        process.started.connect(
            lambda process=process, payload=payload: self._write_batch_process_payload(process, payload)
        )
        process.finished.connect(
            lambda exit_code, exit_status, process=process: self._on_batch_process_finished(
                process,
                exit_code,
                exit_status,
            )
        )
        process.errorOccurred.connect(
            lambda error, process=process: self._on_batch_process_error(process, error)
        )
        self._batch_processes[process] = entry
        process.start()
        self._on_batch_progress(
            self._batch_action,
            self._batch_completed_count,
            len(self._batch_entries),
            len(self._batch_processes),
            time.monotonic() - self._batch_started_at,
        )

    def _write_batch_process_payload(self, process: QProcess, payload: str) -> None:
        if process not in self._batch_processes:
            return
        process.write(payload.encode("utf-8"))
        process.closeWriteChannel()

    def _on_batch_process_error(self, process: QProcess, error: QProcess.ProcessError) -> None:
        if process in self._batch_handled_processes:
            return
        if error != QProcess.ProcessError.FailedToStart:
            return
        entry = self._batch_processes.get(process)
        if entry is None:
            return
        detail = bytes(process.readAllStandardError()).decode("utf-8", errors="replace").strip()
        self._handle_batch_process_failure(process, entry, detail or "模型子进程启动失败。")

    def _on_batch_process_finished(
        self,
        process: QProcess,
        exit_code: int,
        _exit_status: QProcess.ExitStatus,
    ) -> None:
        if process in self._batch_handled_processes:
            return
        if self._example_generator is None:
            return
        self._batch_handled_processes.add(process)
        entry = self._batch_processes.pop(process, None)
        if entry is None:
            return
        stdout = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        stderr = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        process.deleteLater()

        try:
            data = self._example_generator.parse_isolated_result(stdout, stderr, exit_code)
            self._apply_batch_result(entry, data)
            self._refresh_words(False)
            self._restore_table_selection(self._batch_updated_ids)
        except Exception as error:
            self._batch_errors.append(f"{entry.word}: {error}")

        self._batch_completed_count += 1
        self._on_batch_progress(
            self._batch_action,
            self._batch_completed_count,
            len(self._batch_entries),
            len(self._batch_processes),
            time.monotonic() - self._batch_started_at,
        )
        QTimer.singleShot(0, self._pump_batch_processes)

    def _handle_batch_process_failure(self, process: QProcess, entry: WordEntry, message: str) -> None:
        self._batch_handled_processes.add(process)
        self._batch_processes.pop(process, None)
        process.deleteLater()
        self._batch_errors.append(f"{entry.word}: {message}")
        self._batch_completed_count += 1
        self._on_batch_progress(
            self._batch_action,
            self._batch_completed_count,
            len(self._batch_entries),
            len(self._batch_processes),
            time.monotonic() - self._batch_started_at,
        )
        QTimer.singleShot(0, self._pump_batch_processes)

    def _apply_batch_result(self, entry: WordEntry, data: dict) -> None:
        if self._batch_action == "补充":
            updated = self._service.update_examples(
                entry.id,
                str(data["example_sentence"]),
                str(data["example_sentence_cn"]),
            )
        else:
            updated = self._service.update_text(
                entry.id,
                str(data["corrected_word"]),
                entry.meaning,
                entry.forms,
            )
        if updated is not None:
            self._batch_updated_ids.append(updated.id)

    def _finish_isolated_batch(self) -> None:
        if self._batch_finished:
            return
        self._batch_finished = True
        action = self._batch_action
        total = len(self._batch_entries)
        updated_ids = list(self._batch_updated_ids)
        errors = list(self._batch_errors)
        self._batch_processes = {}
        self._batch_handled_processes = set()
        self._set_batch_idle()
        self._finish_batch_progress(action, len(updated_ids), total, len(errors))
        self._selected_ids = updated_ids
        self._selected_id = updated_ids[0] if updated_ids else None
        self._refresh_words(False)
        self._restore_table_selection(updated_ids)
        message = f"已{action} {len(updated_ids)} / {total} 条。"
        if errors:
            message = f"{message}\n失败 {len(errors)} 条：\n" + "\n".join(errors[:5])
        QTimer.singleShot(0, lambda: self._show_batch_message("information", f"智能{action}完成", message))

    def _batch_parallel_limit(self) -> int:
        if (
            self._example_generator is not None
            and hasattr(self._example_generator, "recommended_process_parallelism")
        ):
            try:
                return max(1, min(8, self._example_generator.recommended_process_parallelism()))
            except Exception:
                return 2

        raw_value = os.getenv("WORDPYCKET_LLM_PROCESS_PARALLEL")
        try:
            return max(1, min(8, int(raw_value))) if raw_value is not None else 2
        except ValueError:
            return 2

    def _on_batch_progress(self, action: str, done: int, total: int, workers: int, elapsed: float) -> None:
        if action == "补充":
            self._supplement_button.setText(f"补充中 {done}/{total} 并行 {workers}")
        else:
            self._correct_button.setText(f"修正中 {done}/{total} 并行 {workers}")
        self._set_batch_progress(action, done, total, workers, elapsed)

    def _on_batch_finished(
        self,
        action: str,
        updates: list[tuple[str, str, str, str]],
        errors: list[str],
        total: int,
    ) -> None:
        updated_ids: list[str] = []
        for entry_id, first_value, second_value, third_value in updates:
            try:
                if action == "补充":
                    updated = self._service.update_examples(entry_id, first_value, second_value)
                else:
                    updated = self._service.update_text(entry_id, first_value, second_value, third_value)
                if updated is not None:
                    updated_ids.append(updated.id)
            except Exception as error:
                entry = self._service.get_word(entry_id)
                word = entry.word if entry is not None else entry_id
                errors.append(f"{word}: {error}")

        self._set_batch_idle()
        self._finish_batch_progress(action, len(updated_ids), total, len(errors))
        self._selected_ids = updated_ids
        self._selected_id = updated_ids[0] if updated_ids else None
        self._refresh_words(False)
        self._restore_table_selection(updated_ids)
        message = f"已{action} {len(updated_ids)} / {total} 条。"
        if errors:
            message = f"{message}\n失败 {len(errors)} 条：\n" + "\n".join(errors[:5])
        self._pending_batch_message = ("information", f"智能{action}完成", message)

    def _on_batch_failed(self, action: str, message: str) -> None:
        self._set_batch_idle()
        self._clear_batch_progress(f"{action}失败")
        self._pending_batch_message = ("critical", f"智能{action}失败", message)

    def _on_batch_thread_finished(self) -> None:
        self._batch_thread = None
        self._batch_worker = None
        pending_message = self._pending_batch_message
        self._pending_batch_message = None
        if pending_message is not None:
            level, title, message = pending_message
            QTimer.singleShot(0, lambda: self._show_batch_message(level, title, message))

    def _show_batch_message(self, level: str, title: str, message: str) -> None:
        if level == "critical":
            QMessageBox.critical(self._window, title, message)
            return
        QMessageBox.information(self._window, title, message)

    def _set_batch_running(self) -> None:
        self._batch_state = "running"
        if self._pause_button is not None:
            self._pause_button.setText("暂停")
            self._pause_button.setEnabled(True)
        if self._stop_button is not None:
            self._stop_button.setEnabled(True)

    def _set_batch_idle(self) -> None:
        self._batch_state = "idle"
        if self._supplement_button is not None:
            self._supplement_button.setText("智能补充选中")
            self._supplement_button.setEnabled(True)
        if self._correct_button is not None:
            self._correct_button.setText("智能修正选中")
            self._correct_button.setEnabled(True)
        if self._pause_button is not None:
            self._pause_button.setText("暂停")
            self._pause_button.setEnabled(False)
        if self._stop_button is not None:
            self._stop_button.setEnabled(False)

    def _toggle_batch_pause(self) -> None:
        if self._batch_state == "running":
            self._batch_state = "paused"
            self._pause_button.setText("继续")
            if self._batch_status_label is not None:
                self._batch_status_label.setText(f"{self._batch_status_label.text()} | 已暂停")
            return
        if self._batch_state == "paused":
            self._batch_state = "running"
            self._pause_button.setText("暂停")
            if self._batch_entries:
                QTimer.singleShot(0, self._pump_batch_processes)

    def _stop_batch(self) -> None:
        if self._batch_state in {"running", "paused"}:
            self._batch_state = "stopped"
            self._pause_button.setText("暂停")
            self._pause_button.setEnabled(False)
            self._stop_button.setEnabled(False)
            if self._batch_status_label is not None:
                self._batch_status_label.setText(f"{self._batch_status_label.text()} | 正在停止")
            if self._batch_processes:
                for process in list(self._batch_processes):
                    process.kill()
            else:
                QTimer.singleShot(0, self._finish_isolated_batch)

    def _batch_control_state(self) -> str:
        return self._batch_state

    def _set_batch_progress(
        self,
        action: str,
        done: int,
        total: int,
        workers: int,
        elapsed_seconds: float,
    ) -> None:
        percent = int(done / total * 100) if total else 0
        if self._progress is not None:
            self._progress.setValue(percent)
        if self._batch_status_label is None:
            return
        if done <= 0:
            eta_text = "预估剩余：估算中"
        else:
            average_seconds = elapsed_seconds / done
            remaining_seconds = max(0.0, average_seconds * (total - done))
            eta_text = f"预估剩余：{self._format_duration(remaining_seconds)}"
        worker_text = f"并行 {workers}" if workers else "准备中"
        self._batch_status_label.setText(
            f"{action}进度：{done} / {total} ({percent}%) | {worker_text} | {eta_text}"
        )

    def _finish_batch_progress(self, action: str, success_count: int, total: int, error_count: int) -> None:
        if self._progress is not None:
            self._progress.setValue(100 if total else 0)
        if self._batch_status_label is not None:
            self._batch_status_label.setText(f"{action}完成：成功 {success_count} / {total}，失败 {error_count}。")

    def _clear_batch_progress(self, message: str = "") -> None:
        if self._progress is not None:
            self._progress.setValue(0)
        if self._batch_status_label is not None:
            self._batch_status_label.setText(message)

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
