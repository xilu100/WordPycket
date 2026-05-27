import os
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QItemSelectionModel, Qt, QThread, QTimer, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFrame,
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

from wordpycket.application.ai_batch import initial_batch_parallel_limit
from wordpycket.application.ports import CsvDatasetResult, CsvImportResult, ExampleGenerator, MeaningTranslator, PdfImportResult
from wordpycket.application.services import WordService
from wordpycket.application.study_session import StudyCardState, StudySessionController
from wordpycket.domain.entities import WordEntry
from wordpycket.presentation.llm_jobs import ExplainCompleted, ExplainFailed, ExplainProgress, LlmJobPoller
from wordpycket.presentation.qt_workers import BackgroundTaskWorker, BatchWorker, PdfImportWorker, UiThreadBridge
from wordpycket.presentation.style import apply_app_style, create_panel


class WordPycketApp:
    def __init__(
        self,
        service: WordService,
        example_generator: ExampleGenerator | None = None,
        meaning_translator: MeaningTranslator | None = None,
        csv_import_loader: Callable[[Path], CsvImportResult] | None = None,
        csv_storage_path: Path | None = None,
        pdf_import_loader: Callable[[Path, bool, Callable[[str, int], None] | None], PdfImportResult] | None = None,
        csv_files_loader: Callable[[], list[Path]] | None = None,
        active_csv_loader: Callable[[], Path | None] | None = None,
        csv_switcher: Callable[[Path], CsvDatasetResult] | None = None,
        csv_upload_handler: Callable[[Path], CsvDatasetResult] | None = None,
        csv_delete_handler: Callable[[Path], CsvDatasetResult | None] | None = None,
        ai_scope_loader: Callable[[], str] | None = None,
        ai_scope_saver: Callable[[str], None] | None = None,
        current_language_loader: Callable[[], str] | None = None,
    ) -> None:
        self._service = service
        self._example_generator = example_generator
        self._meaning_translator = meaning_translator
        self._csv_import_loader = csv_import_loader
        self._csv_storage_path = csv_storage_path
        self._pdf_import_loader = pdf_import_loader
        self._csv_files_loader = csv_files_loader
        self._active_csv_loader = active_csv_loader
        self._csv_switcher = csv_switcher
        self._csv_upload_handler = csv_upload_handler
        self._csv_delete_handler = csv_delete_handler
        self._ai_scope_saver = ai_scope_saver
        self._study_session = StudySessionController(service)
        self._selected_id: str | None = None
        self._selected_ids: list[str] = []
        self._mode: str | None = None
        self._ai_scope = self._load_initial_ai_scope(ai_scope_loader)
        self._current_language = self._load_initial_language(current_language_loader)
        self._ui_bridge = UiThreadBridge(self)
        self._batch_state = "idle"
        self._batch_thread: QThread | None = None
        self._batch_worker: BatchWorker | None = None
        self._pdf_import_thread: QThread | None = None
        self._pdf_import_worker: PdfImportWorker | None = None
        self._model_check_thread: QThread | None = None
        self._model_check_worker: BackgroundTaskWorker | None = None
        self._csv_task_thread: QThread | None = None
        self._csv_task_worker: BackgroundTaskWorker | None = None
        self._llm_jobs = LlmJobPoller(example_generator)
        self._llm_poll_timer: QTimer | None = None
        self._llm_idle_close_timer: QTimer | None = None
        self._batch_action = ""
        self._batch_scope = ""
        self._batch_entries: list[WordEntry] = []
        self._batch_index = 0
        self._batch_parallel_limit_value = 2
        self._batch_mode = "parallel"
        self._batch_chunk_size = 1
        self._batch_completed_count = 0
        self._batch_finished = False
        self._batch_started_at = 0.0
        self._batch_updated_ids: list[str] = []
        self._batch_errors: list[str] = []
        self._pending_batch_message: tuple[str, str, str] | None = None
        self._user_model_warning_shown = False
        self._pdf_import_started_at = 0.0
        self._pdf_ai_started_at = 0.0
        self._pdf_progress_percent = 0
        self._pdf_import_finishing = False
        self._pending_pdf_import_result: PdfImportResult | None = None
        self._active_error_keys: set[tuple[str, str]] = set()
        self._recent_error_times: dict[tuple[str, str], float] = {}

        self._app = QApplication.instance() or QApplication(sys.argv)
        self._llm_poll_timer = QTimer()
        self._llm_poll_timer.setInterval(250)
        self._llm_poll_timer.timeout.connect(self._safe_slot(self._poll_llm_jobs))
        self._llm_idle_close_timer = QTimer()
        self._llm_idle_close_timer.setSingleShot(True)
        self._llm_idle_close_timer.setInterval(self._llm_idle_close_delay_ms())
        self._llm_idle_close_timer.timeout.connect(self._safe_slot(self._close_idle_llm_server))
        self._word_refresh_timer = QTimer()
        self._word_refresh_timer.setSingleShot(True)
        self._word_refresh_timer.setInterval(200)
        self._word_refresh_timer.timeout.connect(self._safe_slot(lambda: self._refresh_words(False)))
        self._install_exception_boundary()
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
        self._upload_csv_button: QPushButton | None = None
        self._upload_pdf_button: QPushButton | None = None
        self._model_check_button: QPushButton | None = None
        self._pause_button: QPushButton | None = None
        self._stop_button: QPushButton | None = None
        self._unknown_button: QPushButton | None = None
        self._known_button: QPushButton | None = None
        self._definitely_known_button: QPushButton | None = None
        self._explain_current_study_button: QPushButton | None = None
        self._edit_current_study_button: QPushButton | None = None
        self._delete_current_study_button: QPushButton | None = None
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

    @Slot()
    def run(self) -> None:
        self._window.show()
        self._bring_window_to_front()
        self._app.exec()

    @staticmethod
    def _load_initial_ai_scope(loader: Callable[[], str] | None) -> str:
        default = "AI 领域的译法"
        if loader is None:
            return default
        try:
            value = loader().strip()
        except Exception:
            return default
        return value or default

    @staticmethod
    def _load_initial_language(loader: Callable[[], str] | None) -> str:
        if loader is None:
            return ""
        try:
            return loader().strip()
        except Exception:
            return ""

    def _bring_window_to_front(self) -> None:
        self._window.showNormal()
        self._window.raise_()
        self._window.activateWindow()

    def _apply_style(self) -> None:
        apply_app_style(self._app)

    def _panel(self) -> QFrame:
        return create_panel()

    def _button(self, text: str, callback: Callable[[], None], variant: str = "") -> QPushButton:
        button = QPushButton(text)
        button.setMinimumWidth(40 if variant == "icon" else 76)
        if variant:
            button.setProperty("variant", variant)
        button.clicked.connect(self._safe_slot(callback))
        return button

    def _safe_slot(self, callback: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            try:
                return callback(*args, **kwargs)
            except TypeError:
                try:
                    return callback()
                except Exception as error:
                    self._handle_ui_exception(error)
            except Exception as error:
                self._handle_ui_exception(error)
            return None

        return wrapped

    def _install_exception_boundary(self) -> None:
        def excepthook(exc_type, exc_value, exc_traceback) -> None:
            self._handle_ui_exception(exc_value, exc_traceback)

        def thread_excepthook(args) -> None:
            self._handle_ui_exception(args.exc_value, args.exc_traceback)

        sys.excepthook = excepthook
        if hasattr(threading, "excepthook"):
            threading.excepthook = thread_excepthook

    def _handle_ui_exception(self, error: BaseException, exc_traceback=None) -> None:
        traceback.print_exception(type(error), error, exc_traceback or error.__traceback__)
        try:
            detail = "".join(traceback.format_exception(type(error), error, exc_traceback or error.__traceback__))
            self._show_error_message("操作失败", str(error) or error.__class__.__name__, detail=detail)
        except Exception:
            pass

    def _show_error_message(
        self,
        title: str,
        message: str,
        parent: QWidget | None = None,
        detail: str = "",
    ) -> None:
        key = (title, message)
        active_keys = getattr(self, "_active_error_keys", None)
        if active_keys is None:
            active_keys = set()
            self._active_error_keys = active_keys
        recent_times = getattr(self, "_recent_error_times", None)
        if recent_times is None:
            recent_times = {}
            self._recent_error_times = recent_times
        now = time.monotonic()
        if key in active_keys or now - recent_times.get(key, 0.0) < 5.0:
            return
        active_keys.add(key)
        recent_times[key] = now
        box = QMessageBox(parent or self._window)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(title)
        box.setText(message)
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        if detail and detail != message:
            box.setDetailedText(detail)
        try:
            box.exec()
        finally:
            active_keys.discard(key)

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
        subtitle = self._meta_label("选择一个页面开始")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        cards = QHBoxLayout()
        cards.setSpacing(24)
        counts = self._study_session.pool_counts()
        cards.addWidget(self._home_card("学习", f"{counts['learning']} 个单词", "开始学习", lambda: self._show_mode("learning")))
        cards.addWidget(self._home_card("复习", f"{counts['review']} 个单词", "开始复习", lambda: self._show_mode("review")))
        cards.addWidget(self._home_card("词表", f"{counts['total']} 个单词", "查看词表", self._show_word_list))
        layout.addLayout(cards, 1)
        layout.addWidget(self._model_status_panel())

        footer = QHBoxLayout()
        footer.addStretch()
        footer.addWidget(self._button("重置学习进度", self._confirm_reset_progress, "danger"))
        layout.addLayout(footer)
        self._set_page(root)

    def _csv_selector_widget(self) -> QHBoxLayout | None:
        if self._csv_files_loader is None or self._active_csv_loader is None or self._csv_switcher is None:
            return None
        row = QHBoxLayout()
        active = self._active_csv_loader()
        row.addWidget(QLabel(f"当前词表：{active.name if active else '无'}"))
        row.addWidget(self._button("管理词表", self._show_csv_manager, "primary"))
        row.addStretch()
        return row

    def _show_csv_manager(self) -> None:
        if self._csv_files_loader is None or self._active_csv_loader is None:
            return
        dialog = QDialog(self._window)
        dialog.setWindowTitle("管理词表")
        dialog.resize(680, 420)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("管理词表")
        title.setObjectName("title")
        layout.addWidget(title)

        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["CSV", "状态", "路径"])
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.setColumnWidth(0, 220)
        table.setColumnWidth(1, 90)
        self._refresh_csv_manager_table(table)
        layout.addWidget(table, 1)

        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self._button("刷新", lambda: self._refresh_csv_manager_table(table)))
        actions.addWidget(self._button("关闭", dialog.reject))
        actions.addWidget(self._button("选择", lambda: self._select_csv_from_manager(dialog, table), "primary"))
        actions.addWidget(self._button("删除", lambda: self._delete_csv_from_manager(dialog, table), "danger"))
        layout.addLayout(actions)
        dialog.exec()

    def _refresh_csv_manager_table(self, table: QTableWidget) -> None:
        if self._csv_files_loader is None or self._active_csv_loader is None:
            return
        selected_path = self._selected_csv_path_from_table(table)
        files = self._csv_files_loader()
        active = self._active_csv_loader()
        table.setRowCount(len(files))
        selected_row = -1
        active_row = -1
        for row, path in enumerate(files):
            values = [path.name, "当前" if active == path else "", str(path)]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                table.setItem(row, column, item)
            if selected_path == path:
                selected_row = row
            if active == path:
                active_row = row
        table.clearSelection()
        if selected_row >= 0:
            table.selectRow(selected_row)
        elif active_row >= 0:
            table.selectRow(active_row)

    @staticmethod
    def _selected_csv_path_from_table(table: QTableWidget) -> Path | None:
        selected = table.selectionModel().selectedRows()
        if not selected:
            return None
        item = table.item(selected[0].row(), 0)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return Path(str(value)) if value else None

    def _select_csv_from_manager(self, dialog: QDialog, table: QTableWidget) -> None:
        csv_path = self._selected_csv_path_from_table(table)
        if csv_path is None:
            QMessageBox.information(dialog, "未选择词表", "请先选择一个词表。")
            return
        if self._csv_switcher is None:
            return
        dialog.accept()
        self._start_csv_task("switch_csv", lambda: self._csv_switcher(csv_path))

    def _delete_csv_from_manager(self, dialog: QDialog, table: QTableWidget) -> None:
        csv_path = self._selected_csv_path_from_table(table)
        if csv_path is None:
            QMessageBox.information(dialog, "未选择词表", "请先选择一个词表。")
            return
        if self._csv_delete_handler is None:
            QMessageBox.information(dialog, "无法删除词表", "此功能暂不可用。")
            return
        answer = QMessageBox.question(
            dialog,
            "确认删除词表",
            f"确定删除 {csv_path.name} 吗？\n这会同时删除它对应的数据库和学习记录。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        dialog.accept()
        self._start_csv_task("delete_csv", lambda: self._csv_delete_handler(csv_path))

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
        self._model_check_button = self._button("检查模型", self._check_model, "primary")
        layout.addWidget(self._model_check_button)
        return panel

    def _model_status_text(self) -> str:
        if self._example_generator is None:
            return "AI 模型：未配置。学习、复习和 Argos 释义可正常使用，AI 例句补全暂不可用。"
        if not hasattr(self._example_generator, "model_status"):
            return "AI 模型：当前模型服务不支持状态检查。"
        try:
            status = self._example_generator.model_status()
        except Exception as error:
            return f"AI 模型：配置需要处理。{error}"
        path = getattr(status, "path", None)
        device_text = self._device_status_text()
        if path is None:
            return (
                "AI 模型：未找到。点击“检查模型”可下载默认 Hugging Face 模型；普通学习和复习不受影响。"
                f"\n{device_text}"
            )
        if getattr(status, "is_user_model", False):
            model_text = f"AI 模型：正在使用自定义模型 {path.name}，兼容性需自行确认。"
        else:
            model_text = f"AI 模型：正在使用默认模型 {path.name}。"
        return f"{model_text}\n{device_text}"

    def _device_status_text(self) -> str:
        if self._example_generator is None:
            return "运行设备：未配置。"
        if not hasattr(self._example_generator, "device_status"):
            return "运行设备：当前模型服务不支持设备检查。"
        try:
            status = self._example_generator.device_status()
        except Exception as error:
            return f"运行设备：检查失败。{error}"
        detected = self._device_label(getattr(status, "detected", "cpu"))
        selected = getattr(status, "selected", None)
        error = getattr(status, "error", "")
        if error:
            return f"运行设备：检测到 {detected}，当前不可用。{error}"
        return self._device_summary_text(status)

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

    def _device_summary_text(self, device: object) -> str:
        requested = self._device_label(getattr(device, "requested", "auto"))
        detected = self._device_label(getattr(device, "detected", "cpu"))
        selected = self._device_label(getattr(device, "selected", None))
        supported = getattr(device, "gpu_offload_supported", None)
        if supported is None:
            offload = "未知"
        else:
            offload = "支持" if supported else "不支持"
        error = str(getattr(device, "error", "")).strip()
        text = f"运行设备：检测到 {detected}；GPU 加速：{offload}；将使用 {selected}。"
        if requested != "Auto":
            text = f"{text}\n已指定设备：{requested}。"
        if error:
            text = f"{text}\n设备检查错误：{error}"
        return text

    def _check_model(self) -> None:
        if self._model_check_thread is not None:
            QMessageBox.information(self._window, "模型检查", "模型检查正在运行。")
            return
        if self._example_generator is None:
            QMessageBox.information(self._window, "模型检查", "未配置本地 AI 模型服务。")
            return
        if not hasattr(self._example_generator, "check_model_runtime"):
            QMessageBox.information(self._window, "模型检查", "当前模型服务不支持模型检查。")
            return
        if not self._confirm_model_download("检查模型"):
            return
        self._cancel_llm_idle_close()
        self._model_check_thread = QThread(self._window)
        self._model_check_worker = BackgroundTaskWorker("model_check", self._example_generator.check_model_runtime)
        self._model_check_worker.moveToThread(self._model_check_thread)
        self._model_check_thread.started.connect(self._model_check_worker.run)
        self._model_check_worker.finished.connect(
            self._ui_bridge.on_model_check_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._model_check_worker.failed.connect(
            self._ui_bridge.on_model_check_failed,
            Qt.ConnectionType.QueuedConnection,
        )
        self._model_check_worker.finished.connect(self._model_check_worker.deleteLater)
        self._model_check_worker.failed.connect(self._model_check_worker.deleteLater)
        self._model_check_worker.finished.connect(self._model_check_thread.quit)
        self._model_check_worker.failed.connect(self._model_check_thread.quit)
        self._model_check_thread.finished.connect(
            self._ui_bridge.on_model_check_thread_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._model_check_thread.finished.connect(self._model_check_thread.deleteLater)
        if self._model_check_button is not None:
            self._model_check_button.setText("检查中")
            self._model_check_button.setEnabled(False)
        self._model_check_thread.start()

    def _on_model_check_finished(self, _name: str, result: object) -> None:
        status = result.model
        device = result.device
        path = getattr(status, "path", None)
        device_line = f"\n{self._device_summary_text(device)}"
        smoke_line = "\n运行测试：通过。"
        if path is None:
            QMessageBox.information(self._window, "模型检查", "未找到可用模型。")
        elif getattr(status, "is_user_model", False):
            QMessageBox.warning(
                self._window,
                "模型检查",
                f"当前使用自定义模型：{path.name}"
                f"{device_line}{smoke_line}\n自定义模型的输出格式和运行兼容性可能不稳定。",
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

    def _on_model_check_failed(self, _name: str, message: str) -> None:
        self._show_error_message("模型检查失败", message)
        self._show_home()

    def _on_model_check_thread_finished(self) -> None:
        self._model_check_thread = None
        self._model_check_worker = None
        if self._model_check_button is not None:
            self._model_check_button.setText("检查模型")
            self._model_check_button.setEnabled(True)

    def _confirm_reset_progress(self) -> None:
        confirmation_text = "我确认重置学习进度"
        text, accepted = QInputDialog.getText(
            self._window,
            "确认重置",
            f"此操作会保留词表内容、例句和例句翻译，只清空学习进度。\n请输入“{confirmation_text}”以继续：",
        )
        if not accepted:
            return
        if text.strip() != confirmation_text:
            QMessageBox.information(self._window, "未重置", "确认文本不匹配，学习进度未重置。")
            return
        self._service.reset_progress()
        self._study_session.clear_last_session()
        QMessageBox.information(
            self._window,
            "已重置",
            "学习进度已重置，词表内容、例句和例句翻译已保留。",
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
        title_bar.setSpacing(10)
        title = QLabel("学习" if mode == "learning" else "复习")
        title.setObjectName("homeTitle")
        title_bar.addWidget(title)
        title_bar.addStretch()
        self._explain_current_study_button = self._button("AI 速解", self._explain_current_study_word, "primary")
        self._explain_current_study_button.setToolTip("用本地 AI 解释当前词条")
        self._edit_current_study_button = self._button("编辑词条", self._edit_current_study_word)
        self._edit_current_study_button.setToolTip("编辑当前词条并保存到数据库")
        self._delete_current_study_button = self._button("删除词条", self._confirm_delete_current_study_word, "danger")
        self._delete_current_study_button.setToolTip("从当前词库中删除这个词条")
        title_bar.addWidget(self._explain_current_study_button)
        title_bar.addWidget(self._edit_current_study_button)
        title_bar.addWidget(self._delete_current_study_button)
        title_bar.addWidget(self._button("返回首页", self._show_home))
        layout.addLayout(title_bar)

        layout.addStretch()
        card = self._panel()
        self._study_card = card
        card.mousePressEvent = lambda event: self._toggle_translation_reveal()
        card.setFixedSize(860, 560)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(34, 26, 34, 30)
        card_layout.setSpacing(10)

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
        nav_row.setSpacing(12)
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
        actions.setHorizontalSpacing(12)
        actions.setVerticalSpacing(10)
        actions.setContentsMargins(90, 0, 90, 0)
        self._unknown_button = self._button("不认识", self._mark_unknown, "danger")
        self._known_button = self._button("认识", self._mark_known, "primary")
        self._definitely_known_button = self._button("很熟", self._mark_definitely_known, "primary")
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
        self._explain_current_study_button.setVisible(has_entry)
        self._edit_current_study_button.setVisible(has_entry)
        self._delete_current_study_button.setVisible(has_entry)
        self._previous_button.setVisible(state.can_show_previous)
        self._next_button.setVisible(state.can_show_next)

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
            QMessageBox.information(self._window, "暂无词条", "当前没有可显示的词条。")
            return
        self._render_study_card(state)

    def _confirm_delete_current_study_word(self) -> None:
        state = self._current_card_state
        entry = state.entry if state is not None else None
        if entry is None:
            return
        answer = QMessageBox.question(
            self._window,
            "确认删除词条",
            f"确定要从当前词库中删除“{entry.word}”吗？\n此操作会删除该词条的学习记录和例句。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        next_state = self._study_session.delete_current()
        if next_state is not None:
            self._render_study_card(next_state)

    def _edit_current_study_word(self) -> None:
        state = self._current_card_state
        entry = state.entry if state is not None else None
        if entry is None:
            return
        self._show_word_editor(entry, refresh_after_save=lambda updated: self._refresh_current_study_entry(updated.id))

    def _explain_current_study_word(self) -> None:
        state = self._current_card_state
        entry = state.entry if state is not None else None
        if entry is None:
            return
        if not self._prepare_exclusive_llm_task("AI 速解"):
            return
        if self._example_generator is None:
            QMessageBox.information(self._window, "无法使用 AI 速解", "未配置本地 AI 模型服务。")
            return
        if not self._confirm_model_download("AI 速解"):
            return
        if not self._can_submit_llm_jobs():
            QMessageBox.information(
                self._window,
                "无法使用 AI 速解",
                "当前模型服务不支持后台任务。",
            )
            return
        self._warn_if_using_user_model()

        try:
            self._llm_jobs.submit_explain(entry, self._ai_scope, self._current_language)
        except Exception as error:
            self._show_error_message("AI 速解失败", str(error))
            return
        if self._explain_current_study_button is not None:
            self._explain_current_study_button.setText("解释中")
            self._explain_current_study_button.setEnabled(False)
        self._ensure_llm_polling()

    def _finish_explain_job(self, schedule_idle_close: bool = True) -> None:
        self._llm_jobs.finish_explain()
        if self._explain_current_study_button is not None:
            self._explain_current_study_button.setText("AI 速解")
            self._explain_current_study_button.setEnabled(True)
        if schedule_idle_close:
            self._stop_llm_polling_if_idle()

    def _show_explanation_dialog(self, word: str, explanation: str) -> None:
        dialog = QDialog(self._window)
        dialog.setWindowTitle(f"AI 速解 - {word}")
        dialog.setObjectName("appSurface")
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.resize(500, 320)
        dialog.setMinimumSize(420, 260)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        title = QLabel(word)
        title.setObjectName("title")
        layout.addWidget(title)
        text = QPlainTextEdit(explanation)
        text.setReadOnly(True)
        layout.addWidget(text, 1)
        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self._button("关闭", dialog.accept, "primary"))
        layout.addLayout(actions)
        dialog.exec()

    def _refresh_current_study_entry(self, entry_id: str) -> None:
        self._study_session.reload()
        self._render_study_card(self._study_session.show_entry_by_id(entry_id, record_history=False))

    def _set_ai_scope(self, text: str) -> None:
        self._ai_scope = text.strip()
        if self._ai_scope_saver is None:
            return
        try:
            self._ai_scope_saver(self._ai_scope)
        except Exception:
            return

    def _show_previous_word(self) -> None:
        state = self._study_session.show_previous_word()
        if state is not None:
            self._render_study_card(state)

    def _continue_from_history(self) -> None:
        if self._current_card_state is None or not self._current_card_state.history_view:
            return
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
        title_bar.addWidget(self._button("返回首页", self._show_home))
        layout.addLayout(title_bar)
        selector = self._csv_selector_widget()
        if selector is not None:
            layout.addLayout(selector)

        panel = self._panel()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(22, 22, 22, 22)
        panel_layout.setSpacing(12)

        search_row = QHBoxLayout()
        search_row.setSpacing(10)
        search_row.addWidget(QLabel("搜索"))
        self._search_input = QLineEdit()
        self._search_input.textChanged.connect(self._safe_slot(lambda _text: self._schedule_words_refresh()))
        search_row.addWidget(self._search_input, 1)
        search_row.addWidget(self._button("刷新", self._refresh_current_dataset))
        panel_layout.addLayout(search_row)

        scope_row = QHBoxLayout()
        scope_row.setSpacing(10)
        scope_row.addWidget(QLabel("AI 参考语境"))
        self._scope_input = QLineEdit(self._ai_scope)
        self._scope_input.textChanged.connect(self._safe_slot(self._set_ai_scope))
        scope_row.addWidget(self._scope_input, 1)
        scope_row.addWidget(self._button("新增词条", self._show_add_word_dialog, "primary"))
        panel_layout.addLayout(scope_row)

        self._table = QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels(["#", "单词", "释义", "词形", "例句", "例句翻译", "频率", "状态", "复习"])
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
        self._table.itemSelectionChanged.connect(self._safe_slot(self._on_table_selection_changed))
        self._table.itemDoubleClicked.connect(self._safe_slot(lambda item: self._edit_selected_word(item.row())))
        panel_layout.addWidget(self._table, 1)

        footer = QVBoxLayout()
        footer.setSpacing(8)
        status_row = QHBoxLayout()
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self._count_label = self._meta_label()
        status_row.addWidget(self._count_label, 1)
        footer.addLayout(status_row)
        self._supplement_button = self._button("AI 补全例句", self._supplement_selected_example)
        self._correct_button = self._button("补充释义", self._supplement_selected_meaning)
        self._upload_csv_button = self._button("导入 CSV", self._upload_csv)
        self._upload_pdf_button = self._button("导入 PDF", self._upload_pdf)
        self._pause_button = self._button("暂停", self._toggle_batch_pause)
        self._stop_button = self._button("停止", self._stop_batch)
        delete_button = self._button("删除选中", self._delete_selected, "danger")
        self._supplement_button.setMinimumWidth(126)
        self._correct_button.setMinimumWidth(126)
        self._upload_csv_button.setMinimumWidth(84)
        self._upload_pdf_button.setMinimumWidth(84)
        self._pause_button.setMinimumWidth(70)
        self._stop_button.setMinimumWidth(70)
        delete_button.setMinimumWidth(90)
        action_row.addStretch()
        action_row.addWidget(self._upload_csv_button)
        action_row.addWidget(self._upload_pdf_button)
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

    def _schedule_words_refresh(self) -> None:
        if self._word_refresh_timer.isActive():
            self._word_refresh_timer.stop()
        self._word_refresh_timer.start()

    def _refresh_current_dataset(self) -> None:
        if self._active_csv_loader is None or self._csv_switcher is None:
            self._refresh_words(False)
            return
        if self._csv_task_thread is not None:
            QMessageBox.information(self._window, "词表处理中", "请等待当前词表处理完成。")
            return
        active = self._active_csv_loader()
        if active is None:
            self._study_session.clear_last_session()
            self._study_session.reset()
            self._selected_id = None
            self._selected_ids = []
            self._show_word_list()
            return
        self._start_csv_task("refresh_csv", lambda: self._csv_switcher(active))

    def _upload_csv(self) -> None:
        if self._csv_upload_handler is None:
            QMessageBox.information(self._window, "无法导入 CSV", "此功能暂不可用。")
            return
        if self._csv_task_thread is not None:
            QMessageBox.information(self._window, "词表处理中", "请等待当前词表处理完成。")
            return
        file_name, _selected_filter = QFileDialog.getOpenFileName(
            self._window,
            "导入 CSV",
            "",
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if not file_name:
            return
        source_path = Path(file_name)
        self._start_csv_task("upload_csv", lambda: self._csv_upload_handler(source_path))

    def _start_csv_task(self, name: str, task: Callable[[], object]) -> None:
        if self._csv_task_thread is not None:
            QMessageBox.information(self._window, "词表处理中", "请等待当前词表处理完成。")
            return
        self._csv_task_thread = QThread(self._window)
        self._csv_task_worker = BackgroundTaskWorker(name, task)
        self._csv_task_worker.moveToThread(self._csv_task_thread)
        self._csv_task_thread.started.connect(self._csv_task_worker.run)
        self._csv_task_worker.finished.connect(
            self._ui_bridge.on_csv_task_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._csv_task_worker.failed.connect(
            self._ui_bridge.on_csv_task_failed,
            Qt.ConnectionType.QueuedConnection,
        )
        self._csv_task_worker.finished.connect(self._csv_task_worker.deleteLater)
        self._csv_task_worker.failed.connect(self._csv_task_worker.deleteLater)
        self._csv_task_worker.finished.connect(self._csv_task_thread.quit)
        self._csv_task_worker.failed.connect(self._csv_task_thread.quit)
        self._csv_task_thread.finished.connect(
            self._ui_bridge.on_csv_task_thread_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._csv_task_thread.finished.connect(self._csv_task_thread.deleteLater)
        self._set_csv_task_busy(True, self._csv_task_message(name))
        self._csv_task_thread.start()

    @staticmethod
    def _csv_task_message(name: str) -> str:
        messages = {
            "upload_csv": "正在导入词表",
            "switch_csv": "正在切换词表",
            "delete_csv": "正在删除词表",
            "refresh_csv": "正在刷新词表",
        }
        return messages.get(name, "正在处理词表")

    def _set_csv_task_busy(self, busy: bool, message: str = "") -> None:
        if self._upload_csv_button is not None:
            self._upload_csv_button.setEnabled(not busy)
        if self._upload_pdf_button is not None:
            self._upload_pdf_button.setEnabled(not busy)
        if self._progress is not None:
            self._progress.setRange(0, 0 if busy else 100)
            if not busy:
                self._progress.setValue(0)
        if self._batch_status_label is not None:
            self._batch_status_label.setText(message)

    def _on_csv_task_finished(self, name: str, result: object) -> None:
        self._study_session.clear_last_session()
        self._study_session.reset()
        self._selected_id = None
        self._selected_ids = []
        if result is None:
            self._current_language = ""
            QMessageBox.information(self._window, "词表已删除", "当前词表已删除，没有其他可切换的词表。")
            self._show_word_list()
            return

        self._current_language = result.language
        if name == "upload_csv":
            self._refresh_words(False)
            QMessageBox.information(
                self._window,
                "词表导入完成",
                f"当前词表：{result.csv_path.name}\n识别语言：{result.language}。\n已导入或更新 {result.imported_count} 个词条。",
            )
            return
        if name == "switch_csv":
            QMessageBox.information(
                self._window,
                "词表已切换",
                f"当前词表：{result.csv_path.name}\n语言：{result.language}\n已导入或更新 {result.imported_count} 个词条。",
            )
            self._show_word_list()
            return
        if name == "refresh_csv":
            self._show_word_list()
            return
        if name == "delete_csv":
            QMessageBox.information(
                self._window,
                "词表已删除",
                f"当前词表已删除。\n已切换到：{result.csv_path.name}\n已导入或更新 {result.imported_count} 个词条。",
            )
            self._show_word_list()

    def _on_csv_task_failed(self, name: str, message: str) -> None:
        titles = {
            "upload_csv": "词表导入失败",
            "switch_csv": "词表切换失败",
            "delete_csv": "词表删除失败",
            "refresh_csv": "词表刷新失败",
        }
        self._show_error_message(titles.get(name, "词表处理失败"), message)

    def _on_csv_task_thread_finished(self) -> None:
        self._csv_task_thread = None
        self._csv_task_worker = None
        self._set_csv_task_busy(False)

    def _upload_pdf(self) -> None:
        if self._pdf_import_loader is None:
            QMessageBox.information(self._window, "无法导入 PDF", "此功能暂不可用。")
            return
        file_name, _selected_filter = QFileDialog.getOpenFileName(
            self._window,
            "导入 PDF",
            "",
            "PDF 文件 (*.pdf);;所有文件 (*)",
        )
        if not file_name:
            return
        use_llm_cleanup = self._should_use_llm_for_pdf_cleanup()
        self._start_pdf_import(Path(file_name), use_llm_cleanup)

    def _start_pdf_import(self, pdf_path: Path, use_llm_cleanup: bool) -> None:
        if self._pdf_import_loader is None:
            return
        if self._pdf_import_thread is not None:
            QMessageBox.information(self._window, "PDF 导入中", "请等待当前 PDF 导入完成。")
            return
        if use_llm_cleanup:
            if not self._prepare_exclusive_llm_task("PDF 词表优化"):
                return
            self._cancel_llm_idle_close()
        self._pdf_import_started_at = time.monotonic()
        self._pdf_ai_started_at = 0.0
        self._pdf_progress_percent = 0
        self._pdf_import_finishing = False
        self._set_pdf_progress("准备导入 PDF", 0)
        if self._upload_pdf_button is not None:
            self._upload_pdf_button.setEnabled(False)

        self._pdf_import_thread = QThread(self._window)
        self._pdf_import_worker = PdfImportWorker(pdf_path, use_llm_cleanup, self._pdf_import_loader)
        self._pdf_import_worker.moveToThread(self._pdf_import_thread)
        self._pdf_import_thread.started.connect(self._pdf_import_worker.run)
        self._pdf_import_worker.progress_changed.connect(
            self._ui_bridge.set_pdf_progress,
            Qt.ConnectionType.QueuedConnection,
        )
        self._pdf_import_worker.finished.connect(
            self._ui_bridge.on_pdf_import_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._pdf_import_worker.failed.connect(
            self._ui_bridge.on_pdf_import_failed,
            Qt.ConnectionType.QueuedConnection,
        )
        self._pdf_import_worker.finished.connect(self._pdf_import_worker.deleteLater)
        self._pdf_import_worker.failed.connect(self._pdf_import_worker.deleteLater)
        self._pdf_import_worker.finished.connect(self._pdf_import_thread.quit)
        self._pdf_import_worker.failed.connect(self._pdf_import_thread.quit)
        self._pdf_import_thread.finished.connect(
            self._ui_bridge.on_pdf_import_thread_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._pdf_import_thread.finished.connect(self._pdf_import_thread.deleteLater)
        self._pdf_import_thread.start()

    def _on_pdf_import_finished(self, result: PdfImportResult) -> None:
        self._pdf_import_finishing = True
        self._pending_pdf_import_result = result
        self._set_pdf_progress("PDF 导入完成，正在刷新词表", 100)

    def _finalize_pdf_import(self, result: PdfImportResult) -> None:
        self._set_pdf_progress("PDF 导入完成，正在刷新词表", 100)
        self._study_session.clear_last_session()
        self._study_session.reset()
        self._selected_id = None
        self._selected_ids = []
        self._current_language = result.language
        self._refresh_words(False)
        QMessageBox.information(
            self._window,
            "PDF 导入完成",
            f"识别语言：{result.language}。\n已生成词表 {result.csv_path.name}，导入或更新 {result.imported_count} 个词条。",
        )
        self._clear_batch_progress("PDF 导入完成")
        self._pdf_import_started_at = 0.0
        self._pdf_ai_started_at = 0.0
        self._pdf_progress_percent = 0
        self._pdf_import_finishing = False
        self._pending_pdf_import_result = None
        self._bring_window_to_front()

    def _on_pdf_import_failed(self, message: str) -> None:
        self._pdf_import_started_at = 0.0
        self._pdf_ai_started_at = 0.0
        self._pdf_progress_percent = 0
        self._pdf_import_finishing = False
        self._pending_pdf_import_result = None
        self._clear_batch_progress("PDF 导入失败")
        self._show_error_message("PDF 导入失败", message)
        self._bring_window_to_front()

    def _on_pdf_import_thread_finished(self) -> None:
        if self._upload_pdf_button is not None:
            self._upload_pdf_button.setEnabled(True)
        self._pdf_import_worker = None
        self._pdf_import_thread = None
        result = self._pending_pdf_import_result
        if result is not None:
            QTimer.singleShot(0, lambda result=result: self._finalize_pdf_import(result))

    def _set_pdf_progress(self, message: str, percent: int) -> None:
        percent = max(0, min(100, int(percent)))
        if self._pdf_import_started_at > 0 and percent < self._pdf_progress_percent:
            return
        if self._pdf_import_finishing and percent < 100:
            return
        self._pdf_progress_percent = percent
        if self._progress is not None:
            self._progress.setValue(percent)
        if self._batch_status_label is not None:
            self._batch_status_label.setText(f"PDF 导入：{message} ({percent}%) | {self._pdf_eta_text(message, percent)}")

    def _pdf_eta_text(self, message: str, percent: int) -> str:
        if percent >= 100:
            elapsed = time.monotonic() - self._pdf_import_started_at
            return f"耗时：{self._format_duration(elapsed)}"
        batch_progress = self._pdf_ai_batch_progress(message)
        if batch_progress is None:
            return "当前阶段：进行中"
        done, total, is_completed_update = batch_progress
        if total <= 0:
            return "AI 剩余时间：估算中"
        if self._pdf_ai_started_at <= 0:
            self._pdf_ai_started_at = time.monotonic()
        completed = done if is_completed_update else done - 1
        if completed <= 0:
            return f"AI 进度：{done}/{total} | 剩余时间：估算中"
        elapsed = time.monotonic() - self._pdf_ai_started_at
        remaining = elapsed * (total - completed) / max(1, completed)
        return f"AI 进度：{done}/{total} | 剩余时间：{self._format_duration(remaining)}"

    @staticmethod
    def _pdf_ai_batch_progress(message: str) -> tuple[int, int, bool] | None:
        match = re.search(r"AI (?:检查|审阅) (?:词表|CSV)：(?:已完成 )?(\d+)/(\d+)", message)
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2)), "已完成" in message

    def _should_use_llm_for_pdf_cleanup(self) -> bool:
        if self._example_generator is None:
            return False
        if not hasattr(self._example_generator, "model_status"):
            return False
        answer = QMessageBox.question(
            self._window,
            "PDF 词表优化",
            "是否用本地 AI 模型检查 PDF 生成的词表，并移除不是单词或词组的内容？\n\n"
            "选择“否”将直接使用未经 AI 检查的词表。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        return self._confirm_model_download(
            "PDF 词表优化",
            no_text="选择“否”将使用未经 AI 检查的词表。",
        )

    def _has_local_llm_model(self) -> bool:
        if self._example_generator is None or not hasattr(self._example_generator, "model_status"):
            return False
        try:
            status = self._example_generator.model_status()
        except Exception:
            return False
        return getattr(status, "path", None) is not None

    def _confirm_model_download(self, action: str, no_text: str = "") -> bool:
        if self._has_local_llm_model():
            return True
        message = (
            f"{action}需要本地 AI 模型，但本地模型文件夹中还没有 .gguf 模型。\n"
            "是否下载默认模型？下载完成后会保存在本地模型文件夹，以后 AI 例句补全和 PDF 词表优化都会复用它。"
        )
        if no_text:
            message = f"{message}\n\n{no_text}"
        answer = QMessageBox.question(
            self._window,
            "下载默认模型？",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _visible_word_entries(self, query: str = "") -> list[WordEntry]:
        if self._mode in {"learning", "review"}:
            return self._study_session.mode_entries(query)
        return self._service.list_words(query)

    def _refresh_words(self, reload_current: bool = True) -> None:
        if self._table is None:
            return
        selected_ids = self._selected_entry_ids()
        self._table.blockSignals(True)
        self._table.setUpdatesEnabled(False)
        try:
            self._table.setRowCount(0)
            query = self._search_input.text() if self._search_input is not None else ""
            entries = self._visible_word_entries(query)
            self._table.setRowCount(len(entries))
            for row, entry in enumerate(entries):
                self._insert_entry(row, entry)
        finally:
            self._table.setUpdatesEnabled(True)
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

    def _show_word_editor(self, entry: WordEntry, refresh_after_save: Callable[[WordEntry], None] | None = None) -> None:
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
        form.addWidget(self._form_label("例句翻译"), 4, 0, Qt.AlignmentFlag.AlignTop)
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
                self._show_error_message("保存失败", str(error), parent=dialog)
                return
            self._selected_id = updated.id if updated else entry.id
            self._selected_ids = [self._selected_id]
            if updated is not None and refresh_after_save is not None:
                refresh_after_save(updated)
            else:
                self._refresh_words(False)
                self._restore_table_selection(self._selected_ids)
            dialog.accept()

        actions.addWidget(self._button("保存", save, "primary"))
        form.addLayout(actions, 5, 0, 1, 2)
        layout.addWidget(panel)
        dialog.exec()

    def _show_add_word_dialog(self) -> None:
        dialog = QDialog(self._window)
        dialog.setWindowTitle("新增词条")
        dialog.setObjectName("appSurface")
        dialog.resize(700, 560)
        dialog.setMinimumSize(620, 500)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("新增词条")
        title.setObjectName("homeTitle")
        subtitle = self._meta_label("新词会添加到词表顶部，并自动调整顺序。")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        panel = self._panel()
        form = QGridLayout(panel)
        form.setContentsMargins(24, 24, 24, 24)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        form.setColumnStretch(1, 1)

        word_input = QLineEdit()
        meaning_input = QLineEdit()
        forms_input = QLineEdit()
        example_input = QPlainTextEdit()
        example_cn_input = QPlainTextEdit()
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
        form.addWidget(self._form_label("例句翻译"), 4, 0, Qt.AlignmentFlag.AlignTop)
        form.addWidget(example_cn_input, 4, 1)

        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self._button("取消", dialog.reject))

        def save() -> None:
            word = word_input.text().strip()
            meaning = meaning_input.text().strip()
            forms = forms_input.text().strip()
            if not word:
                QMessageBox.information(dialog, "无法新增", "单词不能为空。")
                return
            try:
                entry = self._service.insert_word_at_front(
                    word,
                    meaning,
                    forms,
                    example_input.toPlainText().strip(),
                    example_cn_input.toPlainText().strip(),
                )
            except Exception as error:
                self._show_error_message("新增失败", str(error), parent=dialog)
                return
            self._selected_id = entry.id
            self._selected_ids = [entry.id]
            self._refresh_words(False)
            self._restore_table_selection(self._selected_ids)
            dialog.accept()

        actions.addWidget(self._button("新增", save, "primary"))
        form.addLayout(actions, 5, 0, 1, 2)
        layout.addWidget(panel)
        dialog.exec()

    def _supplement_selected_example(self) -> None:
        self._start_batch("补充")

    def _supplement_selected_meaning(self) -> None:
        self._start_batch("释义")

    def _start_batch(self, action: str) -> None:
        if self._batch_state != "idle":
            return
        action_label = self._batch_action_label(action)
        selected_ids = self._selected_entry_ids()
        if not selected_ids:
            QMessageBox.information(self._window, "未选择词条", "请先在列表中选择一个词条。")
            return
        entries = self._entries_by_id(selected_ids)
        if not entries:
            QMessageBox.information(self._window, f"{action_label}失败", "当前词条不存在。")
            return
        if action == "释义":
            if self._meaning_translator is None:
                QMessageBox.information(self._window, "无法补充释义", "未配置 Argos Translate 翻译器。")
                return
            self._start_threaded_batch(action, entries, self._meaning_translator)
            return

        if self._example_generator is None:
            QMessageBox.information(self._window, f"无法使用 AI {action_label}", "未配置本地 AI 模型服务。")
            return
        if not self._confirm_model_download(f"AI {action_label}"):
            return
        if not self._can_submit_llm_jobs():
            QMessageBox.information(
                self._window,
                f"无法使用 AI {action_label}",
                "当前模型服务不支持后台任务。",
            )
            return
        if not self._prepare_exclusive_llm_task(f"AI {action_label}"):
            return
        self._warn_if_using_user_model()

        scope = self._ai_scope
        self._set_batch_running()
        if action == "补充":
            self._supplement_button.setText(f"补全中 0/{len(entries)}")
            self._supplement_button.setEnabled(False)
        else:
            self._correct_button.setText(f"{action_label}中 0/{len(entries)}")
            self._correct_button.setEnabled(False)
        self._set_batch_progress(action, 0, len(entries), 0, 0.0)

        self._batch_action = action
        self._batch_scope = scope
        self._batch_entries = entries
        self._batch_index = 0
        strategy = self._recommended_batch_strategy(action)
        self._batch_mode = strategy["mode"]
        self._batch_chunk_size = strategy["batch_size"]
        self._batch_parallel_limit_value = strategy["parallelism"]
        self._batch_completed_count = 0
        self._batch_finished = False
        self._batch_started_at = time.monotonic()
        self._batch_updated_ids = []
        self._batch_errors = []
        self._llm_jobs.clear_batch_jobs()
        self._pump_batch_processes()

    def _start_threaded_batch(self, action: str, entries: list[WordEntry], generator) -> None:
        action_label = self._batch_action_label(action)
        self._set_batch_running()
        if action == "补充":
            self._supplement_button.setText(f"{action_label}中 0/{len(entries)}")
            self._supplement_button.setEnabled(False)
        else:
            self._correct_button.setText(f"{action_label}中 0/{len(entries)}")
            self._correct_button.setEnabled(False)
        self._set_batch_progress(action, 0, len(entries), 0, 0.0)
        self._batch_action = action
        self._batch_entries = entries
        self._batch_updated_ids = []
        self._batch_errors = []
        self._pending_batch_message = None

        thread = QThread()
        worker = BatchWorker(action, entries, self._ai_scope, generator, self._batch_control_state, self._current_language)
        worker.moveToThread(thread)
        worker.progress_changed.connect(self._ui_bridge.on_batch_progress)
        worker.finished.connect(self._ui_bridge.on_batch_finished)
        worker.failed.connect(self._ui_bridge.on_batch_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.started.connect(worker.run)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._ui_bridge.on_batch_thread_finished)
        self._batch_thread = thread
        self._batch_worker = worker
        thread.start()

    def _can_submit_llm_jobs(self) -> bool:
        return self._llm_jobs.can_submit_jobs()

    def _prepare_exclusive_llm_task(self, action: str) -> bool:
        if self._llm_jobs.is_idle():
            return True
        answer = QMessageBox.question(
            self._window,
            action,
            "当前已有 AI 任务正在运行。为了保证同一时间只加载一个本地模型实例，需要中断当前任务并关闭模型服务后再开始新任务。\n\n是否中断当前 AI 任务？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        self._abort_active_llm_jobs()
        return True

    def _abort_active_llm_jobs(self) -> None:
        try:
            self._llm_jobs.finish_explain()
            self._llm_jobs.clear_batch_jobs()
        except Exception:
            pass
        if self._batch_state in {"running", "paused"}:
            self._batch_state = "stopped"
            self._batch_finished = True
            self._set_batch_idle()
            self._clear_batch_progress("AI 任务已中断")
        if self._explain_current_study_button is not None:
            self._explain_current_study_button.setText("AI 速解")
            self._explain_current_study_button.setEnabled(True)
        if self._llm_poll_timer is not None and self._llm_poll_timer.isActive():
            self._llm_poll_timer.stop()
        if self._example_generator is not None and hasattr(self._example_generator, "close"):
            try:
                self._example_generator.close()
            except Exception:
                pass

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
            "使用自定义模型",
            "检测到本地模型文件夹中已有用户提供的 GGUF 模型，将优先使用该模型。\n"
            "不同模型的提示词格式、JSON 输出稳定性和运行兼容性可能不同，"
            "软件不保证自定义模型完全兼容。",
        )

    def _pump_batch_processes(self) -> None:
        if self._batch_finished:
            return
        if self._batch_state == "stopped":
            if not self._llm_jobs.has_batch_jobs():
                self._finish_isolated_batch()
            return
        if self._batch_state == "paused":
            return
        if self._batch_index >= len(self._batch_entries) and not self._llm_jobs.has_batch_jobs():
            self._finish_isolated_batch()
            return
        if self._example_generator is None:
            self._batch_errors.append("AI 模型服务已不可用。")
            self._finish_isolated_batch()
            return

        while (
            self._batch_state == "running"
            and self._batch_index < len(self._batch_entries)
            and self._llm_jobs.batch_job_count < self._batch_parallel_limit()
        ):
            entries = self._next_batch_entries()
            if not entries:
                break
            self._start_one_batch_process(entries)
            self._batch_index += len(entries)

    def _next_batch_entries(self) -> list[WordEntry]:
        if self._batch_action == "补充" and self._batch_mode == "batch":
            end = min(len(self._batch_entries), self._batch_index + self._batch_chunk_size)
            return self._batch_entries[self._batch_index:end]
        return [self._batch_entries[self._batch_index]]

    def _start_one_batch_process(self, entries: list[WordEntry]) -> None:
        self._on_batch_progress(
            self._batch_action,
            self._batch_completed_count,
            len(self._batch_entries),
            self._llm_jobs.batch_job_count,
            time.monotonic() - self._batch_started_at,
        )

        try:
            if self._batch_action == "补充" and self._batch_mode == "batch":
                self._llm_jobs.submit_supplement_batch_job(entries, self._batch_scope, self._current_language)
            else:
                self._llm_jobs.submit_batch_job(
                    self._batch_action,
                    entries[0],
                    self._batch_scope,
                    self._current_language,
                )
        except Exception as error:
            label = "、".join(entry.word for entry in entries[:3])
            if len(entries) > 3:
                label = f"{label} 等 {len(entries)} 条"
            self._batch_errors.append(f"{label}: {error}")
            self._batch_completed_count += len(entries)
            QTimer.singleShot(0, self._pump_batch_processes)
            return

        self._ensure_llm_polling()
        self._on_batch_progress(
            self._batch_action,
            self._batch_completed_count,
            len(self._batch_entries),
            self._llm_jobs.batch_job_count,
            time.monotonic() - self._batch_started_at,
        )

    def _ensure_llm_polling(self) -> None:
        self._cancel_llm_idle_close()
        if self._llm_poll_timer is not None and not self._llm_poll_timer.isActive():
            self._llm_poll_timer.start()

    def _stop_llm_polling_if_idle(self) -> None:
        if self._llm_poll_timer is not None and self._llm_jobs.is_idle():
            self._llm_poll_timer.stop()
            self._schedule_llm_idle_close()

    def _schedule_llm_idle_close(self) -> None:
        if self._llm_idle_close_timer is None:
            return
        if not self._llm_jobs.is_idle():
            return
        if self._model_check_thread is not None or self._pdf_import_thread is not None:
            return
        self._llm_idle_close_timer.start()

    def _cancel_llm_idle_close(self) -> None:
        if self._llm_idle_close_timer is not None and self._llm_idle_close_timer.isActive():
            self._llm_idle_close_timer.stop()

    def _close_idle_llm_server(self) -> None:
        if not self._llm_jobs.is_idle():
            return
        if self._model_check_thread is not None or self._pdf_import_thread is not None:
            return
        if self._example_generator is None or not hasattr(self._example_generator, "close"):
            return
        try:
            self._example_generator.close()
        except Exception:
            return

    @staticmethod
    def _llm_idle_close_delay_ms() -> int:
        raw_value = os.getenv("WORDPYCKET_LLM_IDLE_CLOSE_SECONDS", "60")
        try:
            return max(1, int(float(raw_value))) * 1000
        except ValueError:
            return 60_000

    def _poll_llm_jobs(self) -> None:
        if self._example_generator is None:
            return
        try:
            self._poll_explain_job()
            self._poll_batch_jobs()
            self._stop_llm_polling_if_idle()
        except Exception as error:
            self._fail_llm_polling(error)

    def _fail_llm_polling(self, error: Exception) -> None:
        if self._llm_poll_timer is not None and self._llm_poll_timer.isActive():
            self._llm_poll_timer.stop()
        try:
            self._llm_jobs.finish_explain()
            self._llm_jobs.clear_batch_jobs()
        except Exception:
            pass
        if self._explain_current_study_button is not None:
            self._explain_current_study_button.setText("AI 速解")
            self._explain_current_study_button.setEnabled(True)
        if getattr(self, "_batch_state", "idle") in {"running", "paused", "stopped"}:
            self._set_batch_idle()
        self._show_error_message("AI 任务失败", str(error) or error.__class__.__name__)

    def _poll_explain_job(self) -> None:
        event = self._llm_jobs.poll_explain()
        if event is None:
            return
        if isinstance(event, ExplainProgress):
            if self._explain_current_study_button is not None:
                self._explain_current_study_button.setText(event.message[:8] or "解释中")
            return
        if isinstance(event, ExplainFailed):
            self._finish_explain_job(schedule_idle_close=False)
            self._show_error_message("AI 速解失败", event.message)
            return
        if isinstance(event, ExplainCompleted):
            self._finish_explain_job(schedule_idle_close=False)
            self._show_explanation_dialog(event.entry.word if event.entry is not None else "当前词", event.explanation)

    def _poll_batch_jobs(self) -> None:
        for event in self._llm_jobs.poll_batch():
            self._finish_batch_job(event.entries, result=event.result, error=event.error)

    def _finish_batch_job(
        self,
        entries: list[WordEntry],
        result: dict | None = None,
        error: str = "",
    ) -> None:
        if error:
            label = "、".join(entry.word for entry in entries[:3])
            if len(entries) > 3:
                label = f"{label} 等 {len(entries)} 条"
            self._batch_errors.append(f"{label}: {error}")
        elif result is not None:
            try:
                self._apply_batch_result(entries, result)
                self._refresh_words(False)
                self._restore_table_selection(self._batch_updated_ids)
            except Exception as apply_error:
                label = "、".join(entry.word for entry in entries[:3])
                if len(entries) > 3:
                    label = f"{label} 等 {len(entries)} 条"
                self._batch_errors.append(f"{label}: {apply_error}")
        self._batch_completed_count += len(entries)
        self._on_batch_progress(
            self._batch_action,
            self._batch_completed_count,
            len(self._batch_entries),
            self._llm_jobs.batch_job_count,
            time.monotonic() - self._batch_started_at,
        )
        QTimer.singleShot(0, self._pump_batch_processes)

    def _apply_batch_result(self, entries: list[WordEntry], data: dict) -> None:
        if self._batch_action == "补充":
            if self._batch_mode == "batch":
                self._apply_supplement_batch_result(entries, data)
                return
            entry = entries[0]
            updated = self._update_supplemented_entry(
                entry.id,
                str(data["example_sentence"]),
                str(data["example_sentence_cn"]),
                str(data.get("meaning", "")),
            )
        else:
            entry = entries[0]
            corrected_word = str(data["corrected_word"]).strip()
            should_update = bool(data.get("should_update", True))
            if not should_update or not corrected_word or corrected_word == entry.word.strip():
                return
            updated = self._service.update_text(
                entry.id,
                corrected_word,
                entry.meaning,
                entry.forms,
            )
        if updated is not None:
            self._batch_updated_ids.append(updated.id)

    def _apply_supplement_batch_result(self, entries: list[WordEntry], data: dict) -> None:
        items = data.get("items", [])
        if not isinstance(items, list):
            raise RuntimeError(f"AI 批量结果格式不正确：{data}")
        if len(items) != len(entries):
            raise RuntimeError(f"AI 批量结果数量不匹配：期望 {len(entries)}，得到 {len(items)}")
        entries_by_identity = {
            self._batch_result_entry_identity(entry.source_index, entry.word): entry
            for entry in entries
        }
        used_entry_ids: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                raise RuntimeError(f"AI 批量结果包含无效条目：{item}")
            entry = self._entry_for_supplement_item(item, entries_by_identity)
            if entry.id in used_entry_ids:
                raise RuntimeError(f"AI 批量结果重复指向词条：{entry.word}")
            used_entry_ids.add(entry.id)
            updated = self._update_supplemented_entry(
                entry.id,
                str(item["example_sentence"]),
                str(item["example_sentence_cn"]),
                str(item.get("meaning", "")),
            )
            if updated is not None:
                self._batch_updated_ids.append(updated.id)

    @classmethod
    def _entry_for_supplement_item(
        cls,
        item: dict,
        entries_by_identity: dict[tuple[int, str], WordEntry],
    ) -> WordEntry:
        raw_entry = item.get("entry")
        if not isinstance(raw_entry, dict):
            raise RuntimeError(f"AI 批量结果缺少词条身份：{item}")
        try:
            source_index = int(raw_entry.get("source_index", 0))
        except (TypeError, ValueError):
            source_index = 0
        word = str(raw_entry.get("word", "")).strip()
        identity = cls._batch_result_entry_identity(source_index, word)
        matched = entries_by_identity.get(identity)
        if matched is None:
            raise RuntimeError(f"AI 批量结果词条不匹配：{source_index}, {word}")
        return matched

    @staticmethod
    def _batch_result_entry_identity(source_index: int, word: str) -> tuple[int, str]:
        return source_index, re.sub(r"\s+", " ", word.strip()).casefold()

    def _update_supplemented_entry(
        self,
        entry_id: str,
        example_sentence: str,
        example_sentence_cn: str,
        meaning: str = "",
    ) -> WordEntry | None:
        entry = self._service.get_word(entry_id)
        if entry is None:
            return None
        if (not entry.meaning.strip() or not self._has_cjk_text(entry.meaning)) and self._has_cjk_text(meaning):
            updated = self._service.update_text(
                entry.id,
                entry.word,
                meaning.strip(),
                entry.forms,
            )
            if updated is None:
                return None
        return self._service.update_examples(
            entry.id,
            example_sentence,
            example_sentence_cn,
        )

    @staticmethod
    def _has_cjk_text(text: str) -> bool:
        return bool(re.search(r"[\u3400-\u9fff]", text or ""))

    def _finish_isolated_batch(self) -> None:
        if self._batch_finished:
            return
        self._batch_finished = True
        action = self._batch_action
        total = len(self._batch_entries)
        updated_ids = list(self._batch_updated_ids)
        errors = list(self._batch_errors)
        self._llm_jobs.clear_batch_jobs()
        self._stop_llm_polling_if_idle()
        self._set_batch_idle()
        self._finish_batch_progress(action, len(updated_ids), total, len(errors))
        self._selected_ids = updated_ids
        self._selected_id = updated_ids[0] if updated_ids else None
        self._refresh_words(False)
        self._restore_table_selection(updated_ids)
        action_label = self._batch_action_label(action)
        message = f"已{action_label} {len(updated_ids)} / {total} 条。"
        if errors:
            message = f"{message}\n失败 {len(errors)} 条：\n" + "\n".join(errors[:5])
        QTimer.singleShot(0, lambda: self._show_batch_message("information", f"AI {action_label}完成", message))

    def _batch_parallel_limit(self) -> int:
        return self._batch_parallel_limit_value

    @staticmethod
    def _batch_action_label(action: str) -> str:
        if action == "补充":
            return "补全"
        if action == "释义":
            return "补充释义"
        return "校对"

    @staticmethod
    def _initial_batch_parallel_limit() -> int:
        return initial_batch_parallel_limit()

    def _recommended_batch_parallel_limit(self) -> int:
        fallback = 1
        if self._example_generator is None or not hasattr(self._example_generator, "recommended_process_parallelism"):
            return fallback
        try:
            int(self._example_generator.recommended_process_parallelism())
            return 1
        except Exception:
            return fallback

    def _recommended_batch_strategy(self, action: str) -> dict[str, int | str]:
        fallback = {
            "mode": "parallel",
            "parallelism": self._recommended_batch_parallel_limit(),
            "batch_size": 1,
        }
        if action != "补充":
            return fallback
        if self._example_generator is None or not hasattr(self._example_generator, "recommended_supplement_strategy"):
            return fallback
        try:
            raw = self._example_generator.recommended_supplement_strategy()
            mode = str(raw.get("mode", "parallel"))
            if mode not in {"parallel", "batch"}:
                return fallback
            return {
                "mode": mode,
                "parallelism": 1,
                "batch_size": max(1, min(32, int(raw.get("batch_size", 1)))),
            }
        except Exception:
            return fallback

    def _on_batch_progress(self, action: str, done: int, total: int, workers: int, elapsed: float) -> None:
        action_label = self._batch_action_label(action)
        if action == "补充":
            self._supplement_button.setText(f"{action_label}中 {done}/{total} · {workers} 个任务")
        else:
            self._correct_button.setText(f"{action_label}中 {done}/{total} · {workers} 个任务")
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
                    updated = self._update_supplemented_entry(
                        entry_id,
                        first_value,
                        second_value,
                        third_value,
                    )
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
        action_label = self._batch_action_label(action)
        message = f"已{action_label} {len(updated_ids)} / {total} 条。"
        if errors:
            message = f"{message}\n失败 {len(errors)} 条：\n" + "\n".join(errors[:5])
        title = f"AI {action_label}完成" if action == "补充" else f"{action_label}完成"
        self._pending_batch_message = ("information", title, message)

    def _on_batch_failed(self, action: str, message: str) -> None:
        action_label = self._batch_action_label(action)
        self._set_batch_idle()
        self._clear_batch_progress(f"{action_label}失败")
        title = f"AI {action_label}失败" if action == "补充" else f"{action_label}失败"
        self._pending_batch_message = ("critical", title, message)

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
            self._show_error_message(title, message)
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
            self._supplement_button.setText("AI 补全例句")
            self._supplement_button.setEnabled(True)
        if self._correct_button is not None:
            self._correct_button.setText("补充释义")
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
            if self._batch_entries and self._batch_thread is None:
                QTimer.singleShot(0, self._pump_batch_processes)

    def _stop_batch(self) -> None:
        if self._batch_state in {"running", "paused"}:
            self._batch_state = "stopped"
            self._pause_button.setText("暂停")
            self._pause_button.setEnabled(False)
            self._stop_button.setEnabled(False)
            if self._batch_status_label is not None:
                self._batch_status_label.setText(f"{self._batch_status_label.text()} | 正在停止")
            if self._batch_thread is not None:
                return
            if self._llm_jobs.has_batch_jobs():
                self._llm_jobs.clear_batch_jobs()
                QTimer.singleShot(0, self._finish_isolated_batch)
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
            eta_text = "剩余时间：估算中"
        else:
            average_seconds = elapsed_seconds / done
            remaining_seconds = max(0.0, average_seconds * (total - done))
            eta_text = f"剩余时间：{self._format_duration(remaining_seconds)}"
        worker_text = f"{workers} 个任务" if workers else "准备中"
        action_label = self._batch_action_label(action)
        self._batch_status_label.setText(
            f"{action_label}进度：{done} / {total} ({percent}%) | {worker_text} | {eta_text}"
        )

    def _finish_batch_progress(self, action: str, success_count: int, total: int, error_count: int) -> None:
        if self._progress is not None:
            self._progress.setValue(100 if total else 0)
        if self._batch_status_label is not None:
            action_label = self._batch_action_label(action)
            self._batch_status_label.setText(f"{action_label}完成：成功 {success_count} / {total}，失败 {error_count}。")

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

