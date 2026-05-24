from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, Signal, Slot

from wordpycket.application.ai_batch import AiBatchRunner
from wordpycket.application.ports import ExampleGenerator, PdfImportResult
from wordpycket.domain.entities import WordEntry

if TYPE_CHECKING:
    from wordpycket.presentation.qt_app import WordPycketApp


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
        self._runner = AiBatchRunner(
            action,
            entries,
            scope,
            generator,
            control,
            sleep=lambda seconds: QThread.msleep(int(seconds * 1000)),
        )

    def run(self) -> None:
        started_at = time.monotonic()

        def progress(done: int, count: int, workers: int) -> None:
            self.progress_changed.emit(self._action, done, count, workers, time.monotonic() - started_at)

        try:
            result = self._runner.run(progress)
        except Exception as error:
            self.failed.emit(self._action, str(error))
            return

        updates = [
            (update.entry_id, update.first_value, update.second_value, update.third_value)
            for update in result.updates
        ]
        self.finished.emit(self._action, updates, result.errors, result.total)

    def _generate(self, progress):
        return self._runner.generate(progress)

    def _correct(self, progress):
        return self._runner.correct(progress)


class PdfImportWorker(QObject):
    progress_changed = Signal(str, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        pdf_path: Path,
        use_llm_cleanup: bool,
        pdf_import_loader: Callable[[Path, bool, Callable[[str, int], None] | None], PdfImportResult],
    ) -> None:
        super().__init__()
        self._pdf_path = pdf_path
        self._use_llm_cleanup = use_llm_cleanup
        self._pdf_import_loader = pdf_import_loader

    @Slot()
    def run(self) -> None:
        try:
            result = self._pdf_import_loader(
                self._pdf_path,
                self._use_llm_cleanup,
                lambda message, percent: self.progress_changed.emit(message, percent),
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.progress_changed.emit("PDF 导入完成，准备刷新界面", 100)
        self.finished.emit(result)


class BackgroundTaskWorker(QObject):
    finished = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, name: str, task: Callable[[], object]) -> None:
        super().__init__()
        self._name = name
        self._task = task

    @Slot()
    def run(self) -> None:
        try:
            result = self._task()
        except Exception as error:
            self.failed.emit(self._name, str(error))
            return
        self.finished.emit(self._name, result)


class UiThreadBridge(QObject):
    def __init__(self, app: WordPycketApp) -> None:
        super().__init__()
        self._app = app

    @Slot(str, int)
    def set_pdf_progress(self, message: str, percent: int) -> None:
        self._app._set_pdf_progress(message, percent)

    @Slot(object)
    def on_pdf_import_finished(self, result: object) -> None:
        self._app._on_pdf_import_finished(result)

    @Slot(str)
    def on_pdf_import_failed(self, message: str) -> None:
        self._app._on_pdf_import_failed(message)

    @Slot()
    def on_pdf_import_thread_finished(self) -> None:
        self._app._on_pdf_import_thread_finished()

    @Slot()
    def run_pdf_import_worker(self) -> None:
        if self._app._pdf_import_worker is not None:
            self._app._pdf_import_worker.run()

    @Slot(str, object)
    def on_model_check_finished(self, name: str, result: object) -> None:
        self._app._on_model_check_finished(name, result)

    @Slot(str, str)
    def on_model_check_failed(self, name: str, message: str) -> None:
        self._app._on_model_check_failed(name, message)

    @Slot()
    def on_model_check_thread_finished(self) -> None:
        self._app._on_model_check_thread_finished()

    @Slot(str, object)
    def on_csv_task_finished(self, name: str, result: object) -> None:
        self._app._on_csv_task_finished(name, result)

    @Slot(str, str)
    def on_csv_task_failed(self, name: str, message: str) -> None:
        self._app._on_csv_task_failed(name, message)

    @Slot()
    def on_csv_task_thread_finished(self) -> None:
        self._app._on_csv_task_thread_finished()
