from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from wordpycket.application.services import WordService
from wordpycket.domain.entities import WordEntry
from wordpycket.domain.repositories import WordRepository


ProgressCallback = Callable[[str, int], None]


@dataclass(frozen=True)
class DatasetResult:
    language: str
    csv_path: Path
    imported_count: int


class CsvImportResult(Protocol):
    entries: list[WordEntry]
    language: str


class CsvLibraryPort(Protocol):
    def active_csv(self) -> Path | None: ...

    def cleanup_orphan_databases(self) -> None: ...

    def database_path(self, csv_path: Path) -> Path: ...

    def set_active_csv(self, csv_path: Path) -> None: ...

    def path_for_uploaded_csv(self, source_path: Path) -> Path: ...

    def path_for_pdf_csv(self, pdf_path: Path) -> Path: ...

    def delete_csv(self, csv_path: Path) -> None: ...


class VocabularyCleaner(Protocol):
    def clean_pdf_vocabulary_entries(
        self,
        entries: list[WordEntry],
        language: str,
        progress_callback: ProgressCallback | None = None,
    ) -> list[WordEntry]: ...


PdfVocabularyBuilder = Callable[[Path, Path, VocabularyCleaner | None, ProgressCallback | None], object]


class DatasetService:
    def __init__(
        self,
        word_service: WordService,
        csv_library: CsvLibraryPort,
        repository_factory: Callable[[Path], WordRepository],
        csv_loader: Callable[[Path], CsvImportResult],
        pdf_builder: PdfVocabularyBuilder,
        empty_database_path: Path,
        vocabulary_cleaner: VocabularyCleaner | None = None,
    ) -> None:
        self._word_service = word_service
        self._csv_library = csv_library
        self._repository_factory = repository_factory
        self._csv_loader = csv_loader
        self._pdf_builder = pdf_builder
        self._empty_database_path = empty_database_path
        self._vocabulary_cleaner = vocabulary_cleaner

    def activate_csv(self, csv_path: Path) -> DatasetResult:
        return self._activate_csv(csv_path)

    def _activate_csv(
        self,
        csv_path: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> DatasetResult:
        csv_path = csv_path.resolve()
        if not csv_path.exists():
            self._csv_library.cleanup_orphan_databases()
            raise FileNotFoundError(f"CSV 不存在：{csv_path}")
        if progress_callback is not None:
            progress_callback("读取生成的 CSV", 93)
        result = self._csv_loader(csv_path)
        if progress_callback is not None:
            progress_callback("准备词库数据库", 94)
        self._word_service.use_repository(self._repository_factory(self._csv_library.database_path(csv_path)))
        if progress_callback is not None:
            progress_callback(f"写入词库数据库：0/{len(result.entries)}", 95)
        imported_count = self._word_service.import_words(result.entries)
        if progress_callback is not None:
            progress_callback(f"写入词库数据库：{imported_count}/{len(result.entries)}", 98)
        self._csv_library.set_active_csv(csv_path)
        self._csv_library.cleanup_orphan_databases()
        if progress_callback is not None:
            progress_callback("词库切换完成", 100)
        return DatasetResult(result.language, csv_path, imported_count)

    def upload_csv(self, source_path: Path) -> DatasetResult:
        self._csv_loader(source_path)
        target_path = self._csv_library.path_for_uploaded_csv(source_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != target_path.resolve():
            shutil.copyfile(source_path, target_path)
        return self.activate_csv(target_path)

    def import_pdf(
        self,
        pdf_path: Path,
        use_llm_cleanup: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> DatasetResult:
        target_path = self._csv_library.path_for_pdf_csv(pdf_path)
        vocabulary_cleaner = self._vocabulary_cleaner if use_llm_cleanup else None
        if progress_callback is not None:
            progress_callback("准备解析 PDF", 1)
        build_result = self._pdf_builder(pdf_path, target_path, vocabulary_cleaner, progress_callback)
        if progress_callback is not None:
            progress_callback("导入 CSV 到数据库", 92)
        result = self._activate_csv(target_path, progress_callback)
        detected_language = getattr(build_result, "language", "")
        if detected_language:
            return DatasetResult(detected_language, result.csv_path, result.imported_count)
        return result

    def delete_csv(self, csv_path: Path) -> DatasetResult | None:
        self._csv_library.delete_csv(csv_path)
        self._csv_library.cleanup_orphan_databases()
        next_csv = self._csv_library.active_csv()
        if next_csv is None:
            self._word_service.use_repository(self._repository_factory(self._empty_database_path))
            return None
        return self.activate_csv(next_csv)
