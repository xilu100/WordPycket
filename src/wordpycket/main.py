import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from wordpycket.application.services import WordService
from wordpycket.infrastructure.csv_importer import WordFrequencyCsvImporter
from wordpycket.infrastructure.csv_library import CsvLibrary
from wordpycket.infrastructure.example_generator import LocalLlmExampleGenerator
from wordpycket.infrastructure.pdf_vocabulary_importer import PdfVocabularyImporter
from wordpycket.infrastructure.repositories import SqliteWordRepository
from wordpycket.presentation.qt_app import WordPycketApp


@dataclass(frozen=True)
class DatasetResult:
    language: str
    csv_path: Path
    imported_count: int


def configure_project_runtime(runtime_dir: Path) -> None:
    nltk_data = runtime_dir / "nltk_data"
    cache_dir = runtime_dir / "cache"
    nltk_data.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NLTK_DATA", str(nltk_data))
    os.environ.setdefault("PIP_CACHE_DIR", str(cache_dir / "pip"))
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def main() -> None:
    if getattr(sys, "frozen", False):
        project_root = Path(sys.executable).resolve().parent
    else:
        project_root = Path(__file__).resolve().parents[2]
    input_dir = project_root / "input"
    model_dir = project_root / "model"
    runtime_dir = project_root / "runtime"
    data_dir = project_root / "data"
    database_dir = data_dir / "csv_databases"
    for directory in (input_dir, model_dir, database_dir):
        directory.mkdir(parents=True, exist_ok=True)
    configure_project_runtime(runtime_dir)

    library = CsvLibrary(input_dir, database_dir)
    active_csv = library.active_csv()
    initial_database = library.database_path(active_csv) if active_csv is not None else database_dir / "empty.db"
    repository = SqliteWordRepository(initial_database)
    service = WordService(repository)

    def activate_csv(csv_path: Path) -> DatasetResult:
        csv_path = csv_path.resolve()
        if not csv_path.exists():
            library.cleanup_orphan_databases()
            raise FileNotFoundError(f"CSV 不存在：{csv_path}")
        result = WordFrequencyCsvImporter(csv_path).load_with_metadata()
        service.use_repository(SqliteWordRepository(library.database_path(csv_path)))
        imported_count = service.import_words(result.entries)
        library.set_active_csv(csv_path)
        library.cleanup_orphan_databases()
        return DatasetResult(result.language, csv_path, imported_count)

    def upload_csv(source_path: Path) -> DatasetResult:
        WordFrequencyCsvImporter(source_path).load_with_metadata()
        target_path = library.path_for_uploaded_csv(source_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != target_path.resolve():
            shutil.copyfile(source_path, target_path)
        return activate_csv(target_path)

    def import_pdf(pdf_path: Path) -> DatasetResult:
        target_path = library.path_for_pdf_csv(pdf_path)
        PdfVocabularyImporter(pdf_path, target_path).build()
        return activate_csv(target_path)

    def delete_csv(csv_path: Path) -> DatasetResult | None:
        library.delete_csv(csv_path)
        library.cleanup_orphan_databases()
        next_csv = library.active_csv()
        if next_csv is None:
            service.use_repository(SqliteWordRepository(database_dir / "__empty__.db"))
            return None
        return activate_csv(next_csv)

    if active_csv is not None:
        activate_csv(active_csv)
    else:
        library.cleanup_orphan_databases()

    app = WordPycketApp(
        service,
        example_generator=LocalLlmExampleGenerator(model_dir),
        csv_import_loader=lambda path: WordFrequencyCsvImporter(path).load_with_metadata(),
        pdf_import_loader=import_pdf,
        csv_files_loader=library.list_csv_files,
        active_csv_loader=library.active_csv,
        csv_switcher=activate_csv,
        csv_upload_handler=upload_csv,
        csv_delete_handler=delete_csv,
    )
    app.run()


if __name__ == "__main__":
    main()
