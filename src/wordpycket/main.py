import os
import sys
from pathlib import Path

from wordpycket.application.datasets import DatasetService
from wordpycket.application.services import WordService
from wordpycket.infrastructure.csv_importer import WordFrequencyCsvImporter
from wordpycket.infrastructure.csv_library import CsvLibrary
from wordpycket.infrastructure.example_generator import LocalLlmExampleGenerator
from wordpycket.infrastructure.meaning_translator import ArgosMeaningTranslator
from wordpycket.infrastructure.pdf_vocabulary_importer import run_pdf_import_isolated
from wordpycket.infrastructure.repositories import SqliteWordRepository
from wordpycket.infrastructure.settings_store import JsonSettingsStore
from wordpycket.presentation.qt_app import WordPycketApp


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
    if active_csv is None:
        library.cleanup_orphan_databases()
    initial_database = library.database_path(active_csv) if active_csv is not None else database_dir / "empty.db"
    repository = SqliteWordRepository(initial_database)
    service = WordService(repository)
    example_generator = LocalLlmExampleGenerator(model_dir)
    meaning_translator = ArgosMeaningTranslator()
    settings = JsonSettingsStore(data_dir / "settings.json")
    dataset_service = DatasetService(
        service,
        library,
        repository_factory=SqliteWordRepository,
        csv_loader=lambda path: WordFrequencyCsvImporter(path).load_with_metadata(),
        pdf_builder=run_pdf_import_isolated,
        empty_database_path=database_dir / "__empty__.db",
        vocabulary_cleaner=example_generator,
    )

    if active_csv is not None:
        dataset_service.activate_csv(active_csv)

    def current_language() -> str:
        csv_path = library.active_csv()
        if csv_path is None:
            return ""
        return WordFrequencyCsvImporter(csv_path).load_with_metadata().language

    app = WordPycketApp(
        service,
        example_generator=example_generator,
        meaning_translator=meaning_translator,
        csv_import_loader=lambda path: WordFrequencyCsvImporter(path).load_with_metadata(),
        pdf_import_loader=dataset_service.import_pdf,
        csv_files_loader=library.list_csv_files,
        active_csv_loader=library.active_csv,
        csv_switcher=dataset_service.activate_csv,
        csv_upload_handler=dataset_service.upload_csv,
        csv_delete_handler=dataset_service.delete_csv,
        ai_scope_loader=lambda: settings.get_string("ai_scope", "AI 领域的译法"),
        ai_scope_saver=lambda value: settings.set_string("ai_scope", value),
        current_language_loader=current_language,
    )
    app.run()


if __name__ == "__main__":
    main()
