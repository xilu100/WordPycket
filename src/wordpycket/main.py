from pathlib import Path

from wordpycket.application.services import WordService
from wordpycket.infrastructure.csv_importer import WordFrequencyCsvImporter
from wordpycket.infrastructure.example_generator import LocalLlmExampleGenerator
from wordpycket.infrastructure.repositories import SqliteWordRepository
from wordpycket.presentation.qt_app import WordPycketApp


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    csv_file = project_root / "input" / "word_frequency.csv"
    model_dir = project_root / "model"
    database_file = Path.home() / ".wordpycket" / "wordpycket.db"

    repository = SqliteWordRepository(database_file)
    service = WordService(repository)
    importer = WordFrequencyCsvImporter(csv_file)
    service.import_words(importer.load())

    app = WordPycketApp(
        service,
        reset_entries_loader=importer.load,
        example_generator=LocalLlmExampleGenerator(model_dir),
    )
    app.run()


if __name__ == "__main__":
    main()
