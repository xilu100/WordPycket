from pathlib import Path

from wordpycket.application.services import WordService
from wordpycket.infrastructure.csv_importer import WordFrequencyCsvImporter
from wordpycket.infrastructure.repositories import SqliteWordRepository
from wordpycket.presentation.tk_app import WordPycketApp


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    csv_file = project_root / "input" / "word_frequency.csv"
    database_file = Path.home() / ".wordpycket" / "wordpycket.db"

    repository = SqliteWordRepository(database_file)
    service = WordService(repository)
    service.import_words(WordFrequencyCsvImporter(csv_file).load())

    app = WordPycketApp(service)
    app.run()


if __name__ == "__main__":
    main()
