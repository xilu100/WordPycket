from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from wordpycket.application.datasets import DatasetService
from wordpycket.application.services import WordService
from wordpycket.domain.entities import WordEntry


class FakeRepository:
    def __init__(self) -> None:
        self.entries: list[WordEntry] = []

    def list(self) -> list[WordEntry]:
        return self.entries

    def get(self, entry_id: str) -> WordEntry | None:
        return next((entry for entry in self.entries if entry.id == entry_id), None)

    def save(self, entry: WordEntry) -> None:
        self.entries.append(entry)

    def save_many(self, entries: list[WordEntry]) -> int:
        self.entries.extend(entries)
        return len(entries)

    def delete(self, entry_id: str) -> None:
        self.entries = [entry for entry in self.entries if entry.id != entry_id]

    def reset_progress(self) -> None:
        return None

    def replace_all(self, entries: list[WordEntry]) -> int:
        self.entries = list(entries)
        return len(entries)

    def update_examples(
        self,
        entry_id: str,
        example_sentence: str,
        example_sentence_cn: str,
    ) -> None:
        return None

    def update_text(
        self,
        entry_id: str,
        word: str,
        meaning: str,
        forms: str,
    ) -> None:
        return None


class FakeCsvLibrary:
    def __init__(self, input_dir: Path, database_dir: Path) -> None:
        self.input_dir = input_dir
        self.database_dir = database_dir
        self.active: Path | None = None

    def active_csv(self) -> Path | None:
        return self.active

    def cleanup_orphan_databases(self) -> None:
        return None

    def database_path(self, csv_path: Path) -> Path:
        return self.database_dir / f"{csv_path.stem}.db"

    def set_active_csv(self, csv_path: Path) -> None:
        self.active = csv_path

    def path_for_uploaded_csv(self, source_path: Path) -> Path:
        return self.input_dir / source_path.name

    def path_for_pdf_csv(self, pdf_path: Path) -> Path:
        return self.input_dir / f"{pdf_path.stem}.csv"

    def delete_csv(self, csv_path: Path) -> None:
        csv_path.unlink(missing_ok=True)


def test_pdf_import_reports_database_activation_after_92_percent(tmp_path) -> None:
    csv_path = tmp_path / "input" / "source.csv"
    csv_path.parent.mkdir()
    csv_path.write_text("placeholder", encoding="utf-8")
    entries = [WordEntry(word="model", meaning="", frequency=2)]
    progress: list[tuple[str, int]] = []

    def build_pdf(_pdf_path, target_path, _cleaner, progress_callback) -> None:
        assert target_path == csv_path
        if progress_callback is not None:
            progress_callback("PDF 词表生成完成", 90)

    service = DatasetService(
        WordService(FakeRepository()),
        FakeCsvLibrary(csv_path.parent, tmp_path / "data"),
        repository_factory=lambda _path: FakeRepository(),
        csv_loader=lambda _path: SimpleNamespace(entries=entries, language="英语"),
        pdf_builder=build_pdf,
        empty_database_path=tmp_path / "empty.db",
    )

    result = service.import_pdf(
        tmp_path / "source.pdf",
        progress_callback=lambda message, percent: progress.append((message, percent)),
    )

    assert result.imported_count == 1
    assert progress[-4:] == [
        ("准备词库数据库", 94),
        ("写入词库数据库：0/1", 95),
        ("写入词库数据库：1/1", 98),
        ("词库切换完成", 100),
    ]
