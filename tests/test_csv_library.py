from __future__ import annotations

from wordpycket.infrastructure.csv_library import CsvLibrary


def test_csv_library_uses_distinct_database_per_csv(tmp_path) -> None:
    input_dir = tmp_path / "input"
    database_dir = tmp_path / "db"
    input_dir.mkdir()
    first = input_dir / "english.csv"
    second = input_dir / "german.csv"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    library = CsvLibrary(input_dir, database_dir)

    first_database = library.database_path(first)
    second_database = library.database_path(second)

    assert first_database != second_database
    assert first_database.parent == database_dir
    assert second_database.parent == database_dir


def test_csv_library_removes_database_when_csv_is_deleted(tmp_path) -> None:
    input_dir = tmp_path / "input"
    database_dir = tmp_path / "db"
    input_dir.mkdir()
    csv_path = input_dir / "english.csv"
    csv_path.write_text("", encoding="utf-8")
    library = CsvLibrary(input_dir, database_dir)
    database_path = library.database_path(csv_path)
    database_path.write_text("db", encoding="utf-8")
    database_path.with_name(f"{database_path.name}-wal").write_text("wal", encoding="utf-8")
    database_path.with_name(f"{database_path.name}-shm").write_text("shm", encoding="utf-8")

    csv_path.unlink()
    library.cleanup_orphan_databases()

    assert not database_path.exists()
    assert not database_path.with_name(f"{database_path.name}-wal").exists()
    assert not database_path.with_name(f"{database_path.name}-shm").exists()


def test_csv_library_delete_csv_removes_csv_database_and_active_marker(tmp_path) -> None:
    input_dir = tmp_path / "input"
    database_dir = tmp_path / "db"
    input_dir.mkdir()
    csv_path = input_dir / "english.csv"
    csv_path.write_text("", encoding="utf-8")
    library = CsvLibrary(input_dir, database_dir)
    database_path = library.database_path(csv_path)
    database_path.write_text("db", encoding="utf-8")
    library.set_active_csv(csv_path)

    library.delete_csv(csv_path)

    assert not csv_path.exists()
    assert not database_path.exists()
    assert library.active_csv() is None
