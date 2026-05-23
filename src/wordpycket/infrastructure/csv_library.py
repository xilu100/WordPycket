import hashlib
import re
from pathlib import Path


class CsvLibrary:
    def __init__(self, input_dir: Path, database_dir: Path) -> None:
        self._input_dir = input_dir
        self._database_dir = database_dir
        self._active_file = database_dir / "active_csv.txt"
        self._input_dir.mkdir(parents=True, exist_ok=True)
        self._database_dir.mkdir(parents=True, exist_ok=True)

    @property
    def input_dir(self) -> Path:
        return self._input_dir

    def list_csv_files(self) -> list[Path]:
        return sorted(
            (path for path in self._input_dir.glob("*.csv") if path.is_file()),
            key=lambda path: path.name.lower(),
        )

    def active_csv(self) -> Path | None:
        files = self.list_csv_files()
        if not files:
            return None
        saved_name = self._active_file.read_text(encoding="utf-8").strip() if self._active_file.exists() else ""
        for path in files:
            if path.name == saved_name:
                return path
        default_path = self._input_dir / "word_frequency.csv"
        if default_path in files:
            return default_path
        return files[0]

    def set_active_csv(self, csv_path: Path) -> None:
        self._active_file.write_text(csv_path.name, encoding="utf-8")

    def database_path(self, csv_path: Path) -> Path:
        return self._database_dir / f"{self._database_key(csv_path)}.db"

    def cleanup_orphan_databases(self) -> None:
        active_keys = {self._database_key(path) for path in self.list_csv_files()}
        for database_path in self._database_dir.glob("*.db"):
            if database_path.stem not in active_keys:
                self._delete_database_files(database_path)

    def delete_csv(self, csv_path: Path) -> None:
        database_path = self.database_path(csv_path)
        try:
            csv_path.unlink()
        except FileNotFoundError:
            pass
        self._delete_database_files(database_path)
        saved_name = self._active_file.read_text(encoding="utf-8").strip() if self._active_file.exists() else ""
        if saved_name == csv_path.name:
            try:
                self._active_file.unlink()
            except FileNotFoundError:
                pass

    def path_for_uploaded_csv(self, source_path: Path) -> Path:
        return self._input_dir / source_path.name

    def path_for_pdf_csv(self, pdf_path: Path) -> Path:
        base_name = f"{pdf_path.stem}.csv"
        return self._input_dir / base_name

    @staticmethod
    def _delete_database_files(database_path: Path) -> None:
        for path in (
            database_path,
            database_path.with_name(f"{database_path.name}-wal"),
            database_path.with_name(f"{database_path.name}-shm"),
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _database_key(csv_path: Path) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", csv_path.stem).strip("._") or "csv"
        digest = hashlib.sha1(csv_path.name.encode("utf-8")).hexdigest()[:10]
        return f"{slug}_{digest}"
