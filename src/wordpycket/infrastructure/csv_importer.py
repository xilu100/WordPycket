import csv
from pathlib import Path

from wordpycket.domain.entities import WordEntry


class WordFrequencyCsvImporter:
    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path

    def load(self) -> list[WordEntry]:
        if not self._csv_path.exists():
            return []

        with self._csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            return [self._from_row(row) for row in reader]

    @staticmethod
    def _from_row(row: dict[str, str]) -> WordEntry:
        return WordEntry(
            source_index=int(row.get("Index", "0") or 0),
            word=row.get("English", ""),
            meaning=row.get("Chinese", ""),
            frequency=int(row.get("Frequency", "0") or 0),
            forms=row.get("Forms", ""),
            example_sentence=row.get("Example", ""),
            example_sentence_cn=row.get("ExampleChinese", ""),
        )
