import csv
from dataclasses import dataclass
from pathlib import Path

from wordpycket.domain.entities import WordEntry


@dataclass(frozen=True)
class CsvColumnSchema:
    language: str
    columns: tuple[str, str, str, str, str]

    @property
    def source_index(self) -> str:
        return self.columns[0]

    @property
    def word(self) -> str:
        return self.columns[1]

    @property
    def meaning(self) -> str:
        return self.columns[2]

    @property
    def frequency(self) -> str:
        return self.columns[3]

    @property
    def forms(self) -> str:
        return self.columns[4]


@dataclass(frozen=True)
class CsvImportResult:
    entries: list[WordEntry]
    language: str


class WordFrequencyCsvImporter:
    SCHEMAS = (
        CsvColumnSchema("英语", ("Index", "English", "Chinese", "Frequency", "Forms")),
        CsvColumnSchema("德语", ("Index", "Deutsch", "Chinesisch", "Häufigkeit", "Formen")),
        CsvColumnSchema("法语", ("Index", "Français", "Chinois", "Fréquence", "Formes")),
        CsvColumnSchema("西班牙语", ("Index", "Español", "Chino", "Frecuencia", "Formas")),
        CsvColumnSchema("意大利语", ("Index", "Italiano", "Cinese", "Frequenza", "Forme")),
        CsvColumnSchema("葡萄牙语", ("Index", "Português", "Chinês", "Frequência", "Formas")),
        CsvColumnSchema("荷兰语", ("Index", "Nederlands", "Chinees", "Frequentie", "Vormen")),
        CsvColumnSchema("日语", ("Index", "日本語", "中国語", "頻度", "語形")),
        CsvColumnSchema("韩语", ("Index", "한국어", "중국어", "빈도", "형태")),
    )

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path

    def load(self) -> list[WordEntry]:
        return self.load_with_metadata().entries

    def load_with_metadata(self) -> CsvImportResult:
        if not self._csv_path.exists():
            return CsvImportResult([], "")

        with self._csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            schema = self._detect_schema(reader.fieldnames)
            return CsvImportResult(
                [self._from_row(row, schema) for row in reader],
                schema.language,
            )

    @classmethod
    def schema_for_language(cls, language: str) -> CsvColumnSchema:
        for schema in cls.SCHEMAS:
            if schema.language == language:
                return schema
        raise ValueError(f"不支持的 CSV 语言：{language}")

    @classmethod
    def _detect_schema(cls, fieldnames: list[str] | None) -> CsvColumnSchema:
        headers = tuple((name or "").strip() for name in fieldnames or [])
        for schema in cls.SCHEMAS:
            if headers == schema.columns:
                return schema
        expected = "；".join(", ".join(schema.columns) for schema in cls.SCHEMAS)
        actual = ", ".join(headers) if headers else "空表头"
        raise ValueError(
            "CSV 列名不符合支持的固定格式。"
            f"当前列名：{actual}。"
            f"支持格式：{expected}。"
        )

    @staticmethod
    def _from_row(row: dict[str, str], schema: CsvColumnSchema) -> WordEntry:
        return WordEntry(
            source_index=int(row.get(schema.source_index, "0") or 0),
            word=row.get(schema.word, ""),
            meaning=row.get(schema.meaning, ""),
            frequency=int(row.get(schema.frequency, "0") or 0),
            forms=row.get(schema.forms, ""),
            example_sentence=row.get("Example", ""),
            example_sentence_cn=row.get("ExampleChinese", ""),
        )
