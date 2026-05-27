from __future__ import annotations

import pytest

from wordpycket.infrastructure.csv_importer import WordFrequencyCsvImporter


def test_csv_importer_loads_english_schema(tmp_path) -> None:
    csv_path = tmp_path / "english.csv"
    csv_path.write_text(
        '"Index","English","Chinese","Frequency","Forms"\n'
        '1,"vector","向量",42,"vectors"\n',
        encoding="utf-8",
    )

    result = WordFrequencyCsvImporter(csv_path).load_with_metadata()

    assert result.language == "英语"
    assert len(result.entries) == 1
    assert result.entries[0].source_index == 1
    assert result.entries[0].word == "vector"
    assert result.entries[0].meaning == "向量"
    assert result.entries[0].frequency == 42
    assert result.entries[0].forms == "vectors"


def test_csv_importer_loads_german_schema(tmp_path) -> None:
    csv_path = tmp_path / "german.csv"
    csv_path.write_text(
        '"Index","Deutsch","Chinesisch","Häufigkeit","Formen"\n'
        '1,"Vektor","向量",42,"Vektoren"\n',
        encoding="utf-8",
    )

    result = WordFrequencyCsvImporter(csv_path).load_with_metadata()

    assert result.language == "德语"
    assert len(result.entries) == 1
    assert result.entries[0].source_index == 1
    assert result.entries[0].word == "Vektor"
    assert result.entries[0].meaning == "向量"
    assert result.entries[0].frequency == 42
    assert result.entries[0].forms == "Vektoren"


def test_csv_importer_rejects_changed_column_format(tmp_path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        '"English","Index","Chinese","Frequency","Forms"\n'
        '"vector",1,"向量",42,"vectors"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CSV 列名不符合"):
        WordFrequencyCsvImporter(csv_path).load()


def test_csv_importer_rejects_unsupported_language_schema(tmp_path) -> None:
    csv_path = tmp_path / "french.csv"
    csv_path.write_text(
        '"Index","Français","Chinois","Fréquence","Formes"\n'
        '1,"vecteur","向量",42,"vecteurs"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CSV 列名不符合"):
        WordFrequencyCsvImporter(csv_path).load()
