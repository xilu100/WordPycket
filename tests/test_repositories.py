from __future__ import annotations

from datetime import datetime, timedelta

from wordpycket.domain.entities import WordEntry
from wordpycket.infrastructure.repositories import SqliteWordRepository


def test_sqlite_reset_progress_preserves_csv_fields_and_examples(tmp_path) -> None:
    repository = SqliteWordRepository(tmp_path / "words.db")
    reviewed_at = datetime(2026, 5, 23, 10, 0, 0)
    entry = WordEntry(
        word="vector",
        meaning="向量",
        source_index=3,
        frequency=42,
        forms="vectors",
        example_sentence="Vectors represent direction.",
        example_sentence_cn="向量表示方向。",
        mastery_level=2,
        review_count=4,
        correct_count=5,
        wrong_count=1,
        last_reviewed_at=reviewed_at,
        learned_available_at=reviewed_at + timedelta(hours=24),
    )
    repository.save(entry)

    repository.reset_progress()

    reset_entry = repository.get(entry.id)
    assert reset_entry is not None
    assert reset_entry.id == entry.id
    assert reset_entry.created_at == entry.created_at
    assert reset_entry.word == "vector"
    assert reset_entry.meaning == "向量"
    assert reset_entry.source_index == 3
    assert reset_entry.frequency == 42
    assert reset_entry.forms == "vectors"
    assert reset_entry.example_sentence == "Vectors represent direction."
    assert reset_entry.example_sentence_cn == "向量表示方向。"
    assert reset_entry.mastery_level == 0
    assert reset_entry.review_count == 0
    assert reset_entry.correct_count == 0
    assert reset_entry.wrong_count == 0
    assert reset_entry.last_reviewed_at is None
    assert reset_entry.learned_available_at is None
    assert reset_entry.status == "学习池"


def test_sqlite_import_replaces_existing_meaning_with_csv_meaning(tmp_path) -> None:
    repository = SqliteWordRepository(tmp_path / "words.db")
    original = WordEntry(word="kernel", meaning="内核")
    repository.save(original)

    imported_count = repository.save_many(
        [
            WordEntry(
                word="kernel",
                meaning="",
                frequency=10,
                forms="kernels",
            )
        ]
    )

    imported = repository.get(original.id)
    assert imported_count == 1
    assert imported is not None
    assert imported.meaning == ""
    assert imported.frequency == 10
    assert imported.forms == "kernels"
