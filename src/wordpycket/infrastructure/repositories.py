from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from wordpycket.domain.entities import WordEntry
from wordpycket.domain.repositories import WordRepository


class SqliteWordRepository(WordRepository):
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def list(self) -> list[WordEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    word,
                    meaning,
                    source_index,
                    frequency,
                    forms,
                    example_sentence,
                    example_sentence_cn,
                    created_at,
                    mastery_level,
                    review_count,
                    correct_count,
                    wrong_count,
                    last_reviewed_at,
                    learned_available_at
                FROM word_entries
                ORDER BY source_index ASC, frequency DESC
                """
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def get(self, entry_id: str) -> WordEntry | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    word,
                    meaning,
                    source_index,
                    frequency,
                    forms,
                    example_sentence,
                    example_sentence_cn,
                    created_at,
                    mastery_level,
                    review_count,
                    correct_count,
                    wrong_count,
                    last_reviewed_at,
                    learned_available_at
                FROM word_entries
                WHERE id = ?
                """,
                (entry_id,),
            ).fetchone()
        return self._from_row(row) if row else None

    def save(self, entry: WordEntry) -> None:
        with self._connect() as connection:
            self._upsert(connection, entry, preserve_progress=False)

    def save_many(self, entries: list[WordEntry]) -> int:
        with self._connect() as connection:
            for entry in entries:
                self._upsert(connection, entry, preserve_progress=True)
        return len(entries)

    def delete(self, entry_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM word_entries WHERE id = ?", (entry_id,))

    def reset_progress(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE word_entries
                SET
                    mastery_level = 0,
                    review_count = 0,
                    correct_count = 0,
                    wrong_count = 0,
                    last_reviewed_at = NULL,
                    learned_available_at = NULL
                """
            )

    def replace_all(self, entries: list[WordEntry]) -> int:
        with self._connect() as connection:
            connection.execute("DELETE FROM word_entries")
            for entry in entries:
                self._upsert(connection, entry, preserve_progress=False)
        return len(entries)

    def update_examples(
        self,
        entry_id: str,
        example_sentence: str,
        example_sentence_cn: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE word_entries
                SET example_sentence = ?, example_sentence_cn = ?
                WHERE id = ?
                """,
                (example_sentence, example_sentence_cn, entry_id),
            )

    def update_text(
        self,
        entry_id: str,
        word: str,
        meaning: str,
        forms: str,
    ) -> None:
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    UPDATE word_entries
                    SET word = ?, meaning = ?, forms = ?
                    WHERE id = ?
                    """,
                    (word, meaning, forms, entry_id),
                )
            except sqlite3.IntegrityError as error:
                raise RuntimeError(f"词条 {word} 已存在，无法修正为重复单词。") from error

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS word_entries (
                    id TEXT PRIMARY KEY,
                    source_index INTEGER NOT NULL DEFAULT 0,
                    word TEXT NOT NULL,
                    meaning TEXT NOT NULL,
                    frequency INTEGER NOT NULL DEFAULT 0,
                    forms TEXT NOT NULL DEFAULT '',
                    example_sentence TEXT NOT NULL DEFAULT '',
                    example_sentence_cn TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    mastery_level INTEGER NOT NULL DEFAULT 0,
                    review_count INTEGER NOT NULL DEFAULT 0,
                    correct_count INTEGER NOT NULL DEFAULT 0,
                    wrong_count INTEGER NOT NULL DEFAULT 0,
                    last_reviewed_at TEXT,
                    learned_available_at TEXT
                )
                """
            )
            self._migrate_existing_schema(connection)
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_word_entries_word
                ON word_entries(word)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_word_entries_review
                ON word_entries(mastery_level, frequency, source_index)
                """
            )

    @staticmethod
    def _migrate_existing_schema(connection: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(word_entries)").fetchall()
        }
        migrations = {
            "source_index": "ALTER TABLE word_entries ADD COLUMN source_index INTEGER NOT NULL DEFAULT 0",
            "frequency": "ALTER TABLE word_entries ADD COLUMN frequency INTEGER NOT NULL DEFAULT 0",
            "forms": "ALTER TABLE word_entries ADD COLUMN forms TEXT NOT NULL DEFAULT ''",
            "example_sentence": "ALTER TABLE word_entries ADD COLUMN example_sentence TEXT NOT NULL DEFAULT ''",
            "example_sentence_cn": "ALTER TABLE word_entries ADD COLUMN example_sentence_cn TEXT NOT NULL DEFAULT ''",
            "mastery_level": "ALTER TABLE word_entries ADD COLUMN mastery_level INTEGER NOT NULL DEFAULT 0",
            "review_count": "ALTER TABLE word_entries ADD COLUMN review_count INTEGER NOT NULL DEFAULT 0",
            "correct_count": "ALTER TABLE word_entries ADD COLUMN correct_count INTEGER NOT NULL DEFAULT 0",
            "wrong_count": "ALTER TABLE word_entries ADD COLUMN wrong_count INTEGER NOT NULL DEFAULT 0",
            "last_reviewed_at": "ALTER TABLE word_entries ADD COLUMN last_reviewed_at TEXT",
            "learned_available_at": "ALTER TABLE word_entries ADD COLUMN learned_available_at TEXT",
        }

        for column, statement in migrations.items():
            if column not in columns:
                connection.execute(statement)

    @staticmethod
    def _upsert(
        connection: sqlite3.Connection,
        entry: WordEntry,
        preserve_progress: bool,
    ) -> None:
        progress_update = (
            ""
            if preserve_progress
            else """
                ,
                mastery_level = excluded.mastery_level,
                review_count = excluded.review_count,
                correct_count = excluded.correct_count,
                wrong_count = excluded.wrong_count,
                last_reviewed_at = excluded.last_reviewed_at,
                learned_available_at = excluded.learned_available_at
            """
        )
        connection.execute(
            f"""
            INSERT INTO word_entries (
                id,
                source_index,
                word,
                meaning,
                frequency,
                forms,
                example_sentence,
                example_sentence_cn,
                created_at,
                mastery_level,
                review_count,
                correct_count,
                wrong_count,
                last_reviewed_at,
                learned_available_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(word) DO UPDATE SET
                source_index = excluded.source_index,
                meaning = CASE
                    WHEN excluded.meaning = ''
                    THEN word_entries.meaning
                    ELSE excluded.meaning
                END,
                frequency = excluded.frequency,
                forms = excluded.forms,
                example_sentence = CASE
                    WHEN excluded.example_sentence = ''
                    THEN word_entries.example_sentence
                    ELSE excluded.example_sentence
                END,
                example_sentence_cn = CASE
                    WHEN excluded.example_sentence_cn = ''
                    THEN word_entries.example_sentence_cn
                    ELSE excluded.example_sentence_cn
                END
                {progress_update}
            """,
            (
                entry.id,
                entry.source_index,
                entry.word,
                entry.meaning,
                entry.frequency,
                entry.forms,
                entry.example_sentence,
                entry.example_sentence_cn,
                entry.created_at.isoformat(),
                entry.mastery_level,
                entry.review_count,
                entry.correct_count,
                entry.wrong_count,
                entry.last_reviewed_at.isoformat() if entry.last_reviewed_at else None,
                entry.learned_available_at.isoformat()
                if entry.learned_available_at
                else None,
            ),
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> WordEntry:
        return WordEntry(
            id=row["id"],
            word=row["word"],
            meaning=row["meaning"],
            source_index=row["source_index"],
            frequency=row["frequency"],
            forms=row["forms"],
            example_sentence=row["example_sentence"],
            example_sentence_cn=row["example_sentence_cn"],
            created_at=datetime.fromisoformat(row["created_at"]),
            mastery_level=row["mastery_level"],
            review_count=row["review_count"],
            correct_count=row["correct_count"],
            wrong_count=row["wrong_count"],
            last_reviewed_at=(
                datetime.fromisoformat(row["last_reviewed_at"])
                if row["last_reviewed_at"]
                else None
            ),
            learned_available_at=(
                datetime.fromisoformat(row["learned_available_at"])
                if row["learned_available_at"]
                else None
            ),
        )
