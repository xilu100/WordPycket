import random
import re
from dataclasses import replace

from wordpycket.domain.entities import WordEntry
from wordpycket.domain.repositories import WordRepository


class WordService:
    def __init__(self, repository: WordRepository) -> None:
        self._repository = repository

    def use_repository(self, repository: WordRepository) -> None:
        self._repository = repository

    def add_word(
        self,
        word: str,
        meaning: str,
        source_index: int = 0,
        frequency: int = 0,
        forms: str = "",
        example_sentence: str = "",
        example_sentence_cn: str = "",
    ) -> WordEntry:
        entry = WordEntry(
            word=word,
            meaning=meaning,
            source_index=source_index,
            frequency=frequency,
            forms=forms,
            example_sentence=example_sentence,
            example_sentence_cn=example_sentence_cn,
        )
        self._repository.save(entry)
        return entry

    def import_words(self, entries: list[WordEntry]) -> int:
        return self._repository.save_many(entries)

    def get_word(self, entry_id: str) -> WordEntry | None:
        return self._repository.get(entry_id)

    def list_words(self, query: str = "") -> list[WordEntry]:
        entries = sorted(
            self._repository.list(),
            key=lambda entry: (entry.source_index == 0, entry.source_index, -entry.frequency),
        )
        keyword = query.strip().lower()
        if not keyword:
            return entries
        keywords = {
            keyword,
            self._normalize_search_text(keyword),
        }

        return [
            entry
            for entry in entries
            if any(
                candidate in self._entry_search_text(entry)
                for candidate in keywords
                if candidate
            )
        ]

    def next_review_word(self) -> WordEntry | None:
        entries = [entry for entry in self._repository.list() if not entry.is_learned]
        if not entries:
            return None

        return random.choice(entries)

    def mark_known(self, entry_id: str) -> WordEntry | None:
        entry = self._repository.get(entry_id)
        if entry is None:
            return None

        updated = entry.mark_known()
        self._repository.save(updated)
        return updated

    def mark_definitely_known(self, entry_id: str) -> WordEntry | None:
        entry = self._repository.get(entry_id)
        if entry is None:
            return None

        updated = entry.mark_definitely_known()
        self._repository.save(updated)
        return updated

    def mark_unknown(self, entry_id: str) -> WordEntry | None:
        entry = self._repository.get(entry_id)
        if entry is None:
            return None

        updated = entry.mark_unknown()
        self._repository.save(updated)
        return updated

    def delete_word(self, entry_id: str) -> None:
        self._repository.delete(entry_id)

    def reset_progress(self) -> None:
        self._repository.reset_progress()

    def replace_words(self, entries: list[WordEntry]) -> int:
        return self._repository.replace_all(entries)

    def insert_word_at_front(
        self,
        word: str,
        meaning: str,
        forms: str = "",
        example_sentence: str = "",
        example_sentence_cn: str = "",
    ) -> WordEntry:
        entries = self.list_words()
        max_frequency = max((entry.frequency for entry in entries), default=0)
        new_entry = WordEntry(
            source_index=1,
            word=word,
            meaning=meaning,
            frequency=max_frequency,
            forms=forms,
            example_sentence=example_sentence,
            example_sentence_cn=example_sentence_cn,
        )
        reindexed_entries = [new_entry] + [
            replace(entry, source_index=index)
            for index, entry in enumerate(entries, start=2)
        ]
        self._repository.replace_all(reindexed_entries)
        return new_entry

    def update_examples(
        self,
        entry_id: str,
        example_sentence: str,
        example_sentence_cn: str,
    ) -> WordEntry | None:
        self._repository.update_examples(
            entry_id,
            example_sentence,
            example_sentence_cn,
        )
        return self._repository.get(entry_id)

    def update_text(
        self,
        entry_id: str,
        word: str,
        meaning: str,
        forms: str,
    ) -> WordEntry | None:
        self._repository.update_text(
            entry_id,
            word,
            meaning,
            forms,
        )
        return self._repository.get(entry_id)

    @classmethod
    def _entry_search_text(cls, entry: WordEntry) -> str:
        values = (
            entry.word,
            entry.meaning,
            entry.forms,
            entry.example_sentence,
            entry.example_sentence_cn,
        )
        raw_text = " ".join(value.lower() for value in values)
        return f"{raw_text} {cls._normalize_search_text(raw_text)}"

    @staticmethod
    def _normalize_search_text(text: str) -> str:
        normalized = re.sub(r"[_]+", " ", text)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()
