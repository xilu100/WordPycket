from datetime import datetime

from wordpycket.domain.entities import WordEntry
from wordpycket.domain.repositories import WordRepository


class WordService:
    def __init__(self, repository: WordRepository) -> None:
        self._repository = repository

    def add_word(
        self,
        word: str,
        meaning: str,
        source_index: int = 0,
        frequency: int = 0,
        forms: str = "",
    ) -> WordEntry:
        entry = WordEntry(
            word=word,
            meaning=meaning,
            source_index=source_index,
            frequency=frequency,
            forms=forms,
        )
        self._repository.save(entry)
        return entry

    def import_words(self, entries: list[WordEntry]) -> int:
        return self._repository.save_many(entries)

    def list_words(self, query: str = "") -> list[WordEntry]:
        entries = sorted(
            self._repository.list(),
            key=lambda entry: (entry.source_index == 0, entry.source_index, -entry.frequency),
        )
        keyword = query.strip().lower()
        if not keyword:
            return entries

        return [
            entry
            for entry in entries
            if keyword in entry.word.lower()
            or keyword in entry.meaning.lower()
            or keyword in entry.forms.lower()
        ]

    def next_review_word(self) -> WordEntry | None:
        entries = self._repository.list()
        if not entries:
            return None

        now = datetime.now()
        due_entries = [entry for entry in entries if entry.next_review_at() <= now]
        if not due_entries:
            return None

        return min(
            due_entries,
            key=lambda entry: (
                entry.next_review_at(),
                entry.mastery_level,
                -entry.frequency,
                entry.source_index,
            ),
        )

    def mark_known(self, entry_id: str) -> WordEntry | None:
        entry = self._repository.get(entry_id)
        if entry is None:
            return None

        updated = entry.mark_known()
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
