from __future__ import annotations

from abc import ABC, abstractmethod

from wordpycket.domain.entities import WordEntry


class WordRepository(ABC):
    @abstractmethod
    def list(self) -> list[WordEntry]:
        raise NotImplementedError

    @abstractmethod
    def get(self, entry_id: str) -> WordEntry | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, entry: WordEntry) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_many(self, entries: list[WordEntry]) -> int:
        raise NotImplementedError

    @abstractmethod
    def delete(self, entry_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_progress(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def replace_all(self, entries: list[WordEntry]) -> int:
        raise NotImplementedError

    @abstractmethod
    def update_examples(
        self,
        entry_id: str,
        example_sentence: str,
        example_sentence_cn: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_text(
        self,
        entry_id: str,
        word: str,
        meaning: str,
        forms: str,
    ) -> None:
        raise NotImplementedError
