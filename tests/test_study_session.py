from __future__ import annotations

from datetime import datetime, timedelta

from wordpycket.application.services import WordService
from wordpycket.application.study_session import StudySessionController
from wordpycket.domain.entities import WordEntry
from wordpycket.domain.repositories import WordRepository


class InMemoryWordRepository(WordRepository):
    def __init__(self, entries: list[WordEntry]) -> None:
        self._entries = {entry.id: entry for entry in entries}

    def list(self) -> list[WordEntry]:
        return list(self._entries.values())

    def get(self, entry_id: str) -> WordEntry | None:
        return self._entries.get(entry_id)

    def save(self, entry: WordEntry) -> None:
        self._entries[entry.id] = entry

    def save_many(self, entries: list[WordEntry]) -> int:
        for entry in entries:
            self.save(entry)
        return len(entries)

    def delete(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)

    def reset_progress(self) -> None:
        self._entries = {
            entry.id: WordEntry(
                id=entry.id,
                word=entry.word,
                meaning=entry.meaning,
                source_index=entry.source_index,
                frequency=entry.frequency,
                forms=entry.forms,
                example_sentence=entry.example_sentence,
                example_sentence_cn=entry.example_sentence_cn,
                created_at=entry.created_at,
            )
            for entry in self._entries.values()
        }

    def replace_all(self, entries: list[WordEntry]) -> int:
        self._entries = {entry.id: entry for entry in entries}
        return len(entries)

    def update_examples(
        self,
        entry_id: str,
        example_sentence: str,
        example_sentence_cn: str,
    ) -> None:
        entry = self._entries[entry_id]
        self._entries[entry_id] = WordEntry(
            id=entry.id,
            word=entry.word,
            meaning=entry.meaning,
            source_index=entry.source_index,
            frequency=entry.frequency,
            forms=entry.forms,
            example_sentence=example_sentence,
            example_sentence_cn=example_sentence_cn,
            created_at=entry.created_at,
            mastery_level=entry.mastery_level,
            review_count=entry.review_count,
            correct_count=entry.correct_count,
            wrong_count=entry.wrong_count,
            last_reviewed_at=entry.last_reviewed_at,
            learned_available_at=entry.learned_available_at,
        )

    def update_text(
        self,
        entry_id: str,
        word: str,
        meaning: str,
        forms: str,
    ) -> None:
        entry = self._entries[entry_id]
        self._entries[entry_id] = WordEntry(
            id=entry.id,
            word=word,
            meaning=meaning,
            source_index=entry.source_index,
            frequency=entry.frequency,
            forms=forms,
            example_sentence=entry.example_sentence,
            example_sentence_cn=entry.example_sentence_cn,
            created_at=entry.created_at,
            mastery_level=entry.mastery_level,
            review_count=entry.review_count,
            correct_count=entry.correct_count,
            wrong_count=entry.wrong_count,
            last_reviewed_at=entry.last_reviewed_at,
            learned_available_at=entry.learned_available_at,
        )


def test_word_entering_review_pool_is_visible_after_switching_modes() -> None:
    entry = WordEntry(word="vector", meaning="向量")
    service = WordService(InMemoryWordRepository([entry]))
    session = StudySessionController(service)

    learning_state = session.begin("learning")
    assert learning_state.entry == entry

    session.mark_current("definitely_known")
    session.leave_active_session()
    session.reset()

    review_state = session.begin("review")

    assert review_state.kind == "entry"
    assert review_state.entry is not None
    assert review_state.entry.id == entry.id


def test_same_mode_does_not_repeat_last_session_words() -> None:
    entry = WordEntry(word="matrix", meaning="矩阵")
    service = WordService(InMemoryWordRepository([entry]))
    session = StudySessionController(service)

    assert session.begin("learning").entry == entry
    session.leave_active_session()
    session.reset()

    next_state = session.begin("learning")

    assert next_state.kind == "empty"


def test_word_moves_from_learning_to_review_after_five_known_marks() -> None:
    entry = WordEntry(word="tensor", meaning="张量")

    for _ in range(5):
        entry = entry.mark_known()

    assert entry.correct_count == 5
    assert entry.status == "复习池"
    assert entry.is_in_review_pool
    assert not entry.is_learned


def test_review_word_becomes_learned_only_after_delay_expires() -> None:
    reviewed_at = datetime(2026, 5, 23, 10, 0, 0)
    entry = WordEntry(word="kernel", meaning="内核", correct_count=5)

    entry = entry.mark_known(reviewed_at=reviewed_at)
    entry = entry.mark_known(reviewed_at=reviewed_at)
    entry = entry.mark_known(reviewed_at=reviewed_at)

    assert entry.correct_count == 8
    assert entry.learned_available_at == reviewed_at + timedelta(hours=24)
    assert entry.status == "复习池"
    assert entry.is_in_review_pool

    learned_entry = WordEntry(
        word=entry.word,
        meaning=entry.meaning,
        correct_count=entry.correct_count,
        learned_available_at=datetime.now() - timedelta(seconds=1),
    )

    assert learned_entry.status == "已掌握"
    assert learned_entry.is_learned
    assert not learned_entry.is_in_review_pool


def test_definitely_known_enters_review_pool_without_becoming_learned() -> None:
    entry = WordEntry(word="gradient", meaning="梯度")

    entry = entry.mark_definitely_known()

    assert entry.correct_count == 5
    assert entry.status == "复习池"
    assert entry.is_in_review_pool
    assert entry.learned_available_at is None
