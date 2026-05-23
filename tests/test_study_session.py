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


def test_frequency_weight_mildly_prefers_frequent_words_without_dominating() -> None:
    low = WordEntry(word="rare", meaning="少见", frequency=1)
    high = WordEntry(word="common", meaning="常见", frequency=10_000)

    low_weight = StudySessionController._study_weight(low, high.frequency)
    high_weight = StudySessionController._study_weight(high, high.frequency)

    assert low_weight > 1.0
    assert high_weight == 1.6
    assert high_weight / low_weight < 1.6


def test_frequency_weight_falls_back_when_frequency_is_missing() -> None:
    entry = WordEntry(word="unknown", meaning="未知", frequency=0)

    assert StudySessionController._study_weight(entry, 0) == 1.0


def test_delete_current_removes_entry_and_continues_session() -> None:
    first = WordEntry(word="remove", meaning="删除")
    second = WordEntry(word="keep", meaning="保留")
    repository = InMemoryWordRepository([first, second])
    service = WordService(repository)
    session = StudySessionController(service)

    state = session.begin("learning")
    assert state.entry is not None
    deleted_id = state.entry.id

    next_state = session.delete_current()

    assert service.get_word(deleted_id) is None
    assert next_state is not None
    assert next_state.kind in {"entry", "complete"}
    if next_state.kind == "entry":
        assert next_state.entry is not None
        assert next_state.entry.id != deleted_id


def test_mark_buttons_advance_to_next_normal_card_without_next_button() -> None:
    first = WordEntry(word="first", meaning="第一")
    second = WordEntry(word="second", meaning="第二")
    service = WordService(InMemoryWordRepository([first, second]))
    session = StudySessionController(service)

    state = session.begin("learning")
    assert state.entry is not None
    first_id = state.entry.id

    next_state = session.mark_current("known")

    assert next_state is not None
    assert next_state.kind == "entry"
    assert next_state.entry is not None
    assert next_state.entry.id != first_id
    assert next_state.history_view is False
    assert next_state.can_show_next is False


def test_next_button_is_only_for_history_view() -> None:
    first = WordEntry(word="first", meaning="第一")
    second = WordEntry(word="second", meaning="第二")
    service = WordService(InMemoryWordRepository([first, second]))
    session = StudySessionController(service)

    first_state = session.begin("learning")
    assert first_state.can_show_next is False
    session.mark_current("known")

    history_state = session.show_previous_word()

    assert history_state is not None
    assert history_state.history_view is True
    assert history_state.can_show_next is True


def test_history_next_returns_to_current_sequence_word_without_redrawing() -> None:
    entries = [
        WordEntry(word="first", meaning="第一"),
        WordEntry(word="second", meaning="第二"),
        WordEntry(word="third", meaning="第三"),
    ]
    service = WordService(InMemoryWordRepository(entries))
    session = StudySessionController(service)

    first_state = session.begin("learning")
    assert first_state.entry is not None
    second_state = session.mark_current("known")
    assert second_state is not None
    assert second_state.entry is not None
    current_id = second_state.entry.id

    history_state = session.show_previous_word()
    assert history_state is not None
    assert history_state.history_view is True

    resumed_state = session.continue_from_history()

    assert resumed_state.entry is not None
    assert resumed_state.entry.id == current_id
    assert resumed_state.history_view is False


def test_insert_word_at_front_uses_max_frequency_and_reindexes() -> None:
    first = WordEntry(word="alpha", meaning="阿尔法", source_index=1, frequency=7)
    second = WordEntry(word="beta", meaning="贝塔", source_index=2, frequency=3)
    service = WordService(InMemoryWordRepository([first, second]))

    inserted = service.insert_word_at_front("gamma", "伽马")

    entries = service.list_words()
    assert inserted.source_index == 1
    assert inserted.frequency == 7
    assert [entry.word for entry in entries] == ["gamma", "alpha", "beta"]
    assert [entry.source_index for entry in entries] == [1, 2, 3]
