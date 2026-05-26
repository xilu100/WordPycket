import random
import math
from dataclasses import dataclass
from typing import Literal

from wordpycket.application.services import WordService
from wordpycket.domain.entities import WordEntry


StudyMode = Literal["learning", "review"]
MarkResult = Literal["known", "unknown", "definitely_known"]


@dataclass(frozen=True)
class StudyCardState:
    kind: Literal["entry", "empty", "complete"]
    entry: WordEntry | None = None
    history_view: bool = False
    can_show_previous: bool = False
    can_show_next: bool = False
    meta_text: str = ""
    word_text: str = ""
    meaning_text: str = ""
    forms_text: str = ""
    example_text: str = ""
    example_cn_text: str = ""

    @property
    def has_entry(self) -> bool:
        return self.entry is not None


class StudySessionController:
    def __init__(self, service: WordService) -> None:
        self._service = service
        self._mode: StudyMode | None = None
        self._current_entry: WordEntry | None = None
        self._current_is_history_view = False
        self._entries: list[WordEntry] = []
        self._history: list[str] = []
        self._history_position = -1
        self._session_seen_ids: set[str] = set()
        self._last_session_seen_ids: set[str] = set()
        self._last_session_mode: StudyMode | None = None
        self._entry_index = 0
        self._sequence_ids: list[str] = []
        self._sequence_position = -1
        self._resume_entry_id: str | None = None

    @property
    def mode(self) -> StudyMode | None:
        return self._mode

    @property
    def current_entry(self) -> WordEntry | None:
        return self._current_entry

    def pool_counts(self) -> dict[str, int]:
        entries = self._service.list_words()
        return {
            "learning": sum(1 for entry in entries if entry.correct_count < 5),
            "review": sum(1 for entry in entries if entry.is_in_review_pool),
            "total": len(entries),
        }

    def leave_active_session(self) -> None:
        if self._mode in {"learning", "review"}:
            self._last_session_seen_ids = set(self._session_seen_ids)
            self._last_session_mode = self._mode

    def reset(self) -> None:
        self._mode = None
        self._current_entry = None
        self._current_is_history_view = False
        self._entries = []
        self._history = []
        self._history_position = -1
        self._session_seen_ids = set()
        self._entry_index = 0
        self._sequence_ids = []
        self._sequence_position = -1
        self._resume_entry_id = None

    def clear_last_session(self) -> None:
        self._last_session_seen_ids = set()
        self._last_session_mode = None

    def begin(self, mode: StudyMode) -> StudyCardState:
        self._mode = mode
        self._current_entry = None
        self._current_is_history_view = False
        self._entries = []
        self._history = []
        self._history_position = -1
        self._session_seen_ids = set()
        self._entry_index = 0
        self._sequence_ids = []
        self._sequence_position = -1
        self._resume_entry_id = None
        return self.reload()

    def reload(self, selected_id: str | None = None) -> StudyCardState:
        self._entries = self._weighted_random_order(self.mode_entries())
        self._sequence_ids = [entry.id for entry in self._entries]
        self._sequence_position = -1
        if not self._entries:
            return self._empty_state()

        if selected_id is not None:
            return self.show_entry_by_id(selected_id)
        return self.show_next_word()

    def mode_entries(self, query: str = "") -> list[WordEntry]:
        entries = self._service.list_words(query)
        if self._mode in {"learning", "review"} and self._mode == self._last_session_mode:
            entries = [
                entry
                for entry in entries
                if entry.id not in self._last_session_seen_ids
            ]
        if self._mode == "learning":
            return [entry for entry in entries if entry.correct_count < 5]
        if self._mode == "review":
            return [entry for entry in entries if entry.is_in_review_pool]
        return entries

    def show_entry_by_id(
        self,
        entry_id: str,
        history_view: bool = False,
        record_history: bool = True,
        allow_seen_normal: bool = False,
    ) -> StudyCardState:
        entry = self._find_entry_by_id(entry_id, allow_outside_pool=history_view)
        if entry is None:
            return self.show_next_word()

        is_reopened_seen_entry = (
            not allow_seen_normal
            and not history_view
            and
            entry.id in self._session_seen_ids
            and (
                self._current_entry is None
                or self._current_entry.id != entry.id
            )
        )
        is_history_view = history_view or is_reopened_seen_entry
        if is_history_view:
            self._history_position = self._history_index(entry.id)
        sequence_index = self._sequence_index_for(entry.id)
        if sequence_index >= 0 and not is_history_view:
            self._sequence_position = sequence_index
        self._entry_index = self._entry_index_for(entry.id)
        return self._entry_state(
            entry,
            history_view=is_history_view,
            record_history=record_history,
        )

    def show_random_word(self) -> StudyCardState:
        return self.show_next_word()

    def show_next_word(self) -> StudyCardState:
        while self._sequence_position + 1 < len(self._sequence_ids):
            self._sequence_position += 1
            entry_id = self._sequence_ids[self._sequence_position]
            if entry_id in self._session_seen_ids:
                continue
            entry = self._find_entry_by_id(entry_id, allow_outside_pool=True)
            if entry is None:
                continue
            self._entry_index = self._entry_index_for(entry.id)
            return self._entry_state(entry)

        return self._complete_state()

    def show_previous_word(self) -> StudyCardState | None:
        if not self._history:
            return None

        if self._current_is_history_view:
            if self._history_position <= 0:
                return None
            self._history_position -= 1
        elif self._current_entry is not None:
            self._resume_entry_id = self._current_entry.id

        previous_id = self._history[self._history_position]
        return self.show_entry_by_id(previous_id, history_view=True, record_history=False)

    def continue_from_history(self) -> StudyCardState:
        if self._history_position < len(self._history) - 1:
            self._history_position += 1
            next_id = self._history[self._history_position]
            return self.show_entry_by_id(next_id, history_view=True, record_history=False)

        if self._resume_entry_id is not None:
            resume_entry_id = self._resume_entry_id
            self._resume_entry_id = None
            return self.show_entry_by_id(
                resume_entry_id,
                history_view=False,
                record_history=False,
                allow_seen_normal=True,
            )

        return self.show_next_word()

    def mark_current(self, result: MarkResult) -> StudyCardState | None:
        if self._current_entry is None:
            return None

        current_id = self._current_entry.id
        if result == "known":
            self._service.mark_known(current_id)
        elif result == "definitely_known":
            self._service.mark_definitely_known(current_id)
        else:
            self._service.mark_unknown(current_id)

        self._record_history(current_id)
        self._refresh_sequence_entries()
        return self.show_next_word()

    def delete_current(self) -> StudyCardState | None:
        if self._current_entry is None:
            return None

        entry_id = self._current_entry.id
        self._service.delete_word(entry_id)
        self._history = [history_id for history_id in self._history if history_id != entry_id]
        self._history_position = min(self._history_position, len(self._history) - 1)
        self._session_seen_ids.discard(entry_id)
        self._last_session_seen_ids.discard(entry_id)
        self._entries = [entry for entry in self._entries if entry.id != entry_id]
        self._sequence_ids = [sequence_id for sequence_id in self._sequence_ids if sequence_id != entry_id]
        self._sequence_position = min(self._sequence_position, len(self._sequence_ids) - 1)
        if self._resume_entry_id == entry_id:
            self._resume_entry_id = None
        self._current_entry = None
        self._current_is_history_view = False
        return self.show_next_word()

    def _empty_state(self) -> StudyCardState:
        self._current_entry = None
        self._current_is_history_view = False
        self._history = []
        self._history_position = -1
        if self._mode == "review":
            message = "复习池暂无词条。连续认识 5 次或点击“很熟”后，词条会进入复习池。"
        elif self._mode == "learning":
            message = "学习池暂无词条。"
        else:
            message = "请确认 CSV 已放在 input/word_frequency.csv。"
        return StudyCardState(
            kind="empty",
            word_text="暂无词条",
            meta_text=message,
            can_show_previous=self._can_show_previous(False),
        )

    def _complete_state(self) -> StudyCardState:
        self._current_entry = None
        self._current_is_history_view = False
        return StudyCardState(
            kind="complete",
            word_text="本次完成",
            meta_text="本次会话中的单词已经全部显示。返回主页后再进入会开始新会话。",
            can_show_previous=self._can_show_previous(False),
        )

    def _entry_state(
        self,
        entry: WordEntry,
        history_view: bool = False,
        record_history: bool = False,
    ) -> StudyCardState:
        self._current_entry = entry
        self._current_is_history_view = history_view
        if not history_view:
            self._session_seen_ids.add(entry.id)
            if record_history:
                self._record_history(entry.id)

        learned_note = (
            f" | 已学习时间 {entry.learned_available_at:%Y-%m-%d %H:%M}"
            if entry.learned_available_at and not entry.is_learned
            else ""
        )
        history_note = (
            f" | 历史 {self._history_position + 1} / {len(self._history)}"
            if self._history and history_view
            else ""
        )
        return StudyCardState(
            kind="entry",
            entry=entry,
            history_view=history_view,
            can_show_previous=self._can_show_previous(history_view),
            can_show_next=history_view,
            word_text=entry.word,
            meaning_text=entry.meaning,
            forms_text=f"词形: {entry.forms}" if entry.forms else "",
            example_text=f"例句: {entry.example_sentence}" if entry.example_sentence else "",
            example_cn_text=f"例句翻译: {entry.example_sentence_cn}" if entry.example_sentence_cn else "",
            meta_text=(
                f"{self._entry_index + 1} / {len(self._entries)} | "
                f"频率 {entry.frequency} | {entry.status} | "
                f"认识 {entry.correct_count} / 不认识 {entry.wrong_count}"
                f"{learned_note}"
                f"{history_note}"
            ),
        )

    def _find_entry_by_id(self, entry_id: str, allow_outside_pool: bool = False) -> WordEntry | None:
        entries = self._service.list_words() if allow_outside_pool else self._entries
        for entry in entries:
            if entry.id == entry_id:
                return entry
        return None

    def _entry_index_for(self, entry_id: str) -> int:
        for index, entry in enumerate(self._entries):
            if entry.id == entry_id:
                return index
        return self._entry_index

    def _sequence_index_for(self, entry_id: str) -> int:
        try:
            return self._sequence_ids.index(entry_id)
        except ValueError:
            return -1

    def _refresh_sequence_entries(self) -> None:
        latest_entries = {entry.id: entry for entry in self._service.list_words()}
        refreshed_entries: list[WordEntry] = []
        refreshed_ids: list[str] = []
        for entry_id in self._sequence_ids:
            entry = latest_entries.get(entry_id)
            if entry is None:
                continue
            refreshed_entries.append(entry)
            refreshed_ids.append(entry_id)
        self._entries = refreshed_entries
        self._sequence_ids = refreshed_ids

    @staticmethod
    def _weighted_random_choice(candidates: list[tuple[int, WordEntry]]) -> tuple[int, WordEntry]:
        max_frequency = max((entry.frequency for _index, entry in candidates), default=0)
        weights = [
            StudySessionController._study_weight(entry, max_frequency)
            for _index, entry in candidates
        ]
        return random.choices(candidates, weights=weights, k=1)[0]

    @staticmethod
    def _weighted_random_order(entries: list[WordEntry]) -> list[WordEntry]:
        remaining = list(entries)
        ordered: list[WordEntry] = []
        max_frequency = max((entry.frequency for entry in remaining), default=0)
        while remaining:
            weights = [
                StudySessionController._study_weight(entry, max_frequency)
                for entry in remaining
            ]
            selected = random.choices(remaining, weights=weights, k=1)[0]
            ordered.append(selected)
            remaining.remove(selected)
        return ordered

    @staticmethod
    def _study_weight(entry: WordEntry, max_frequency: int) -> float:
        learning_weight = (entry.wrong_count + 1) / (entry.correct_count + 1)
        frequency_weight = StudySessionController._frequency_weight(entry.frequency, max_frequency)
        return learning_weight * frequency_weight

    @staticmethod
    def _frequency_weight(frequency: int, max_frequency: int) -> float:
        if frequency <= 0 or max_frequency <= 0:
            return 1.0
        normalized = math.log1p(frequency) / math.log1p(max_frequency)
        return 1.0 + min(0.6, normalized * 0.6)

    def _record_history(self, entry_id: str) -> None:
        if self._history_position < len(self._history) - 1:
            self._history = self._history[: self._history_position + 1]

        if self._history and self._history[-1] == entry_id:
            self._history_position = len(self._history) - 1
            return

        self._history.append(entry_id)
        self._history_position = len(self._history) - 1

    def _history_index(self, entry_id: str) -> int:
        try:
            return self._history.index(entry_id)
        except ValueError:
            return self._history_position

    def _can_show_previous(self, history_view: bool) -> bool:
        if not self._history:
            return False
        if history_view:
            return self._history_position > 0
        return self._history_position >= 0
