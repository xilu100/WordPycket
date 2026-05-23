import random
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
        return self.reload()

    def reload(self, selected_id: str | None = None) -> StudyCardState:
        self._entries = self.mode_entries()
        if not self._entries:
            return self._empty_state()

        if selected_id is not None:
            return self.show_entry_by_id(selected_id)
        return self.show_random_word()

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
    ) -> StudyCardState:
        entry = self._find_entry_by_id(entry_id, allow_outside_pool=history_view)
        if entry is None:
            return self.show_random_word()

        is_reopened_seen_entry = (
            entry.id in self._session_seen_ids
            and (
                self._current_entry is None
                or self._current_entry.id != entry.id
            )
        )
        is_history_view = history_view or is_reopened_seen_entry
        if is_history_view:
            self._history_position = self._history_index(entry.id)
        self._entry_index = self._entry_index_for(entry.id)
        return self._entry_state(
            entry,
            history_view=is_history_view,
            record_history=record_history,
        )

    def show_random_word(self) -> StudyCardState:
        if not self._entries:
            return self._complete_state()

        candidates = [
            (index, entry)
            for index, entry in enumerate(self._entries)
            if entry.id not in self._session_seen_ids
        ]
        if not candidates:
            return self._complete_state()

        self._entry_index, entry = self._weighted_random_choice(candidates)
        return self._entry_state(entry)

    def show_previous_word(self) -> StudyCardState | None:
        if not self._history:
            return None

        if self._current_is_history_view:
            if self._history_position <= 0:
                return None
            self._history_position -= 1

        previous_id = self._history[self._history_position]
        return self.show_entry_by_id(previous_id, history_view=True, record_history=False)

    def continue_from_history(self) -> StudyCardState:
        if self._history_position < len(self._history) - 1:
            self._history_position += 1
            next_id = self._history[self._history_position]
            return self.show_entry_by_id(next_id, history_view=True, record_history=False)

        return self.show_random_word()

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
        self._entries = self.mode_entries()
        return self.show_random_word()

    def _empty_state(self) -> StudyCardState:
        self._current_entry = None
        self._current_is_history_view = False
        self._history = []
        self._history_position = -1
        if self._mode == "review":
            message = "复习池暂无词条。学习池中会 5 次或点击绝对会后会进入复习池。"
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
            word_text=entry.word,
            meaning_text=entry.meaning,
            forms_text=f"词形: {entry.forms}" if entry.forms else "",
            example_text=f"例句: {entry.example_sentence}" if entry.example_sentence else "",
            example_cn_text=f"例句中文: {entry.example_sentence_cn}" if entry.example_sentence_cn else "",
            meta_text=(
                f"{self._entry_index + 1} / {len(self._entries)} | "
                f"频率 {entry.frequency} | {entry.status} | "
                f"会 {entry.correct_count} / 不会 {entry.wrong_count}"
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

    @staticmethod
    def _weighted_random_choice(candidates: list[tuple[int, WordEntry]]) -> tuple[int, WordEntry]:
        weights = [
            (entry.wrong_count + 1) / (entry.correct_count + 1)
            for _index, entry in candidates
        ]
        return random.choices(candidates, weights=weights, k=1)[0]

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
