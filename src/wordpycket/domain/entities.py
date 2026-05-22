from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import uuid4


@dataclass(frozen=True)
class WordEntry:
    word: str
    meaning: str
    source_index: int = 0
    frequency: int = 0
    forms: str = ""
    example_sentence: str = ""
    example_sentence_cn: str = ""
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    mastery_level: int = 0
    review_count: int = 0
    correct_count: int = 0
    wrong_count: int = 0
    last_reviewed_at: datetime | None = None
    learned_available_at: datetime | None = None

    def __post_init__(self) -> None:
        normalized_word = self.word.strip()
        normalized_meaning = self.meaning.strip()
        normalized_forms = self.forms.strip()
        normalized_example_sentence = self.example_sentence.strip()
        normalized_example_sentence_cn = self.example_sentence_cn.strip()

        if not normalized_word:
            raise ValueError("单词不能为空。")
        if not normalized_meaning:
            raise ValueError("释义不能为空。")

        object.__setattr__(self, "word", normalized_word)
        object.__setattr__(self, "meaning", normalized_meaning)
        object.__setattr__(self, "forms", normalized_forms)
        object.__setattr__(self, "example_sentence", normalized_example_sentence)
        object.__setattr__(self, "example_sentence_cn", normalized_example_sentence_cn)

    @property
    def status(self) -> str:
        if self.is_learned:
            return "已学习池"
        if self.correct_count >= 5:
            return "复习池"
        return "学习池"

    @property
    def is_learned(self) -> bool:
        return (
            self.learned_available_at is not None
            and self.learned_available_at <= datetime.now()
        )

    @property
    def is_in_review_pool(self) -> bool:
        return self.correct_count >= 5 and not self.is_learned

    def mark_known(self, reviewed_at: datetime | None = None) -> "WordEntry":
        reviewed_at = reviewed_at or datetime.now()
        correct_count = self.correct_count + 1
        learned_available_at = self.learned_available_at
        if learned_available_at is None and self.correct_count >= 5 and correct_count >= 8:
            learned_available_at = reviewed_at + timedelta(hours=24)

        return self._review(
            mastery_level=min(self.mastery_level + 1, 5),
            correct_count=correct_count,
            wrong_count=self.wrong_count,
            reviewed_at=reviewed_at,
            learned_available_at=learned_available_at,
        )

    def mark_definitely_known(self, reviewed_at: datetime | None = None) -> "WordEntry":
        return self._review(
            mastery_level=5,
            correct_count=max(self.correct_count + 1, 5),
            wrong_count=self.wrong_count,
            reviewed_at=reviewed_at,
            learned_available_at=self.learned_available_at,
        )

    def mark_unknown(self, reviewed_at: datetime | None = None) -> "WordEntry":
        return self._review(
            mastery_level=max(self.mastery_level - 1, -5),
            correct_count=self.correct_count,
            wrong_count=self.wrong_count + 1,
            reviewed_at=reviewed_at,
            learned_available_at=self.learned_available_at,
        )

    def next_review_at(self) -> datetime:
        if self.last_reviewed_at is None:
            return datetime.min
        if self.mastery_level <= 0:
            return self.last_reviewed_at

        interval_days = 2 ** min(self.mastery_level - 1, 6)
        return self.last_reviewed_at + timedelta(days=interval_days)

    def _review(
        self,
        mastery_level: int,
        correct_count: int,
        wrong_count: int,
        reviewed_at: datetime | None,
        learned_available_at: datetime | None,
    ) -> "WordEntry":
        return WordEntry(
            id=self.id,
            word=self.word,
            meaning=self.meaning,
            source_index=self.source_index,
            frequency=self.frequency,
            forms=self.forms,
            example_sentence=self.example_sentence,
            example_sentence_cn=self.example_sentence_cn,
            created_at=self.created_at,
            mastery_level=mastery_level,
            review_count=self.review_count + 1,
            correct_count=correct_count,
            wrong_count=wrong_count,
            last_reviewed_at=reviewed_at or datetime.now(),
            learned_available_at=learned_available_at,
        )
