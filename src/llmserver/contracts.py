from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WordEntry:
    word: str
    meaning: str
    source_index: int = 0
    frequency: int = 0
    forms: str = ""
    example_sentence: str = ""
    example_sentence_cn: str = ""


@dataclass(frozen=True)
class GeneratedExample:
    example_sentence: str
    example_sentence_cn: str
    meaning: str = ""


@dataclass(frozen=True)
class GeneratedCorrection:
    corrected_word: str
    note: str = ""
    should_update: bool = False


@dataclass(frozen=True)
class GeneratedExplanation:
    explanation: str


@dataclass(frozen=True)
class ModelStatus:
    path: Path | None
    is_user_model: bool
    downloaded: bool = False


@dataclass(frozen=True)
class DeviceStatus:
    requested: str
    detected: str
    selected: str | None
    gpu_offload_supported: bool | None
    error: str = ""


@dataclass(frozen=True)
class ModelCheckResult:
    model: ModelStatus
    device: DeviceStatus
    smoke_test_passed: bool
