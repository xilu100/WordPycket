from __future__ import annotations

from wordpycket.domain.entities import WordEntry
from pathlib import Path

from wordpycket.infrastructure.example_generator import GeneratedExample, LocalLlmExampleGenerator
from wordpycket.presentation.qt_app import BatchWorker


class GeneratorWithUnsafeParallelPath:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.generate_isolated_calls = 0
        self.generate_many_calls = 0

    def generate(self, entry: WordEntry, scope: str = "") -> GeneratedExample:
        self.generate_calls += 1
        return GeneratedExample(
            example_sentence=f"{entry.word} is useful.",
            example_sentence_cn=f"{entry.word} 很有用。",
        )

    def generate_isolated(self, entry: WordEntry, scope: str = "") -> GeneratedExample:
        self.generate_isolated_calls += 1
        return GeneratedExample(
            example_sentence=f"{entry.word} is isolated.",
            example_sentence_cn=f"{entry.word} 已隔离。",
        )

    def generate_many(self, *args, **kwargs):
        self.generate_many_calls += 1
        raise AssertionError("GUI batch worker must not use the parallel llama path.")

    def correct_entry(self, entry: WordEntry, scope: str = ""):
        raise NotImplementedError


class GeneratorWithOneCrashedChild:
    def generate_isolated(self, entry: WordEntry, scope: str = "") -> GeneratedExample:
        if entry.word == "crash":
            raise RuntimeError("模型子进程退出代码 -1073741819。")
        return GeneratedExample(
            example_sentence=f"{entry.word} survived.",
            example_sentence_cn=f"{entry.word} 已完成。",
        )

    def generate(self, entry: WordEntry, scope: str = "") -> GeneratedExample:
        raise NotImplementedError

    def correct_entry(self, entry: WordEntry, scope: str = ""):
        raise NotImplementedError


def test_batch_worker_uses_isolated_sequential_generation_for_gui_stability() -> None:
    entries = [
        WordEntry(word="vector", meaning="向量"),
        WordEntry(word="matrix", meaning="矩阵"),
    ]
    generator = GeneratorWithUnsafeParallelPath()
    worker = BatchWorker("补充", entries, "", generator, lambda: "running")

    results, errors, workers = worker._generate(lambda *_args: None)

    assert errors == []
    assert workers == 1
    assert len(results) == 2
    assert generator.generate_calls == 0
    assert generator.generate_isolated_calls == 2
    assert generator.generate_many_calls == 0


def test_batch_worker_keeps_running_when_one_isolated_generation_crashes() -> None:
    entries = [
        WordEntry(word="vector", meaning="向量"),
        WordEntry(word="crash", meaning="崩溃"),
        WordEntry(word="matrix", meaning="矩阵"),
    ]
    worker = BatchWorker("补充", entries, "", GeneratorWithOneCrashedChild(), lambda: "running")

    results, errors, workers = worker._generate(lambda *_args: None)

    assert workers == 1
    assert [entry.word for entry, _generated in results] == ["vector", "matrix"]
    assert len(errors) == 1
    assert errors[0].startswith("crash: 模型子进程退出代码 -1073741819")


def test_process_parallel_limit_override_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("WORDPYCKET_LLM_PROCESS_PARALLEL", "99")
    generator = LocalLlmExampleGenerator(Path("model"))

    assert generator.recommended_process_parallelism() == 8

    monkeypatch.setenv("WORDPYCKET_LLM_PROCESS_PARALLEL", "bad")

    try:
        generator.recommended_process_parallelism()
    except RuntimeError as error:
        assert "WORDPYCKET_LLM_PROCESS_PARALLEL" in str(error)
    else:
        raise AssertionError("invalid override should fail loudly")


def test_cuda_parallelism_allows_two_workers_on_8gb_cards(monkeypatch) -> None:
    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.delenv("WORDPYCKET_LLM_PROCESS_PARALLEL", raising=False)
    monkeypatch.setattr(generator, "_find_existing_model_path", lambda: None)
    monkeypatch.setattr(generator, "_model_size_mb", lambda _path: 4096)
    monkeypatch.setattr(generator, "_system_memory_mb", lambda: 16 * 1024)
    monkeypatch.setattr(generator, "_detect_accelerator", lambda: generator._CUDA_DEVICE)
    monkeypatch.setattr(generator, "_cuda_free_vram_mb", lambda: 7 * 1024)

    assert generator.recommended_process_parallelism() == 2
