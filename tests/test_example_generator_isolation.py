from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from io import BytesIO
from pathlib import Path

import pytest

from wordpycket.domain.entities import WordEntry
from llmserver.engine import LocalLlmExampleGenerator


class UrlOpenResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
        return False


def test_isolated_generation_reads_child_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "example_sentence": "Vectors represent direction.",
                    "example_sentence_cn": "向量表示方向。",
                },
                ensure_ascii=False,
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    generator = LocalLlmExampleGenerator(model_dir=__file__)

    result = generator.generate_isolated(WordEntry(word="vector", meaning="向量"))

    assert result.example_sentence == "Vectors represent direction."
    assert result.example_sentence_cn == "向量表示方向。"
    assert result.meaning == ""


def test_isolated_generation_can_fill_empty_chinese_meaning(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = []

    def fake_run(*_args, **kwargs):
        payloads.append(json.loads(kwargs["input"]))
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "example_sentence": "A kernel manages system resources.",
                    "example_sentence_cn": "内核管理系统资源。",
                    "meaning": "内核",
                },
                ensure_ascii=False,
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    generator = LocalLlmExampleGenerator(model_dir=__file__)

    result = generator.generate_isolated(WordEntry(word="kernel", meaning=""))

    assert payloads[0]["entry"]["meaning"] == ""
    assert result.example_sentence == "A kernel manages system resources."
    assert result.example_sentence_cn == "内核管理系统资源。"
    assert result.meaning == "内核"


def test_isolated_explanation_reads_child_json(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = []

    def fake_run(*_args, **_kwargs):
        payloads.append(json.loads(_kwargs["input"]))
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"explanation": "vector 表示有大小和方向的量。"}, ensure_ascii=False),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    generator = LocalLlmExampleGenerator(model_dir=__file__)

    result = generator.explain_entry_isolated(WordEntry(word="vector", meaning="向量"), language="英语")

    assert payloads[0]["language"] == "英语"
    assert result.explanation == "vector 表示有大小和方向的量。"


def test_explanation_prompt_targets_vocabulary_language_not_chinese_translation() -> None:
    prompt = LocalLlmExampleGenerator._build_explanation_prompt(
        WordEntry(word="Entlassung", meaning="出院；解雇"),
        "医学",
        "德语",
    )

    assert "Target vocabulary language: 德语" in prompt
    assert "exactly three labeled sections: 意思, 常规用法, 领域用法" in prompt
    assert "In 意思, give the meaning of the target word or phrase itself" in prompt
    assert "In 常规用法, explain ordinary target-language usage" in prompt
    assert "In 领域用法, explain how the target word or phrase is used in the required domain/scope" in prompt
    assert "Required domain for domain-specific usage: 医学" in prompt
    assert "Do not explain how the Chinese translation is used in Chinese" in prompt
    assert "only a reference to disambiguate the vocabulary item" in prompt


def test_generation_requires_chinese_meaning_only_when_entry_meaning_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLlm:
        def create_chat_completion(self, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "example_sentence": "A kernel manages resources.",
                                    "example_sentence_cn": "内核管理资源。",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: FakeLlm())

    with pytest.raises(RuntimeError, match="中文释义"):
        generator.generate(WordEntry(word="kernel", meaning=""))


def test_isolated_generation_reports_child_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=-1073741819,
            stdout="",
            stderr="native crash",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    generator = LocalLlmExampleGenerator(model_dir=__file__)

    with pytest.raises(RuntimeError, match="模型子进程退出代码 -1073741819"):
        generator.generate_isolated(WordEntry(word="matrix", meaning="矩阵"))


def test_isolated_generation_accepts_result_when_child_crashes_during_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=-1073741819,
            stdout=json.dumps(
                {
                    "example_sentence": "Matrices organize values.",
                    "example_sentence_cn": "矩阵组织数值。",
                },
                ensure_ascii=False,
            ),
            stderr="native cleanup crash",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    generator = LocalLlmExampleGenerator(model_dir=__file__)

    result = generator.generate_isolated(WordEntry(word="matrix", meaning="矩阵"))

    assert result.example_sentence == "Matrices organize values."
    assert result.example_sentence_cn == "矩阵组织数值。"


def test_default_model_downloads_when_model_dir_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: UrlOpenResponse(b"gguf model bytes"),
    )
    generator = LocalLlmExampleGenerator(tmp_path)

    model_path = generator._ensure_model_path()

    assert model_path.name == LocalLlmExampleGenerator.DEFAULT_MODEL_FILENAME
    assert model_path.read_bytes() == b"gguf model bytes"
    assert not model_path.with_suffix(model_path.suffix + ".part").exists()


def test_ensure_model_available_reports_download_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: UrlOpenResponse(b"gguf model bytes"),
    )
    generator = LocalLlmExampleGenerator(tmp_path)

    status = generator.ensure_model_available()

    assert status.downloaded
    assert not status.is_user_model
    assert status.path is not None
    assert status.path.name == LocalLlmExampleGenerator.DEFAULT_MODEL_FILENAME


def test_user_model_is_detected_without_downloading(tmp_path: Path) -> None:
    custom_model = tmp_path / "custom-vocab-model.gguf"
    custom_model.write_bytes(b"custom")
    generator = LocalLlmExampleGenerator(tmp_path)

    assert generator.uses_user_model()
    status = generator.model_status()
    assert status.path == custom_model
    assert status.is_user_model
    assert generator._ensure_model_path() == custom_model


def test_model_dir_rejects_multiple_gguf_files(tmp_path: Path) -> None:
    (tmp_path / "first.gguf").write_bytes(b"first")
    (tmp_path / "second.gguf").write_bytes(b"second")
    generator = LocalLlmExampleGenerator(tmp_path)

    with pytest.raises(RuntimeError, match="只能存在一个 .gguf"):
        generator._ensure_model_path()


def test_device_status_selects_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(llama_supports_gpu_offload=lambda: True),
    )
    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_detect_accelerator", lambda: generator._CUDA_DEVICE)

    status = generator.device_status()

    assert status.detected == "cuda"
    assert status.selected == "cuda"
    assert status.gpu_offload_supported is True
    assert status.error == ""


def test_device_status_falls_back_to_cpu_without_gpu_offload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(llama_supports_gpu_offload=lambda: False),
    )
    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_detect_accelerator", lambda: generator._CUDA_DEVICE)

    status = generator.device_status()

    assert status.detected == "cuda"
    assert status.selected == "cpu"
    assert status.gpu_offload_supported is False
    assert status.error == ""


def test_smoke_test_uses_isolated_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = []

    def fake_run(*_args, **kwargs):
        payloads.append(json.loads(kwargs["input"]))
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"ok": True}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    generator = LocalLlmExampleGenerator(Path("model"))

    generator.run_smoke_test_isolated()

    assert payloads[0]["action"] == "smoke_test"


def test_check_model_runtime_runs_smoke_test(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = LocalLlmExampleGenerator(Path("model"))
    model_path = Path("model") / LocalLlmExampleGenerator.DEFAULT_MODEL_FILENAME
    monkeypatch.setattr(
        generator,
        "ensure_model_available",
        lambda: type("Status", (), {"path": model_path, "is_user_model": False, "downloaded": False})(),
    )
    monkeypatch.setattr(
        generator,
        "device_status",
        lambda: type(
            "Device",
            (),
            {
                "requested": "auto",
                "detected": "cpu",
                "selected": "cpu",
                "gpu_offload_supported": False,
                "error": "",
            },
        )(),
    )
    smoke_calls = []
    monkeypatch.setattr(generator, "run_smoke_test_isolated", lambda: smoke_calls.append(True))

    result = generator.check_model_runtime()

    assert result.model.path == model_path
    assert result.device.selected == "cpu"
    assert result.smoke_test_passed is True
    assert smoke_calls == [True]


def test_pdf_vocabulary_cleaner_removes_model_selected_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLlm:
        def create_chat_completion(self, **kwargs):
            prompt = kwargs["messages"][1]["content"]
            assert "vocabulary CSV rows" in kwargs["messages"][0]["content"]
            assert "john" in prompt
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"remove_source_indexes": [2]},
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: FakeLlm())
    entries = [
        WordEntry(source_index=1, word="machine learning", meaning="", frequency=4),
        WordEntry(source_index=2, word="john", meaning="", frequency=2),
        WordEntry(source_index=3, word="translation", meaning="", frequency=1),
    ]

    cleaned = generator.clean_pdf_vocabulary_entries(entries, "英语")

    assert [entry.word for entry in cleaned] == ["machine learning", "translation"]
    assert [entry.source_index for entry in cleaned] == [1, 2]


def test_pdf_vocabulary_cleaner_keeps_rows_when_one_batch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = LocalLlmExampleGenerator(Path("model"))
    entries = [
        WordEntry(source_index=1, word="vector", meaning="", frequency=4),
        WordEntry(source_index=2, word="noise", meaning="", frequency=2),
        WordEntry(source_index=3, word="matrix", meaning="", frequency=1),
    ]
    monkeypatch.setattr(generator, "_parallel_workers", lambda: 2)
    monkeypatch.setattr(generator, "_pdf_clean_batches", lambda _entries: [entries[:2], entries[2:]])

    def clean_batch(_slots, batch, _language):
        if batch[0].word == "vector":
            raise RuntimeError("simulated LLM failure")
        return set()

    monkeypatch.setattr(generator, "_clean_pdf_batch_with_slot", clean_batch)
    progress = []

    cleaned = generator.clean_pdf_vocabulary_entries(
        entries,
        "英语",
        progress_callback=lambda message, percent: progress.append((message, percent)),
    )

    assert [entry.word for entry in cleaned] == ["vector", "noise", "matrix"]
    assert progress[-1] == ("AI 审阅 CSV 完成，1 批失败并已保留", 80)


def test_pdf_vocabulary_cleaner_accepts_batch_row_numbers() -> None:
    batch = [
        WordEntry(source_index=101, word="machine learning", meaning="", frequency=4),
        WordEntry(source_index=102, word="john", meaning="", frequency=2),
        WordEntry(source_index=103, word="translation", meaning="", frequency=1),
    ]

    indexes = LocalLlmExampleGenerator._parse_pdf_vocabulary_cleaning_response(
        json.dumps({"remove_csv_indexes": [2]}),
        batch,
    )

    assert indexes == {102}


def test_pdf_vocabulary_cleaner_tolerates_non_json_response() -> None:
    batch = [
        WordEntry(source_index=101, word="machine learning", meaning="", frequency=4),
        WordEntry(source_index=102, word="john", meaning="", frequency=2),
        WordEntry(source_index=103, word="translation", meaning="", frequency=1),
    ]

    indexes = LocalLlmExampleGenerator._parse_pdf_vocabulary_cleaning_response(
        "Remove row 2.",
        batch,
    )

    assert indexes == {102}


def test_llm_auto_tuning_scales_with_available_cuda_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORDPYCKET_LLM_CONTEXT", raising=False)
    monkeypatch.delenv("WORDPYCKET_PDF_CLEAN_BATCH", raising=False)
    monkeypatch.delenv("WORDPYCKET_PDF_CLEAN_CHARS", raising=False)
    monkeypatch.setattr(
        LocalLlmExampleGenerator,
        "_detect_accelerator",
        classmethod(lambda cls: cls._CUDA_DEVICE),
    )
    monkeypatch.setattr(LocalLlmExampleGenerator, "_cuda_capacity_mb", classmethod(lambda cls: 12000))
    monkeypatch.setattr(LocalLlmExampleGenerator, "_system_capacity_mb", classmethod(lambda cls: 32000))

    assert LocalLlmExampleGenerator._context_size() == 16384
    assert LocalLlmExampleGenerator._pdf_clean_batch_size() == 100
    assert LocalLlmExampleGenerator._pdf_clean_prompt_char_budget() == 10000


def test_pdf_vocabulary_cleaning_prompt_keeps_eponyms() -> None:
    prompt = LocalLlmExampleGenerator._build_pdf_vocabulary_cleaning_prompt(
        [WordEntry(source_index=1, word="fourier", meaning="", frequency=3)],
        "英语",
    )

    assert "Keep eponyms" in prompt
    assert "Fourier" in prompt


def test_pdf_vocabulary_cleaning_prompt_rejects_code_and_nonwords() -> None:
    prompt = LocalLlmExampleGenerator._build_pdf_vocabulary_cleaning_prompt(
        [
            WordEntry(source_index=1, word="end if", meaning="", frequency=2),
            WordEntry(source_index=2, word="xy", meaning="", frequency=4),
        ],
        "英语",
    )

    assert "real learnable word" in prompt
    assert "end if" in prompt
    assert "xy" in prompt
