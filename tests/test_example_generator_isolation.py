from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from io import BytesIO
from pathlib import Path

import pytest

from wordpycket.domain.entities import WordEntry
from wordpycket.infrastructure.example_generator import LocalLlmExampleGenerator


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
