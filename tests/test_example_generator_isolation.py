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
from llmserver import prompts
from wordpycket.infrastructure.example_generator import LocalLlmExampleGenerator as ClientLlmExampleGenerator


class UrlOpenResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
        return False


def test_default_model_points_to_current_coder_7b_model() -> None:
    assert LocalLlmExampleGenerator.DEFAULT_MODEL_REPO == "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
    assert LocalLlmExampleGenerator.DEFAULT_MODEL_FILENAME == "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    assert (
        LocalLlmExampleGenerator.DEFAULT_MODEL_URL
        == "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/"
        "qwen2.5-coder-7b-instruct-q4_k_m.gguf?download=true"
    )
    assert ClientLlmExampleGenerator.DEFAULT_MODEL_REPO == LocalLlmExampleGenerator.DEFAULT_MODEL_REPO
    assert ClientLlmExampleGenerator.DEFAULT_MODEL_FILENAME == LocalLlmExampleGenerator.DEFAULT_MODEL_FILENAME
    assert ClientLlmExampleGenerator.DEFAULT_MODEL_URL == LocalLlmExampleGenerator.DEFAULT_MODEL_URL


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
    assert "In 意思, give the meaning of the German word or phrase itself" in prompt
    assert "In 常规用法, write the explanation in Chinese" in prompt
    assert "explicitly explain the German usage of the target word or phrase" in prompt
    assert "common German collocations" in prompt
    assert "In 领域用法, write the explanation in Chinese" in prompt
    assert "common German technical collocations or term patterns" in prompt
    assert "Required domain for domain-specific usage: 医学" in prompt
    assert "Do not explain how the Chinese translation is used in Chinese" in prompt
    assert "both sections must mention German usage" in prompt
    assert "only a reference to disambiguate the vocabulary item" in prompt
    assert "English usage" not in prompt
    assert "common English" not in prompt


def test_german_generation_prompt_is_separate_from_english_prompt() -> None:
    prompt = LocalLlmExampleGenerator._build_prompt(
        WordEntry(word="Vektor", meaning="向量", forms="Vektor;Vektoren"),
        "Mathematik",
        "德语",
    )

    assert "Generate one natural, short German sentence" in prompt
    assert "German word: Vektor" in prompt
    assert "German word forms: Vektor;Vektoren" in prompt
    assert "The German sentence must include" in prompt
    assert "English sentence" not in prompt
    assert "Original English" not in prompt


def test_german_batch_prompt_is_separate_from_english_prompt() -> None:
    prompt = LocalLlmExampleGenerator._build_batch_prompt(
        [WordEntry(word="Dienst", meaning="服务")],
        "Informatik",
        "德语",
    )

    assert "Generate one natural, short German sentence for each vocabulary item" in prompt
    assert "German vocabulary items JSON" in prompt
    assert "must not leave German words untranslated" in prompt
    assert "English sentence" not in prompt


def test_german_correction_prompt_is_separate_from_english_prompt() -> None:
    prompt = LocalLlmExampleGenerator._build_correction_prompt(
        WordEntry(word="maschinelles_Lernen", meaning="机器学习", forms="maschinelles Lernen"),
        "KI",
        "德语",
    )

    assert "Original German: maschinelles_Lernen" in prompt
    assert "Normalized German candidate: maschinelles Lernen" in prompt
    assert "whether the German word/phrase matches the Chinese meaning" in prompt
    assert "Original English" not in prompt
    assert "English word/phrase" not in prompt


def test_explanation_uses_larger_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class FakeLlm:
        def create_chat_completion(self, **kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "explanation": "意思：向量。\n常规用法：表示有方向和大小的量。\n领域用法：用于嵌入表示。"
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: FakeLlm())

    result = generator.explain_entry(WordEntry(word="vector", meaning="向量"), language="英语")

    assert result.explanation.startswith("意思：")
    assert calls[0]["max_tokens"] == 768


def test_explanation_parser_formats_section_object(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLlm:
        def create_chat_completion(self, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "explanation": {
                                        "意思": "标记",
                                        "常规用法": "表示一个独立单位。",
                                        "领域用法": "在 AI 中表示文本或图像的基本处理单元。",
                                    }
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: FakeLlm())

    result = generator.explain_entry(WordEntry(word="token", meaning="标记"), language="英语")

    assert result.explanation == (
        "意思：标记\n"
        "常规用法：表示一个独立单位。\n"
        "领域用法：在 AI 中表示文本或图像的基本处理单元。"
    )
    assert "{" not in result.explanation


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


def test_generation_requires_chinese_meaning_when_existing_meaning_is_not_chinese(
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
                                    "example_sentence": "The gradient changes quickly.",
                                    "example_sentence_cn": "梯度变化很快。",
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
        generator.generate(WordEntry(word="gradient", meaning="Gradient"))


def test_generation_fills_chinese_meaning_when_existing_meaning_is_not_chinese(
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
                                    "example_sentence": "The gradient changes quickly.",
                                    "example_sentence_cn": "梯度变化很快。",
                                    "meaning": "梯度",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: FakeLlm())

    result = generator.generate(WordEntry(word="gradient", meaning="Gradient"))

    assert result.meaning == "梯度"


def test_llama_create_failure_preserves_original_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenLlama:
        def __init__(self, **_kwargs):
            raise OSError("[WinError -1073741795] Windows Error 0xc000001d")

    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_ensure_model_path", lambda: Path("model/model.gguf"))
    monkeypatch.setattr(generator, "_select_device", lambda _supports_gpu: generator._CPU_DEVICE)
    monkeypatch.setattr(generator, "_context_size", lambda _model_path: 2048)
    monkeypatch.setattr(generator, "_threads_per_model", lambda: 1)
    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(Llama=BrokenLlama, llama_supports_gpu_offload=lambda: True),
    )

    with pytest.raises(RuntimeError, match="CPU 指令集不兼容"):
        generator.generate(WordEntry(word="vector", meaning="向量"))


def test_batch_generation_parses_results_by_batch_index(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLlm:
        def create_chat_completion(self, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                [
                                    {
                                        "batch_index": 2,
                                        "example_sentence": "A matrix stores values.",
                                        "example_sentence_cn": "矩阵存储数值。",
                                    },
                                    {
                                        "batch_index": 1,
                                        "example_sentence": "A vector has direction.",
                                        "example_sentence_cn": "向量有方向。",
                                    },
                                ],
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: FakeLlm())

    results = generator.generate_batch(
        [
            WordEntry(word="vector", meaning="向量"),
            WordEntry(word="matrix", meaning="矩阵"),
        ]
    )

    assert [entry.word for entry, _example in results] == ["vector", "matrix"]
    assert results[0][1].example_sentence == "A vector has direction."
    assert results[1][1].example_sentence == "A matrix stores values."


def test_batch_generation_accepts_single_object_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLlm:
        def create_chat_completion(self, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "batch_index": 1,
                                    "example_sentence": "A vector has direction.",
                                    "example_sentence_cn": "向量有方向。",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: FakeLlm())

    results = generator.generate_batch([WordEntry(word="vector", meaning="向量")])

    assert len(results) == 1
    assert results[0][1].example_sentence == "A vector has direction."


def test_batch_generation_accepts_wrapped_items_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLlm:
        def create_chat_completion(self, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "batch_index": 1,
                                            "example_sentence": "A vector has direction.",
                                            "example_sentence_cn": "向量有方向。",
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: FakeLlm())

    results = generator.generate_batch([WordEntry(word="vector", meaning="向量")])

    assert len(results) == 1
    assert results[0][1].example_sentence_cn == "向量有方向。"


def test_batch_generation_splits_failed_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLlm:
        def __init__(self) -> None:
            self.calls = 0

        def create_chat_completion(self, **kwargs):
            self.calls += 1
            prompt = kwargs["messages"][1]["content"]
            if self.calls == 1:
                return {"choices": [{"message": {"content": '{"items": ['}}]}
            if "vector" in prompt:
                content = {
                    "items": [
                        {
                            "batch_index": 1,
                            "example_sentence": "A vector has direction.",
                            "example_sentence_cn": "向量有方向。",
                        }
                    ]
                }
            else:
                content = {
                    "items": [
                        {
                            "batch_index": 1,
                            "example_sentence": "A matrix stores values.",
                            "example_sentence_cn": "矩阵存储数值。",
                        }
                    ]
                }
            return {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]}

    llm = FakeLlm()
    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: llm)

    results = generator.generate_batch(
        [
            WordEntry(word="vector", meaning="向量"),
            WordEntry(word="matrix", meaning="矩阵"),
        ]
    )

    assert [entry.word for entry, _example in results] == ["vector", "matrix"]
    assert llm.calls == 3


def test_batch_generation_rejects_swapped_language_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLlm:
        def create_chat_completion(self, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "batch_index": 1,
                                            "example_sentence": "计算图像块的梯度。",
                                            "example_sentence_cn": "计算图像块的梯度。",
                                            "meaning": "Gradient",
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: FakeLlm())

    with pytest.raises(RuntimeError, match="英文例句不是英文"):
        generator.generate_batch([WordEntry(word="gradient", meaning="")])


def test_batch_generation_rejects_example_for_different_word(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLlm:
        def create_chat_completion(self, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "batch_index": 1,
                                            "example_sentence": "The aggregate function is stable.",
                                            "example_sentence_cn": "聚合函数很稳定。",
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: FakeLlm())

    with pytest.raises(RuntimeError, match="未包含目标词 adapter"):
        generator.generate_batch([WordEntry(word="adapter", meaning="适配器")])


def test_batch_generation_accepts_word_form_in_example(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLlm:
        def create_chat_completion(self, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "batch_index": 1,
                                            "example_sentence": "The image is normalized before inference.",
                                            "example_sentence_cn": "图像在推理前会被归一化。",
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: FakeLlm())

    results = generator.generate_batch([WordEntry(word="normalize", meaning="归一化", forms="normalize;normalized")])

    assert results[0][1].example_sentence == "The image is normalized before inference."


def test_batch_generation_accepts_simple_plural_in_example(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLlm:
        def create_chat_completion(self, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "batch_index": 1,
                                            "example_sentence": "Tokens are generated from the text.",
                                            "example_sentence_cn": "标记从文本中生成。",
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: FakeLlm())

    results = generator.generate_batch([WordEntry(word="token", meaning="标记")])

    assert results[0][1].example_sentence == "Tokens are generated from the text."


def test_batch_generation_rejects_duplicate_batch_index(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLlm:
        def create_chat_completion(self, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "batch_index": 1,
                                            "example_sentence": "Architecture shapes systems.",
                                            "example_sentence_cn": "架构塑造系统。",
                                        },
                                        {
                                            "batch_index": 1,
                                            "example_sentence": "A domain defines context.",
                                            "example_sentence_cn": "领域定义上下文。",
                                        },
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: FakeLlm())

    with pytest.raises(RuntimeError, match="重复 batch_index"):
        generator.generate_batch(
            [
                WordEntry(word="architecture", meaning=""),
                WordEntry(word="domain", meaning=""),
            ]
        )


def test_generation_retries_when_model_puts_chinese_in_target_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLlm:
        def __init__(self) -> None:
            self.calls = 0

        def create_chat_completion(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                content = {
                    "example_sentence": "向量有方向。",
                    "example_sentence_cn": "向量有方向。",
                }
            else:
                prompt = kwargs["messages"][1]["content"]
                assert "previous JSON response was invalid" in prompt
                content = {
                    "example_sentence": "A vector has direction.",
                    "example_sentence_cn": "向量有方向。",
                }
            return {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]}

    llm = FakeLlm()
    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: llm)

    result = generator.generate(WordEntry(word="vector", meaning="向量"))

    assert result.example_sentence == "A vector has direction."
    assert llm.calls == 2


def test_batch_generation_retries_when_model_puts_chinese_in_target_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLlm:
        def __init__(self) -> None:
            self.calls = 0

        def create_chat_completion(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                content = {
                    "items": [
                        {
                            "batch_index": 1,
                            "example_sentence": "服务很重要。",
                            "example_sentence_cn": "服务很重要。",
                        }
                    ]
                }
            else:
                prompt = kwargs["messages"][1]["content"]
                assert "previous JSON response was invalid" in prompt
                assert "example_sentence must be German only" in prompt
                content = {
                    "items": [
                        {
                            "batch_index": 1,
                            "example_sentence": "Der Dienst ist wichtig.",
                            "example_sentence_cn": "这个服务很重要。",
                        }
                    ]
                }
            return {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]}

    llm = FakeLlm()
    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: llm)

    results = generator.generate_batch([WordEntry(word="Dienst", meaning="服务")], language="德语")

    assert results[0][1].example_sentence == "Der Dienst ist wichtig."
    assert llm.calls == 2


def test_generate_many_keeps_progress_argument_position(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class FakeLlm:
        def create_chat_completion(self, **kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "example_sentence": "A vector has direction.",
                                    "example_sentence_cn": "向量有方向。",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    progress_calls = []
    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_load_model", lambda _slot: FakeLlm())
    monkeypatch.setattr(generator, "_parallel_workers", lambda: 1)

    results, errors, workers = generator.generate_many(
        [WordEntry(word="vector", meaning="向量")],
        "AI",
        lambda done, total, worker_count: progress_calls.append((done, total, worker_count)),
        lambda: "running",
    )

    assert errors == []
    assert workers == 1
    assert results[0][1].example_sentence == "A vector has direction."
    assert progress_calls == [(1, 1, 1)]
    assert "Target vocabulary language: 德语" not in calls[0]["messages"][1]["content"]


def test_batch_prompt_requires_concise_chinese_meanings() -> None:
    prompt = LocalLlmExampleGenerator._build_batch_prompt(
        [WordEntry(word="service", meaning="")],
        "人工智能",
    )

    assert "short Chinese word or phrase" in prompt
    assert "must not leave English words untranslated" in prompt


def test_correction_prompt_only_checks_word_meaning_match_and_obvious_word_errors() -> None:
    prompt = LocalLlmExampleGenerator._build_correction_prompt(
        WordEntry(word="machine_learning", meaning="机器学习", forms="machine learnings"),
        "AI",
    )

    assert "whether the English word/phrase matches the Chinese meaning" in prompt
    assert "should_update to false" in prompt
    assert "Do not rewrite the word for style" in prompt
    assert "Word forms" in prompt


def test_correction_parse_can_skip_unchanged_words() -> None:
    correction = prompts.parse_correction_response(
        json.dumps({"should_update": False, "corrected_word": "kernel", "note": ""}),
        WordEntry(word="kernel", meaning="内核"),
    )

    assert correction.corrected_word == "kernel"
    assert correction.should_update is False


def test_correction_parse_forces_update_for_underscore_format_error() -> None:
    correction = prompts.parse_correction_response(
        json.dumps({"should_update": False, "corrected_word": "machine_learning", "note": ""}),
        WordEntry(word="machine_learning", meaning="机器学习"),
    )

    assert correction.corrected_word == "machine learning"
    assert correction.should_update is True


def test_small_cuda_card_prefers_single_instance_batch_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.delenv("WORDPYCKET_LLM_BATCH_SIZE", raising=False)
    monkeypatch.setattr(generator, "_find_existing_model_path", lambda: None)
    monkeypatch.setattr(generator, "_model_size_mb", lambda _path: 4466)
    monkeypatch.setattr(generator, "_detect_accelerator", lambda: generator._CUDA_DEVICE)
    monkeypatch.setattr(generator, "_cuda_capacity_mb", lambda: 8192)

    assert generator.recommended_supplement_strategy() == {
        "mode": "batch",
        "parallelism": 1,
        "batch_size": 8,
    }


def test_large_cuda_card_keeps_parallel_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.delenv("WORDPYCKET_LLM_BATCH_SIZE", raising=False)
    monkeypatch.setattr(generator, "_find_existing_model_path", lambda: None)
    monkeypatch.setattr(generator, "_model_size_mb", lambda _path: 4466)
    monkeypatch.setattr(generator, "_system_capacity_mb", lambda: 64 * 1024)
    monkeypatch.setattr(generator, "_detect_accelerator", lambda: generator._CUDA_DEVICE)
    monkeypatch.setattr(generator, "_cuda_capacity_mb", lambda: 24 * 1024)

    strategy = generator.recommended_supplement_strategy()

    assert strategy["mode"] == "parallel"
    assert strategy["batch_size"] == 1
    assert strategy["parallelism"] >= 3


def test_mps_process_parallelism_uses_single_model_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.delenv("WORDPYCKET_LLM_PROCESS_PARALLEL", raising=False)
    monkeypatch.setattr(generator, "_find_existing_model_path", lambda: None)
    monkeypatch.setattr(generator, "_model_size_mb", lambda _path: 4096)
    monkeypatch.setattr(generator, "_system_capacity_mb", lambda: 16 * 1024)
    monkeypatch.setattr(generator, "_detect_accelerator", lambda: generator._MPS_DEVICE)

    assert generator.recommended_process_parallelism() == 1


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

    with pytest.raises(RuntimeError, match="只能保留一个 .gguf"):
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

    assert [entry.word for entry in cleaned] == ["vector", "matrix"]
    assert progress[-1] == ("AI 检查词表完成，1 批失败并已保留", 80)


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


def test_pdf_vocabulary_cleaner_ignores_truncated_json_response() -> None:
    batch = [
        WordEntry(source_index=101, word="machine learning", meaning="", frequency=4),
        WordEntry(source_index=102, word="john", meaning="", frequency=2),
    ]

    indexes = LocalLlmExampleGenerator._parse_pdf_vocabulary_cleaning_response(
        '{"remove_csv_indexes": [101, 102,',
        batch,
    )

    assert indexes == set()


def test_pdf_vocabulary_cleaner_protects_technical_terms_from_model_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = LocalLlmExampleGenerator(Path("model"))
    entries = [
        WordEntry(source_index=1, word="machine learning", meaning="", frequency=4),
        WordEntry(source_index=2, word="vector", meaning="", frequency=3),
        WordEntry(source_index=3, word="name", meaning="", frequency=2),
        WordEntry(source_index=4, word="output", meaning="", frequency=1),
    ]
    monkeypatch.setattr(generator, "_pdf_clean_batches", lambda _entries: [entries])
    monkeypatch.setattr(generator, "_parallel_workers", lambda: 1)
    monkeypatch.setattr(generator, "_clean_pdf_batch_with_slot", lambda _slots, _batch, _language: {1, 2, 3, 4})

    cleaned = generator.clean_pdf_vocabulary_entries(entries, "英语")

    assert [entry.word for entry in cleaned] == ["machine learning", "vector"]


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
