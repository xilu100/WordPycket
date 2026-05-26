from __future__ import annotations

from wordpycket.domain.entities import WordEntry
from wordpycket.presentation.llm_jobs import ExplainCompleted, ExplainFailed, ExplainProgress, LlmJobPoller


class FakeGenerator:
    def __init__(self, statuses: list[dict] | None = None) -> None:
        self.statuses = statuses or []
        self.submissions: list[tuple[str, dict]] = []

    def submit_job(self, method: str, params: dict) -> str:
        self.submissions.append((method, params))
        return f"job-{len(self.submissions)}"

    def job_status(self, _job_id: str) -> dict:
        return self.statuses.pop(0)


def test_llm_job_poller_submits_explain_without_qt_state() -> None:
    generator = FakeGenerator()
    poller = LlmJobPoller(generator)
    entry = WordEntry(word="vector", meaning="向量")

    poller.submit_explain(entry, "AI", "英语")

    assert poller.has_explain_job()
    assert generator.submissions == [
        (
            "run_action",
            {
                "action": "explain",
                "entry": {
                    "word": "vector",
                    "meaning": "向量",
                    "source_index": 0,
                    "frequency": 0,
                    "forms": "",
                    "example_sentence": "",
                    "example_sentence_cn": "",
                },
                "scope": "AI",
                "language": "英语",
            },
        )
    ]


def test_llm_job_poller_rejects_duplicate_explain_job() -> None:
    generator = FakeGenerator()
    poller = LlmJobPoller(generator)
    entry = WordEntry(word="vector", meaning="向量")

    poller.submit_explain(entry, "AI", "英语")

    try:
        poller.submit_explain(entry, "AI", "英语")
    except RuntimeError as error:
        assert "还在解释中" in str(error)
    else:
        raise AssertionError("duplicate explain job should fail")

    assert len(generator.submissions) == 1


def test_llm_job_poller_normalizes_explain_statuses() -> None:
    entry = WordEntry(word="vector", meaning="向量")
    generator = FakeGenerator(
        [
            {"state": "running", "progress": {"message": "解释中", "percent": 50}},
            {"state": "completed", "result": {"explanation": "向量解释"}},
        ]
    )
    poller = LlmJobPoller(generator)
    poller.submit_explain(entry, "", "")

    first = poller.poll_explain()
    second = poller.poll_explain()

    assert first == ExplainProgress("解释中")
    assert second == ExplainCompleted(entry, "向量解释")
    assert poller.is_idle()


def test_llm_job_poller_formats_explain_section_object() -> None:
    entry = WordEntry(word="token", meaning="标记")
    generator = FakeGenerator(
        [
            {
                "state": "completed",
                "result": {
                    "explanation": {
                        "意思": "标记",
                        "常规用法": "表示一个独立单位。",
                        "领域用法": "在 AI 中表示基本处理单元。",
                    }
                },
            }
        ]
    )
    poller = LlmJobPoller(generator)
    poller.submit_explain(entry, "", "")

    event = poller.poll_explain()

    assert event == ExplainCompleted(
        entry,
        "意思：标记\n常规用法：表示一个独立单位。\n领域用法：在 AI 中表示基本处理单元。",
    )


def test_llm_job_poller_returns_failed_event_for_invalid_explain_result() -> None:
    generator = FakeGenerator([{"state": "completed", "result": []}])
    poller = LlmJobPoller(generator)
    poller.submit_explain(WordEntry(word="vector", meaning="向量"), "", "")

    event = poller.poll_explain()

    assert isinstance(event, ExplainFailed)
    assert "AI 返回结果格式不正确" in event.message
    assert poller.is_idle()


def test_llm_job_poller_pops_completed_batch_jobs() -> None:
    generator = FakeGenerator(
        [
            {
                "state": "completed",
                "result": {
                    "example_sentence": "A vector rotates.",
                    "example_sentence_cn": "向量会旋转。",
                },
            }
        ]
    )
    poller = LlmJobPoller(generator)
    entry = WordEntry(word="vector", meaning="向量")
    poller.submit_batch_job("补充", entry, "AI", "德语")

    events = poller.poll_batch()

    assert len(events) == 1
    assert events[0].entry == entry
    assert events[0].result == {
        "example_sentence": "A vector rotates.",
        "example_sentence_cn": "向量会旋转。",
    }
    assert generator.submissions[0][1]["language"] == "德语"
    assert not poller.has_batch_jobs()


def test_llm_job_poller_submits_supplement_batch_job() -> None:
    generator = FakeGenerator()
    poller = LlmJobPoller(generator)
    entries = [
        WordEntry(word="vector", meaning="向量", source_index=1),
        WordEntry(word="matrix", meaning="矩阵", source_index=2),
    ]

    poller.submit_supplement_batch_job(entries, "AI", "德语")

    assert generator.submissions[0][0] == "run_action"
    assert generator.submissions[0][1]["action"] == "generate_batch"
    assert generator.submissions[0][1]["language"] == "德语"
    assert [item["word"] for item in generator.submissions[0][1]["entries"]] == ["vector", "matrix"]
