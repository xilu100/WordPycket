from __future__ import annotations

from dataclasses import dataclass

from wordpycket.application.ports import ExampleGenerator
from wordpycket.domain.entities import WordEntry


@dataclass(frozen=True)
class ExplainProgress:
    message: str


@dataclass(frozen=True)
class ExplainCompleted:
    entry: WordEntry | None
    explanation: str


@dataclass(frozen=True)
class ExplainFailed:
    message: str


@dataclass(frozen=True)
class BatchJobCompleted:
    job_id: str
    entry: WordEntry
    result: dict | None = None
    error: str = ""


ExplainEvent = ExplainProgress | ExplainCompleted | ExplainFailed


class LlmJobPoller:
    def __init__(self, generator: ExampleGenerator | None) -> None:
        self._generator = generator
        self._explain_job_id: str | None = None
        self._explain_entry: WordEntry | None = None
        self._batch_jobs: dict[str, WordEntry] = {}

    @property
    def explain_entry(self) -> WordEntry | None:
        return self._explain_entry

    @property
    def batch_job_count(self) -> int:
        return len(self._batch_jobs)

    def can_submit_jobs(self) -> bool:
        return (
            self._generator is not None
            and hasattr(self._generator, "submit_job")
            and hasattr(self._generator, "job_status")
        )

    def has_explain_job(self) -> bool:
        return self._explain_job_id is not None

    def has_batch_jobs(self) -> bool:
        return bool(self._batch_jobs)

    def is_idle(self) -> bool:
        return self._explain_job_id is None and not self._batch_jobs

    def submit_explain(self, entry: WordEntry, scope: str, language: str) -> None:
        generator = self._require_generator()
        self._explain_job_id = generator.submit_job(
            "run_action",
            {
                "action": "explain",
                "entry": self.entry_payload(entry),
                "scope": scope,
                "language": language,
            },
        )
        self._explain_entry = entry

    def finish_explain(self) -> None:
        self._explain_job_id = None
        self._explain_entry = None

    def submit_batch_job(self, action: str, entry: WordEntry, scope: str) -> str:
        generator = self._require_generator()
        llm_action = "generate" if action == "补充" else "correct"
        job_id = generator.submit_job(
            "run_action",
            {
                "action": llm_action,
                "entry": self.entry_payload(entry),
                "scope": scope,
                "language": "",
            },
        )
        self._batch_jobs[job_id] = entry
        return job_id

    def clear_batch_jobs(self) -> None:
        self._batch_jobs = {}

    def poll_explain(self) -> ExplainEvent | None:
        if self._generator is None or self._explain_job_id is None:
            return None
        try:
            status = self._generator.job_status(self._explain_job_id)
        except Exception as error:
            self.finish_explain()
            return ExplainFailed(str(error))

        state = str(status.get("state", ""))
        if state in {"queued", "running"}:
            progress = status.get("progress", {})
            message = str(progress.get("message", "解释中")) if isinstance(progress, dict) else "解释中"
            return ExplainProgress(message)
        if state == "failed":
            error = str(status.get("error", "AI解释失败。"))
            self.finish_explain()
            return ExplainFailed(error)
        if state != "completed":
            return None

        result = status.get("result", {})
        entry = self._explain_entry
        self.finish_explain()
        try:
            if not isinstance(result, dict):
                raise RuntimeError(f"LLM 结果不是 JSON 对象：{result}")
            explanation = str(result["explanation"]).strip()
        except Exception as error:
            return ExplainFailed(str(error))
        return ExplainCompleted(entry, explanation)

    def poll_batch(self) -> list[BatchJobCompleted]:
        if self._generator is None or not self._batch_jobs:
            return []

        completed: list[BatchJobCompleted] = []
        for job_id, entry in list(self._batch_jobs.items()):
            try:
                status = self._generator.job_status(job_id)
            except Exception as error:
                completed.append(self._pop_batch_job(job_id, entry, error=str(error)))
                continue
            state = str(status.get("state", ""))
            if state in {"queued", "running"}:
                continue
            if state == "failed":
                completed.append(self._pop_batch_job(job_id, entry, error=str(status.get("error", "LLM 任务失败。"))))
                continue
            if state == "completed":
                result = status.get("result", {})
                if isinstance(result, dict):
                    completed.append(self._pop_batch_job(job_id, entry, result=result))
                else:
                    completed.append(self._pop_batch_job(job_id, entry, error=f"LLM 结果不是 JSON 对象：{result}"))
        return completed

    def _pop_batch_job(
        self,
        job_id: str,
        entry: WordEntry,
        result: dict | None = None,
        error: str = "",
    ) -> BatchJobCompleted:
        self._batch_jobs.pop(job_id, None)
        return BatchJobCompleted(job_id, entry, result=result, error=error)

    def _require_generator(self) -> ExampleGenerator:
        if self._generator is None:
            raise RuntimeError("生成器已不可用。")
        return self._generator

    @staticmethod
    def entry_payload(entry: WordEntry) -> dict:
        return {
            "word": entry.word,
            "meaning": entry.meaning,
            "source_index": entry.source_index,
            "frequency": entry.frequency,
            "forms": entry.forms,
            "example_sentence": entry.example_sentence,
            "example_sentence_cn": entry.example_sentence_cn,
        }
