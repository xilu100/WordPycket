from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

from wordpycket.application.ports import ExampleGenerator
from wordpycket.domain.entities import WordEntry


ProgressCallback = Callable[[int, int, int], None]
ControlState = Callable[[], str]
SleepCallback = Callable[[float], None]


@dataclass(frozen=True)
class BatchUpdate:
    entry_id: str
    first_value: str
    second_value: str
    third_value: str


@dataclass(frozen=True)
class BatchRunResult:
    updates: list[BatchUpdate]
    errors: list[str]
    total: int


def initial_batch_parallel_limit() -> int:
    raw_value = os.getenv("WORDPYCKET_LLM_PROCESS_PARALLEL")
    try:
        return max(1, min(8, int(raw_value))) if raw_value is not None else 2
    except ValueError:
        return 2


class AiBatchRunner:
    def __init__(
        self,
        action: str,
        entries: list[WordEntry],
        scope: str,
        generator: ExampleGenerator,
        control: ControlState,
        parallel_limit: int | None = None,
        sleep: SleepCallback = time.sleep,
    ) -> None:
        self._action = action
        self._entries = entries
        self._scope = scope
        self._generator = generator
        self._control = control
        self._parallel_limit = parallel_limit
        self._sleep = sleep

    def run(self, progress: ProgressCallback) -> BatchRunResult:
        updates: list[BatchUpdate] = []
        if self._action == "补充":
            results, errors, _workers = self.generate(progress)
            for entry, generated in results:
                updates.append(
                    BatchUpdate(
                        entry.id,
                        generated.example_sentence,
                        generated.example_sentence_cn,
                        generated.meaning,
                    )
                )
        else:
            results, errors, _workers = self.correct(progress)
            for entry, corrected in results:
                updates.append(
                    BatchUpdate(
                        entry.id,
                        corrected.corrected_word,
                        entry.meaning,
                        entry.forms,
                    )
                )
        return BatchRunResult(updates, errors, len(self._entries))

    def generate(self, progress: ProgressCallback):
        if hasattr(self._generator, "generate_isolated"):
            return self._run_bounded("generate_isolated", progress)
        return self._run_bounded("generate", progress)

    def correct(self, progress: ProgressCallback):
        if hasattr(self._generator, "correct_entry_isolated"):
            return self._run_bounded("correct_entry_isolated", progress)
        return self._run_bounded("correct_entry", progress)

    def _run_bounded(self, method_name: str, progress: ProgressCallback):
        if not self._entries:
            return [], [], 0

        results = []
        errors = []
        total = len(self._entries)
        worker_count = min(total, self._parallel_limit or initial_batch_parallel_limit())
        next_index = 0
        completed = 0
        active = set()

        def run_entry(entry: WordEntry):
            method = getattr(self._generator, method_name)
            try:
                return entry, method(entry, self._scope)
            except Exception as error:
                raise RuntimeError(f"{entry.word}: {error}") from error

        def submit_next(executor: ThreadPoolExecutor) -> bool:
            nonlocal next_index
            if next_index >= total:
                return False
            if not self._wait_for_resume():
                return False
            entry = self._entries[next_index]
            next_index += 1
            active.add(executor.submit(run_entry, entry))
            return True

        progress(0, total, worker_count)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for _ in range(worker_count):
                if not submit_next(executor):
                    break
            while active:
                finished, active = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    completed += 1
                    try:
                        results.append(future.result())
                    except Exception as error:
                        errors.append(str(error))
                    progress(completed, total, worker_count)
                while len(active) < worker_count and submit_next(executor):
                    pass
        return results, errors, worker_count

    def _wait_for_resume(self) -> bool:
        while self._control() == "paused":
            self._sleep(0.2)
        return self._control() != "stopped"
