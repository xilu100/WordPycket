from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time

from wordpycket.domain.entities import WordEntry
from pathlib import Path

from llmserver.engine import LocalLlmExampleGenerator as LlmEngine
from llmserver.server import JobStore, LlmRpcHandler
from wordpycket.infrastructure.example_generator import GeneratedExample, LocalLlmExampleGenerator
from wordpycket.presentation.qt_app import WordPycketApp
from wordpycket.presentation.qt_workers import BatchWorker


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


def test_batch_worker_uses_bounded_isolated_generation_for_gui_stability() -> None:
    entries = [
        WordEntry(word="vector", meaning="向量"),
        WordEntry(word="matrix", meaning="矩阵"),
    ]
    generator = GeneratorWithUnsafeParallelPath()
    worker = BatchWorker("补充", entries, "", generator, lambda: "running")

    results, errors, workers = worker._generate(lambda *_args: None)

    assert errors == []
    assert workers == 2
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

    assert workers == 2
    assert {entry.word for entry, _generated in results} == {"vector", "matrix"}
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


def test_process_parallel_recommendation_does_not_start_llm_server(monkeypatch) -> None:
    starts = []

    def fail_if_started(*_args, **_kwargs):
        starts.append(True)
        raise AssertionError("parallel recommendation must not start the LLM service")

    monkeypatch.delenv("WORDPYCKET_LLM_PROCESS_PARALLEL", raising=False)
    monkeypatch.setattr("subprocess.Popen", fail_if_started)
    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_find_existing_model_path", lambda: None)
    monkeypatch.setattr(generator, "_detect_accelerator", lambda: generator._CPU_DEVICE)
    monkeypatch.setattr(generator, "_cuda_capacity_mb", lambda: None)
    monkeypatch.setattr(generator, "_system_capacity_mb", lambda: 16 * 1024)

    assert generator.recommended_process_parallelism() >= 1
    assert starts == []


def test_supplement_strategy_does_not_start_llm_server(monkeypatch) -> None:
    starts = []

    def fail_if_started(*_args, **_kwargs):
        starts.append(True)
        raise AssertionError("supplement strategy must not start the LLM service")

    monkeypatch.delenv("WORDPYCKET_LLM_BATCH_SIZE", raising=False)
    monkeypatch.setattr("subprocess.Popen", fail_if_started)
    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_find_existing_model_path", lambda: None)
    monkeypatch.setattr(generator, "_model_size_mb", lambda _path: 4466)
    monkeypatch.setattr(generator, "_detect_accelerator", lambda: generator._CUDA_DEVICE)
    monkeypatch.setattr(generator, "_cuda_capacity_mb", lambda: 8192)

    assert generator.recommended_supplement_strategy() == {
        "mode": "batch",
        "parallelism": 1,
        "batch_size": 8,
    }
    assert starts == []


def test_client_mps_parallel_recommendation_uses_single_model_instance(monkeypatch) -> None:
    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.delenv("WORDPYCKET_LLM_PROCESS_PARALLEL", raising=False)
    monkeypatch.setattr(generator, "_find_existing_model_path", lambda: None)
    monkeypatch.setattr(generator, "_model_size_mb", lambda _path: 4096)
    monkeypatch.setattr(generator, "_system_capacity_mb", lambda: 16 * 1024)
    monkeypatch.setattr(generator, "_detect_accelerator", lambda: generator._MPS_DEVICE)

    assert generator.recommended_process_parallelism() == 1


def test_isolated_environment_does_not_start_llm_server_on_gui_thread(monkeypatch) -> None:
    starts = []

    def fail_if_started(*_args, **_kwargs):
        starts.append(True)
        raise AssertionError("preparing isolated environment must not start the LLM service")

    monkeypatch.delenv("WORDPYCKET_LLM_SERVER_URL", raising=False)
    monkeypatch.setattr("subprocess.Popen", fail_if_started)
    generator = LocalLlmExampleGenerator(Path("model"))

    env = generator.isolated_environment()

    assert "WORDPYCKET_LLM_SERVER_URL" not in env
    assert starts == []


def test_device_status_precheck_does_not_start_llm_server(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.modules",
        {
            **sys.modules,
            "llama_cpp": type("FakeLlamaCpp", (), {"llama_supports_gpu_offload": staticmethod(lambda: False)}),
        },
    )
    generator = LocalLlmExampleGenerator(Path("model"))
    monkeypatch.setattr(generator, "_has_cuda_device", staticmethod(lambda: False))
    monkeypatch.setattr(generator, "_has_mps_device", staticmethod(lambda: False))
    monkeypatch.setattr(generator, "_rpc", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("device precheck must not use RPC")))

    status = generator.device_status()

    assert status.selected == "cpu"


def test_owned_llm_server_process_is_closed() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self.killed = True

    process = FakeProcess()
    generator = LocalLlmExampleGenerator(Path("model"))
    generator._process = process
    generator._base_url = "http://127.0.0.1:12345"
    generator._server_model_available = True
    generator._owns_process = True

    generator.close()

    assert process.terminated is True
    assert process.killed is False
    assert generator._process is None


def test_parse_isolated_result_accepts_json_result_envelope() -> None:
    generator = LocalLlmExampleGenerator(Path("model"))

    data = generator.parse_isolated_result(
        json.dumps({"type": "result", "data": {"example_sentence": "A vector rotates."}}),
        "",
        0,
    )

    assert data == {"example_sentence": "A vector rotates."}


def test_llm_server_without_ready_json_stops_waiting(monkeypatch) -> None:
    class BlockingStdout:
        def readline(self):
            raise queue.Empty()

    class FakeProcess:
        stdout = BlockingStdout()
        stderr = None

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setenv("WORDPYCKET_LLM_STARTUP_TIMEOUT", "1")
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    generator = LocalLlmExampleGenerator(Path("model"))

    try:
        generator.generate(WordEntry(word="vector", meaning="向量"))
    except RuntimeError as error:
        assert "ready JSON" in str(error)
    else:
        raise AssertionError("LLM client should stop waiting when ready JSON is missing")


def test_llm_client_submits_job_and_polls_status(monkeypatch) -> None:
    generator = LocalLlmExampleGenerator(Path("model"))
    generator._base_url = "http://127.0.0.1:12345"
    generator._server_model_available = True
    calls = []
    statuses = [
        {
            "ok": True,
            "result": {
                "state": "running",
                "stage": "loading_model",
                "progress": {"message": "加载模型", "percent": 10},
            },
        },
        {
            "ok": True,
            "result": {
                "state": "completed",
                "stage": "completed",
                "progress": {"message": "完成", "percent": 100},
                "result": {
                    "example_sentence": "Vectors carry magnitude.",
                    "example_sentence_cn": "向量带有大小。",
                },
            },
        },
    ]

    def fake_request(_base_url, payload):
        calls.append(payload)
        if payload["method"] == "submit_job":
            return {"ok": True, "result": {"job_id": "job-1", "state": "queued"}}
        return statuses.pop(0)

    monkeypatch.setattr(generator, "_request_json", fake_request)
    monkeypatch.setattr(generator, "_poll_interval_seconds", lambda: 0.01)
    progress = []

    result = generator._rpc(
        "generate",
        {"entry": generator._entry_payload(WordEntry("vector", "向量"))},
        lambda message, percent: progress.append((message, percent)),
    )

    assert result["example_sentence"] == "Vectors carry magnitude."
    assert calls[0]["method"] == "submit_job"
    assert calls[1]["method"] == "job_status"
    assert progress == [("加载模型", 10), ("完成", 100)]


def test_llm_job_status_reads_cached_state_while_http_poll_blocks(monkeypatch) -> None:
    generator = LocalLlmExampleGenerator(Path("model"))
    generator._base_url = "http://127.0.0.1:12345"
    generator._server_model_available = True
    started = threading.Event()
    release = threading.Event()

    def blocking_request(_base_url, _payload):
        started.set()
        release.wait(timeout=2)
        return {"ok": True, "result": {"job_id": "remote-1", "state": "queued"}}

    monkeypatch.setattr(generator, "_request_json", blocking_request)

    job_id = generator.submit_job(
        "run_action",
        {
            "action": "generate",
            "entry": generator._entry_payload(WordEntry("vector", "向量")),
            "scope": "",
            "language": "",
        },
    )

    assert started.wait(timeout=1)
    status = generator.job_status(job_id)

    assert status["state"] == "queued"
    release.set()


def test_pdf_cleanup_uses_cached_job_polling(monkeypatch) -> None:
    generator = LocalLlmExampleGenerator(Path("model"))
    calls = []
    statuses = [
        {
            "state": "running",
            "progress": {"message": "AI 审阅 CSV：已完成 1/2", "percent": 60},
        },
        {
            "state": "completed",
            "progress": {"message": "AI 审阅 CSV 完成", "percent": 80},
            "result": [
                {
                    "word": "vector",
                    "meaning": "向量",
                    "source_index": 1,
                    "frequency": 1,
                    "forms": "",
                    "example_sentence": "",
                    "example_sentence_cn": "",
                }
            ],
        },
    ]

    def fake_submit(method, params):
        calls.append((method, params))
        return "local-job"

    monkeypatch.setattr(generator, "submit_job", fake_submit)
    monkeypatch.setattr(generator, "job_status", lambda _job_id: statuses.pop(0))
    monkeypatch.setattr(generator, "_poll_interval_seconds", lambda: 0.01)
    progress = []

    cleaned = generator.clean_pdf_vocabulary_entries(
        [WordEntry(word="vector", meaning="向量", source_index=1, frequency=1)],
        "英语",
        progress_callback=lambda message, percent: progress.append((message, percent)),
    )

    assert calls[0][0] == "clean_pdf_vocabulary_entries"
    assert [entry.word for entry in cleaned] == ["vector"]
    assert progress == [("AI 审阅 CSV：已完成 1/2", 60), ("AI 审阅 CSV 完成", 80)]


def test_local_job_timeout_is_based_on_idle_progress(monkeypatch) -> None:
    generator = LocalLlmExampleGenerator(Path("model"))
    now = 0.0
    statuses = [
        {"state": "running", "progress": {"message": "AI 审阅 CSV：已完成 77/93", "percent": 73}},
        {"state": "running", "progress": {"message": "AI 审阅 CSV：已完成 78/93", "percent": 73}},
        {"state": "completed", "progress": {"message": "AI 审阅 CSV 完成", "percent": 80}, "result": []},
    ]

    def fake_monotonic():
        return now

    def fake_status(_job_id):
        nonlocal now
        now += 250.0
        return statuses.pop(0)

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(generator, "job_status", fake_status)
    monkeypatch.setattr(generator, "_operation_timeout_seconds", lambda: 300)

    result = generator._wait_for_local_job("job")

    assert result == []


def test_cuda_parallelism_allows_two_workers_on_8gb_cards(monkeypatch) -> None:
    generator = LlmEngine(Path("model"))
    monkeypatch.delenv("WORDPYCKET_LLM_PROCESS_PARALLEL", raising=False)
    monkeypatch.setattr(generator, "_find_existing_model_path", lambda: None)
    monkeypatch.setattr(generator, "_model_size_mb", lambda _path: 4096)
    monkeypatch.setattr(generator, "_system_capacity_mb", lambda: 16 * 1024)
    monkeypatch.setattr(generator, "_detect_accelerator", lambda: generator._CUDA_DEVICE)
    monkeypatch.setattr(generator, "_cuda_capacity_mb", lambda: 7 * 1024)

    assert generator.recommended_process_parallelism() == 2


def test_parallelism_estimate_is_cached_for_process_run(monkeypatch) -> None:
    generator = LlmEngine(Path("model"))
    memory_values = [16 * 1024, 4 * 1024]
    monkeypatch.delenv("WORDPYCKET_LLM_PROCESS_PARALLEL", raising=False)
    monkeypatch.setattr(generator, "_find_existing_model_path", lambda: None)
    monkeypatch.setattr(generator, "_model_size_mb", lambda _path: 2048)
    monkeypatch.setattr(generator, "_system_capacity_mb", lambda: memory_values.pop(0))
    monkeypatch.setattr(generator, "_detect_accelerator", lambda: generator._CPU_DEVICE)

    first = generator.recommended_process_parallelism()
    second = generator.recommended_process_parallelism()

    assert second == first
    assert memory_values == [4 * 1024]


def test_gui_initial_batch_parallel_limit_does_not_probe_hardware(monkeypatch) -> None:
    monkeypatch.delenv("WORDPYCKET_LLM_PROCESS_PARALLEL", raising=False)

    assert WordPycketApp._initial_batch_parallel_limit() == 2

    monkeypatch.setenv("WORDPYCKET_LLM_PROCESS_PARALLEL", "99")

    assert WordPycketApp._initial_batch_parallel_limit() == 8


def test_gui_batch_parallel_limit_uses_generator_recommendation(monkeypatch) -> None:
    monkeypatch.delenv("WORDPYCKET_LLM_PROCESS_PARALLEL", raising=False)

    class Generator:
        def recommended_process_parallelism(self) -> int:
            return 4

    app = object.__new__(WordPycketApp)
    app._example_generator = Generator()

    assert app._recommended_batch_parallel_limit() == 4


def test_gui_batch_parallel_limit_falls_back_when_recommendation_fails(monkeypatch) -> None:
    monkeypatch.setenv("WORDPYCKET_LLM_PROCESS_PARALLEL", "3")

    class Generator:
        def recommended_process_parallelism(self) -> int:
            raise RuntimeError("not ready")

    app = object.__new__(WordPycketApp)
    app._example_generator = Generator()

    assert app._recommended_batch_parallel_limit() == 3


def test_gui_uses_supplement_batch_strategy() -> None:
    class Generator:
        def recommended_process_parallelism(self) -> int:
            return 4

        def recommended_supplement_strategy(self) -> dict:
            return {"mode": "batch", "parallelism": 1, "batch_size": 8}

    app = object.__new__(WordPycketApp)
    app._example_generator = Generator()

    assert app._recommended_batch_strategy("补充") == {
        "mode": "batch",
        "parallelism": 1,
        "batch_size": 8,
    }
    assert app._recommended_batch_strategy("修正") == {
        "mode": "parallel",
        "parallelism": 4,
        "batch_size": 1,
    }


def test_supplement_update_replaces_non_chinese_meaning() -> None:
    class Service:
        def __init__(self) -> None:
            self.entry = WordEntry(word="gradient", meaning="Gradient", id="entry-1")

        def get_word(self, _entry_id):
            return self.entry

        def update_text(self, entry_id, word, meaning, forms):
            self.entry = WordEntry(id=entry_id, word=word, meaning=meaning, forms=forms)
            return self.entry

        def update_examples(self, entry_id, example_sentence, example_sentence_cn):
            self.entry = WordEntry(
                id=entry_id,
                word=self.entry.word,
                meaning=self.entry.meaning,
                forms=self.entry.forms,
                example_sentence=example_sentence,
                example_sentence_cn=example_sentence_cn,
            )
            return self.entry

    service = Service()
    app = object.__new__(WordPycketApp)
    app._service = service

    updated = app._update_supplemented_entry("entry-1", "The gradient is calculated.", "梯度被计算。", "梯度")

    assert updated.meaning == "梯度"
    assert updated.example_sentence == "The gradient is calculated."


def test_idle_llm_close_closes_generator_only_when_idle() -> None:
    class Jobs:
        def __init__(self, idle: bool) -> None:
            self.idle = idle

        def is_idle(self) -> bool:
            return self.idle

    class Generator:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    generator = Generator()
    app = object.__new__(WordPycketApp)
    app._llm_jobs = Jobs(True)
    app._example_generator = generator
    app._model_check_thread = None
    app._pdf_import_thread = None

    app._close_idle_llm_server()

    assert generator.closed == 1

    app._llm_jobs = Jobs(False)
    app._close_idle_llm_server()

    assert generator.closed == 1


def test_idle_llm_close_delay_can_be_configured(monkeypatch) -> None:
    monkeypatch.setenv("WORDPYCKET_LLM_IDLE_CLOSE_SECONDS", "2")

    assert WordPycketApp._llm_idle_close_delay_ms() == 2000

    monkeypatch.setenv("WORDPYCKET_LLM_IDLE_CLOSE_SECONDS", "bad")

    assert WordPycketApp._llm_idle_close_delay_ms() == 60000


def test_idle_llm_close_timer_cancels_during_next_job_and_reschedules_afterwards() -> None:
    class Timer:
        def __init__(self, active: bool = False) -> None:
            self.active = active
            self.starts = 0
            self.stops = 0

        def isActive(self) -> bool:
            return self.active

        def start(self) -> None:
            self.starts += 1
            self.active = True

        def stop(self) -> None:
            self.stops += 1
            self.active = False

    class Jobs:
        def __init__(self) -> None:
            self.idle = False

        def is_idle(self) -> bool:
            return self.idle

    jobs = Jobs()
    idle_timer = Timer()
    poll_timer = Timer(active=True)
    app = object.__new__(WordPycketApp)
    app._llm_jobs = jobs
    app._llm_idle_close_timer = idle_timer
    app._llm_poll_timer = poll_timer
    app._model_check_thread = None
    app._pdf_import_thread = None

    jobs.idle = True
    app._stop_llm_polling_if_idle()

    assert poll_timer.active is False
    assert idle_timer.active is True
    assert idle_timer.starts == 1

    jobs.idle = False
    app._ensure_llm_polling()

    assert idle_timer.active is False
    assert idle_timer.stops == 1
    assert poll_timer.active is True

    jobs.idle = True
    app._stop_llm_polling_if_idle()

    assert idle_timer.active is True
    assert idle_timer.starts == 2


def test_llm_job_store_runs_bounded_jobs_incrementally() -> None:
    store = JobStore(max_workers=2)
    started: queue.Queue[int] = queue.Queue()
    release = threading.Event()

    def target(_progress, slot):
        started.put(slot)
        release.wait(timeout=2)
        return {"slot": slot}

    first = store.submit(target)
    second = store.submit(target)

    first_slot = started.get(timeout=1)
    second_slot = started.get(timeout=1)
    assert {first_slot, second_slot} == {0, 1}
    assert store.status(first)["state"] == "running"
    assert store.status(second)["state"] == "running"

    release.set()
    deadline = threading.Event()
    for _ in range(20):
        if store.status(first)["state"] == "completed" and store.status(second)["state"] == "completed":
            deadline.set()
            break
        threading.Event().wait(0.01)

    assert deadline.is_set()


def test_generate_batch_rpc_always_uses_single_model_slot() -> None:
    calls = []

    class Generator:
        def _generate_batch_with_slot(self, slot, entries, scope):
            calls.append((slot, [entry.word for entry in entries], scope))
            return []

    handler = object.__new__(LlmRpcHandler)
    handler.generator = Generator()

    result = handler._run_action(
        {
            "action": "generate_batch",
            "entries": [
                {
                    "word": "vector",
                    "meaning": "向量",
                    "source_index": 1,
                    "frequency": 1,
                    "forms": "",
                }
            ],
            "scope": "AI",
        },
        slot=1,
    )

    assert result == {"items": []}
    assert calls == [(0, ["vector"], "AI")]
