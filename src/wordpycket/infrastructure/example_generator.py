from __future__ import annotations

import atexit
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import http.client
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wordpycket.domain.entities import WordEntry


ProgressCallback = Any
ControlCallback = Any


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


class LocalLlmExampleGenerator:
    """HTTP client for the local llmserver process."""

    DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
    DEFAULT_MODEL_FILENAME = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    DEFAULT_MODEL_URL = (
        f"https://huggingface.co/{DEFAULT_MODEL_REPO}/resolve/main/"
        f"{DEFAULT_MODEL_FILENAME}?download=true"
    )
    _AUTO_DEVICE = "auto"
    _CPU_DEVICE = "cpu"
    _CUDA_DEVICE = "cuda"
    _MPS_DEVICE = "mps"

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = Path(model_dir)
        self._process: subprocess.Popen[str] | None = None
        self._base_url = os.getenv("WORDPYCKET_LLM_SERVER_URL", "").strip()
        self._server_model_available = bool(self._base_url)
        self._ready_messages: queue.Queue[str] | None = None
        self._local_jobs: dict[str, dict[str, Any]] = {}
        self._local_jobs_lock = threading.Lock()
        self._server_start_lock = threading.Lock()
        self._recommended_process_parallelism: int | None = None
        self._owns_process = False
        atexit.register(self.close)

    def generate(self, entry: WordEntry, scope: str = "", language: str = "") -> GeneratedExample:
        data = self._rpc("generate", {"entry": self._entry_payload(entry), "scope": scope, "language": language})
        return GeneratedExample(
            example_sentence=str(data["example_sentence"]),
            example_sentence_cn=str(data["example_sentence_cn"]),
            meaning=str(data.get("meaning", "")),
        )

    def generate_isolated(self, entry: WordEntry, scope: str = "", language: str = "") -> GeneratedExample:
        return self.generate(entry, scope, language)

    def correct_entry(self, entry: WordEntry, scope: str = "", language: str = "") -> GeneratedCorrection:
        data = self._rpc("correct_entry", {"entry": self._entry_payload(entry), "scope": scope, "language": language})
        return GeneratedCorrection(
            corrected_word=str(data["corrected_word"]),
            note=str(data.get("note", "")),
            should_update=bool(data.get("should_update", False)),
        )

    def correct_entry_isolated(self, entry: WordEntry, scope: str = "", language: str = "") -> GeneratedCorrection:
        return self.correct_entry(entry, scope, language)

    def explain_entry(self, entry: WordEntry, scope: str = "", language: str = "") -> GeneratedExplanation:
        data = self._rpc(
            "explain_entry",
            {"entry": self._entry_payload(entry), "scope": scope, "language": language},
        )
        return GeneratedExplanation(explanation=self._format_explanation(data["explanation"]))

    def explain_entry_isolated(self, entry: WordEntry, scope: str = "", language: str = "") -> GeneratedExplanation:
        return self.explain_entry(entry, scope, language)

    @staticmethod
    def _format_explanation(value) -> str:
        if isinstance(value, dict):
            preferred_labels = ["意思", "常规用法", "领域用法"]
            lines = []
            used_labels = set()
            for label in preferred_labels:
                text = str(value.get(label, "")).strip()
                if text:
                    lines.append(f"{label}：{text}")
                    used_labels.add(label)
            for label, text_value in value.items():
                if label in used_labels:
                    continue
                text = str(text_value).strip()
                if text:
                    lines.append(f"{label}：{text}")
            return "\n".join(lines).strip()
        return str(value).strip()

    def generate_many(
        self,
        entries: list[WordEntry],
        scope: str = "",
        progress: ProgressCallback | None = None,
        control: ControlCallback | None = None,
        language: str = "",
    ) -> tuple[list[tuple[WordEntry, GeneratedExample]], list[str], int]:
        raw_results, errors, workers = self._run_many_jobs(entries, "generate", scope, language, progress, control)
        results = [
            (
                entry,
                GeneratedExample(
                    example_sentence=str(data["example_sentence"]),
                    example_sentence_cn=str(data["example_sentence_cn"]),
                    meaning=str(data.get("meaning", "")),
                ),
            )
            for entry, data in raw_results
        ]
        return results, errors, workers

    def correct_many(
        self,
        entries: list[WordEntry],
        scope: str = "",
        progress: ProgressCallback | None = None,
        control: ControlCallback | None = None,
        language: str = "",
    ) -> tuple[list[tuple[WordEntry, GeneratedCorrection]], list[str], int]:
        raw_results, errors, workers = self._run_many_jobs(entries, "correct", scope, language, progress, control)
        results = [
            (
                entry,
                GeneratedCorrection(
                    corrected_word=str(data["corrected_word"]),
                    note=str(data.get("note", "")),
                    should_update=bool(data.get("should_update", False)),
                ),
            )
            for entry, data in raw_results
        ]
        return results, errors, workers

    def is_available(self) -> bool:
        return self._find_existing_model_path() is not None

    def uses_user_model(self) -> bool:
        model_path = self._find_existing_model_path()
        return model_path is not None and model_path.name != self.DEFAULT_MODEL_FILENAME

    def model_status(self) -> ModelStatus:
        model_path = self._find_existing_model_path()
        return ModelStatus(
            path=model_path,
            is_user_model=model_path is not None and model_path.name != self.DEFAULT_MODEL_FILENAME,
        )

    def ensure_model_available(self) -> ModelStatus:
        status = self._model_status_from_payload(self._rpc("ensure_model_available"))
        self._server_model_available = status.path is not None
        return status

    def device_status(self) -> DeviceStatus:
        requested = os.getenv("WORDPYCKET_LLM_DEVICE", self._AUTO_DEVICE).lower()
        detected = self._detect_accelerator()
        try:
            from llama_cpp import llama_supports_gpu_offload
        except ImportError:
            return DeviceStatus(
                requested=requested,
                detected=detected,
                selected=None,
                gpu_offload_supported=None,
                error="缺少 llama-cpp-python。",
            )

        try:
            supports_gpu_offload = bool(llama_supports_gpu_offload())
            selected = self._select_device(requested, detected, supports_gpu_offload)
        except Exception as error:
            return DeviceStatus(
                requested=requested,
                detected=detected,
                selected=None,
                gpu_offload_supported=None,
                error=str(error),
            )
        return DeviceStatus(
            requested=requested,
            detected=detected,
            selected=selected,
            gpu_offload_supported=supports_gpu_offload,
        )

    def check_model_runtime(self) -> ModelCheckResult:
        data = self._rpc("check_model_runtime")
        result = ModelCheckResult(
            model=self._model_status_from_payload(data["model"]),
            device=self._device_status_from_payload(data["device"]),
            smoke_test_passed=bool(data["smoke_test_passed"]),
        )
        self._server_model_available = result.model.path is not None
        return result

    def run_smoke_test_isolated(self) -> None:
        data = self._rpc(
            "run_action",
            {
                "action": "smoke_test",
                "entry": self._entry_payload(WordEntry(word="test", meaning="测试")),
                "scope": "",
                "language": "",
            },
        )
        if data.get("ok") is not True:
            raise RuntimeError(f"模型最小执行测试返回异常：{data}")

    def recommended_process_parallelism(self) -> int:
        override = os.getenv("WORDPYCKET_LLM_PROCESS_PARALLEL")
        if override is not None:
            try:
                int(override)
            except ValueError as error:
                raise RuntimeError("WORDPYCKET_LLM_PROCESS_PARALLEL 必须是整数。") from error
        self._recommended_process_parallelism = 1
        return self._recommended_process_parallelism

    def recommended_supplement_strategy(self) -> dict[str, Any]:
        batch_size = self._recommended_batch_generate_size()
        if batch_size > 1:
            return {"mode": "batch", "parallelism": 1, "batch_size": batch_size}
        return {
            "mode": "parallel",
            "parallelism": self.recommended_process_parallelism(),
            "batch_size": 1,
        }

    def clean_pdf_vocabulary_entries(
        self,
        entries: list[WordEntry],
        language: str = "",
        progress_callback: Any | None = None,
    ) -> list[WordEntry]:
        job_id = self.submit_job(
            "clean_pdf_vocabulary_entries",
            {"entries": [self._entry_payload(entry) for entry in entries], "language": language},
        )
        data = self._wait_for_local_job(job_id, progress_callback)
        return [self._entry_from_payload(item) for item in data]

    def submit_job(self, method: str, params: dict[str, Any] | None = None) -> str:
        job_id = uuid.uuid4().hex
        with self._local_jobs_lock:
            self._local_jobs[job_id] = {
                "method": method,
                "params": params or {},
                "remote_job_id": "",
                "state": "queued",
                "stage": "queued",
                "progress": {"message": "正在启动 AI 模型服务", "percent": 0},
                "result": None,
                "error": "",
            }
        threading.Thread(target=self._drive_local_job, args=(job_id,), daemon=True).start()
        return job_id

    def job_status(self, job_id: str) -> dict[str, Any]:
        with self._local_jobs_lock:
            job = self._local_jobs.get(job_id)
            if job is None:
                raise RuntimeError(f"未知 AI 任务：{job_id}")
            return dict(job)

    def close(self) -> None:
        process = self._process
        self._process = None
        self._base_url = os.getenv("WORDPYCKET_LLM_SERVER_URL", "").strip()
        self._server_model_available = bool(self._base_url)
        self._ready_messages = None
        if not self._owns_process:
            return
        self._owns_process = False
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def isolated_command(self) -> list[str]:
        return [sys.executable, "-m", "wordpycket.infrastructure.example_generator"]

    def isolated_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if self._base_url:
            env["WORDPYCKET_LLM_SERVER_URL"] = self._base_url
        python_paths = [path for path in sys.path if path]
        existing_python_path = env.get("PYTHONPATH")
        if existing_python_path:
            python_paths.append(existing_python_path)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        return env

    def isolated_payload(self, action: str, entry: WordEntry, scope: str, language: str = "") -> str:
        payload = {
            "action": action,
            "model_dir": str(self._model_dir),
            "scope": scope,
            "language": language,
            "entry": self._entry_payload(entry),
        }
        return json.dumps(payload, ensure_ascii=False)

    def isolated_pdf_import_environment(self) -> dict[str, str]:
        env = self.isolated_environment()
        env["WORDPYCKET_PDF_CHILD"] = "1"
        return env

    def isolated_pdf_import_payload(self, pdf_path: Path, csv_path: Path) -> str:
        return json.dumps(
            {
                "pdf_path": str(pdf_path),
                "csv_path": str(csv_path),
                "use_llm_cleanup": True,
                "model_dir": str(self._model_dir),
            },
            ensure_ascii=False,
        )

    def parse_isolated_result(self, stdout: str, stderr: str, returncode: int) -> dict[str, Any]:
        if returncode != 0:
            detail = "\n".join(part for part in (stderr.strip(), stdout.strip()) if part)
            raise RuntimeError(f"模型子进程退出代码 {returncode}。{detail}")
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("模型子进程没有返回结果。")
        try:
            data = json.loads(lines[-1])
        except json.JSONDecodeError as error:
            raise RuntimeError(f"模型子进程返回内容不是有效 JSON：{lines[-1]}") from error
        if not isinstance(data, dict):
            raise RuntimeError(f"模型子进程返回内容不是 JSON 对象：{data}")
        if data.get("type") == "result":
            result = data.get("data", {})
            if not isinstance(result, dict):
                raise RuntimeError(f"模型子进程 result.data 不是 JSON 对象：{data}")
            return result
        if data.get("type") == "error":
            raise RuntimeError(str(data.get("error", "模型子进程失败。")))
        return data

    def _run_many_jobs(
        self,
        entries: list[WordEntry],
        action: str,
        scope: str,
        language: str,
        progress: Any,
        control: Any,
    ) -> tuple[list[tuple[WordEntry, dict[str, Any]]], list[str], int]:
        if not entries:
            return [], [], 0

        worker_count = min(len(entries), self.recommended_process_parallelism())
        results: list[tuple[WordEntry, dict[str, Any]]] = []
        errors: list[str] = []
        done = 0
        next_index = 0
        total = len(entries)
        active = set()

        def is_cancelled() -> bool:
            if control is None:
                return False
            if getattr(control, "cancelled", False):
                return True
            if callable(control):
                return control() == "stopped"
            return False

        def run_entry(entry: WordEntry) -> tuple[WordEntry, dict[str, Any]]:
            job_id = self.submit_job(
                "run_action",
                {
                    "action": action,
                    "entry": self._entry_payload(entry),
                    "scope": scope,
                    "language": language,
                },
            )
            while True:
                status = self.job_status(job_id)
                state = str(status.get("state", ""))
                if state == "completed":
                    result = status.get("result", {})
                    if not isinstance(result, dict):
                        raise RuntimeError(f"AI 返回结果格式不正确：{result}")
                    return entry, result
                if state == "failed":
                    raise RuntimeError(str(status.get("error", "AI 任务失败。")))
                time.sleep(self._poll_interval_seconds())

        def submit_next(executor: ThreadPoolExecutor) -> bool:
            nonlocal next_index
            if next_index >= total or is_cancelled():
                return False
            entry = entries[next_index]
            next_index += 1
            active.add(executor.submit(run_entry, entry))
            return True

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for _ in range(worker_count):
                if not submit_next(executor):
                    break
            while active:
                finished, active = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    done += 1
                    try:
                        results.append(future.result())
                    except Exception as error:
                        errors.append(str(error))
                    if progress is not None:
                        progress(done, total)
                while len(active) < worker_count and submit_next(executor):
                    pass
        return results, errors, worker_count

    def _wait_for_local_job(self, job_id: str, progress_callback: Any | None = None) -> Any:
        idle_deadline = time.monotonic() + self._operation_timeout_seconds()
        last_progress: tuple[str, int] | None = None
        while True:
            if time.monotonic() >= idle_deadline:
                raise RuntimeError(f"AI 任务超过 {self._operation_timeout_seconds()} 秒没有新进度。")
            status = self.job_status(job_id)
            progress = status.get("progress", {})
            if isinstance(progress, dict):
                message = str(progress.get("message", status.get("stage", "")))
                percent = int(progress.get("percent", 0))
                current_progress = (message, percent)
                if current_progress != last_progress:
                    idle_deadline = time.monotonic() + self._operation_timeout_seconds()
                if progress_callback is not None and current_progress != last_progress:
                    progress_callback(message, percent)
                last_progress = current_progress
            state = str(status.get("state", ""))
            if state == "completed":
                return status.get("result")
            if state == "failed":
                raise RuntimeError(str(status.get("error", "AI 任务失败。")))
            time.sleep(self._poll_interval_seconds())

    def _drive_local_job(self, job_id: str) -> None:
        try:
            self._ensure_server_for_local_job(job_id)
            if self._local_job_requires_ready_model(job_id) and not self._server_model_available:
                self._update_local_job(
                    job_id,
                    state="failed",
                    stage="failed",
                    error="AI 模型服务未声明可用模型。",
                )
                return
            self._submit_remote_job(job_id)
            while True:
                with self._local_jobs_lock:
                    job = self._local_jobs.get(job_id)
                    if job is None:
                        return
                    if job.get("state") in {"completed", "failed"}:
                        return
                self._poll_remote_job(job_id)
                time.sleep(self._poll_interval_seconds())
        except Exception as error:
            self._update_local_job(
                job_id,
                state="failed",
                stage="failed",
                progress={"message": "AI 任务失败", "percent": 100},
                error=str(error),
            )

    def _ensure_server_for_local_job(self, job_id: str) -> None:
        if self._base_url:
            return
        self._update_local_job(
            job_id,
            state="queued",
            stage="starting",
            progress={"message": "正在启动 AI 模型服务", "percent": 0},
        )
        with self._server_start_lock:
            if self._base_url:
                return
            self._start_server_blocking()

    def _start_server_blocking(self) -> None:
        if self._process is not None and self._ready_messages is not None:
            self._finish_async_server_start()
            return
        if self._process is None:
            self._start_server_async()
        self._finish_async_server_start()

    def _finish_async_server_start(self) -> None:
        if self._ready_messages is None:
            return
        try:
            line = self._ready_messages.get(timeout=self._startup_timeout_seconds()).strip()
        except queue.Empty as error:
            self._stop_unready_process()
            raise RuntimeError(f"AI 模型服务 {self._startup_timeout_seconds()} 秒内没有完成启动，已停止等待。") from error
        if not line:
            self._stop_unready_process()
            raise RuntimeError("AI 模型服务启动失败。")
        try:
            data = json.loads(line)
        except json.JSONDecodeError as error:
            self._stop_unready_process()
            raise RuntimeError(f"AI 模型服务返回内容格式不正确：{line}") from error
        if not isinstance(data, dict) or data.get("type") != "ready":
            self._stop_unready_process()
            raise RuntimeError(f"AI 模型服务未就绪：{data}")
        self._base_url = f"http://{data['host']}:{int(data['port'])}"
        self._server_model_available = bool(data.get("model_available"))
        self._ready_messages = None
        self._update_waiting_local_jobs(
            state="queued",
            stage="ready",
            progress={"message": "AI 模型服务已启动", "percent": 1},
        )

    def _local_job_requires_ready_model(self, job_id: str) -> bool:
        with self._local_jobs_lock:
            job = self._local_jobs.get(job_id)
            if job is None:
                return False
            return self._method_requires_ready_model(str(job["method"]), job.get("params"))

    def _update_local_job(self, job_id: str, **changes: Any) -> None:
        with self._local_jobs_lock:
            job = self._local_jobs.get(job_id)
            if job is not None:
                job.update(changes)

    def _update_waiting_local_jobs(self, **changes: Any) -> None:
        with self._local_jobs_lock:
            for job in self._local_jobs.values():
                if job.get("state") in {"queued", "running"} and not job.get("remote_job_id"):
                    job.update(changes)

    def _rpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        progress_callback: Any | None = None,
    ) -> Any:
        base_url = self._ensure_server(require_model=self._method_requires_ready_model(method, params))
        job = self._request_json(
            base_url,
            {
                "method": "submit_job",
                "params": {
                    "method": method,
                    "params": params or {},
                },
            },
        )
        result = job.get("result")
        if not isinstance(result, dict) or not result.get("job_id"):
            raise RuntimeError(f"AI 模型服务没有返回有效任务编号：{job}")
        return self._wait_for_job(base_url, str(result["job_id"]), progress_callback)

    def _request_json(self, base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
        host, port = self._parse_base_url(base_url)
        body = json.dumps(payload, ensure_ascii=False)
        connection = http.client.HTTPConnection(host, port, timeout=self._request_timeout_seconds())
        try:
            connection.request("POST", "/rpc", body=body.encode("utf-8"), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()
        if not isinstance(payload, dict):
            raise RuntimeError("AI 模型服务返回内容格式不正确。")
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error", "AI 模型服务调用失败。")))
        return payload

    def _wait_for_job(self, base_url: str, job_id: str, progress_callback: Any | None = None) -> Any:
        deadline = time.monotonic() + self._operation_timeout_seconds()
        last_progress: tuple[str, int] | None = None
        while True:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"AI 任务超过 {self._operation_timeout_seconds()} 秒未完成。")
            payload = self._request_json(
                base_url,
                {"method": "job_status", "params": {"job_id": job_id}},
            )
            status = payload.get("result")
            if not isinstance(status, dict):
                raise RuntimeError(f"AI 任务状态格式不正确：{payload}")
            progress = status.get("progress", {})
            if isinstance(progress, dict):
                message = str(progress.get("message", status.get("stage", "")))
                percent = int(progress.get("percent", 0))
                current_progress = (message, percent)
                if progress_callback is not None and current_progress != last_progress:
                    progress_callback(message, percent)
                last_progress = current_progress
            state = str(status.get("state", ""))
            if state == "completed":
                return status.get("result")
            if state == "failed":
                raise RuntimeError(str(status.get("error", "AI 任务失败。")))
            time.sleep(self._poll_interval_seconds())

    def _ensure_server(self, require_model: bool = True) -> str:
        if self._base_url:
            if require_model and not self._server_model_available:
                raise RuntimeError("AI 模型服务未声明可用模型。")
            return self._base_url
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        python_paths = [path for path in sys.path if path]
        existing_python_path = env.get("PYTHONPATH")
        if existing_python_path:
            python_paths.append(existing_python_path)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        self._process = subprocess.Popen(
            [sys.executable, "-m", "llmserver.server", "--model-dir", str(self._model_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )
        self._owns_process = True
        assert self._process.stdout is not None
        try:
            data = self._read_ready_message(self._process)
        except Exception:
            self._stop_unready_process()
            raise
        if data.get("type") != "ready":
            self._stop_unready_process()
            raise RuntimeError(f"AI 模型服务启动消息格式不正确：{data}")
        self._base_url = f"http://{data['host']}:{int(data['port'])}"
        self._server_model_available = bool(data.get("model_available"))
        if require_model and not self._server_model_available:
            self._stop_unready_process()
            raise RuntimeError("AI 模型服务未声明可用模型。")
        return self._base_url

    def _start_server_async(self) -> None:
        if self._process is not None:
            return
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        python_paths = [path for path in sys.path if path]
        existing_python_path = env.get("PYTHONPATH")
        if existing_python_path:
            python_paths.append(existing_python_path)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        self._process = subprocess.Popen(
            [sys.executable, "-m", "llmserver.server", "--model-dir", str(self._model_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )
        self._owns_process = True
        assert self._process.stdout is not None
        self._ready_messages = queue.Queue(maxsize=1)

        def read_ready() -> None:
            assert self._process is not None and self._process.stdout is not None
            try:
                self._ready_messages.put(self._process.stdout.readline())
            except Exception as error:
                self._ready_messages.put(json.dumps({"type": "error", "error": str(error)}))

        threading.Thread(target=read_ready, daemon=True).start()

    def _advance_server_start(self, job: dict[str, Any]) -> None:
        if self._ready_messages is None:
            self._start_server_async()
            return
        try:
            line = self._ready_messages.get_nowait().strip()
        except queue.Empty:
            job.update(
                {
                    "state": "queued",
                    "stage": "starting",
                    "progress": {"message": "正在启动 AI 模型服务", "percent": 0},
                }
            )
            return
        if not line:
            job.update({"state": "failed", "stage": "failed", "error": "AI 模型服务启动失败。"})
            return
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            job.update({"state": "failed", "stage": "failed", "error": f"AI 模型服务返回内容格式不正确：{line}"})
            return
        if not isinstance(data, dict) or data.get("type") != "ready":
            job.update({"state": "failed", "stage": "failed", "error": f"AI 模型服务未就绪：{data}"})
            return
        self._base_url = f"http://{data['host']}:{int(data['port'])}"
        self._server_model_available = bool(data.get("model_available"))
        if self._method_requires_ready_model(str(job["method"]), job.get("params")) and not self._server_model_available:
            job.update({"state": "failed", "stage": "failed", "error": "AI 模型服务未声明可用模型。"})
            return
        job.update(
            {
                "state": "queued",
                "stage": "ready",
                "progress": {"message": "AI 模型服务已启动", "percent": 1},
            }
        )

    def _submit_remote_job(self, job_id: str) -> None:
        with self._local_jobs_lock:
            job = self._local_jobs.get(job_id)
            if job is None:
                return
            method = job["method"]
            params = job["params"]
        payload = self._request_json(
            self._base_url,
            {
                "method": "submit_job",
                "params": {"method": method, "params": params},
            },
        )
        result = payload.get("result")
        if not isinstance(result, dict) or not result.get("job_id"):
            self._update_local_job(
                job_id,
                state="failed",
                stage="failed",
                error=f"AI 模型服务没有返回任务编号：{payload}",
            )
            return
        self._update_local_job(
            job_id,
            remote_job_id=str(result["job_id"]),
            state="queued",
            stage="submitted",
            progress={"message": "任务已提交给 LLM", "percent": 1},
        )

    def _poll_remote_job(self, job_id: str) -> None:
        with self._local_jobs_lock:
            job = self._local_jobs.get(job_id)
            if job is None:
                return
            remote_job_id = str(job["remote_job_id"])
        payload = self._request_json(
            self._base_url,
            {"method": "job_status", "params": {"job_id": remote_job_id}},
        )
        status = payload.get("result")
        if not isinstance(status, dict):
            self._update_local_job(
                job_id,
                state="failed",
                stage="failed",
                error=f"AI 任务状态格式不正确：{payload}",
            )
            return
        self._update_local_job(job_id, **status)

    def _read_ready_message(self, process: subprocess.Popen[str]) -> dict[str, Any]:
        assert process.stdout is not None
        messages: queue.Queue[str] = queue.Queue(maxsize=1)

        def read_line() -> None:
            try:
                messages.put(process.stdout.readline())
            except Exception as error:
                messages.put(json.dumps({"type": "error", "error": str(error)}))

        thread = threading.Thread(target=read_line, daemon=True)
        thread.start()
        timeout = self._startup_timeout_seconds()
        try:
            line = messages.get(timeout=timeout).strip()
        except queue.Empty as error:
            raise RuntimeError(f"AI 模型服务 {timeout} 秒内没有完成启动，已停止等待。") from error
        if not line:
            stderr = ""
            if process.poll() is not None and process.stderr is not None:
                stderr = process.stderr.read().strip()
            raise RuntimeError(f"AI 模型服务启动失败。{stderr}")
        try:
            data = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"AI 模型服务启动消息格式不正确：{line}") from error
        if not isinstance(data, dict):
            raise RuntimeError(f"AI 模型服务启动消息格式不正确：{data}")
        return data

    def _stop_unready_process(self) -> None:
        process = self._process
        self._process = None
        self._base_url = ""
        self._server_model_available = False
        self._owns_process = False
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    @staticmethod
    def _method_requires_ready_model(method: str, params: dict[str, Any] | None) -> bool:
        if method in {
            "ensure_model_available",
            "device_status",
            "check_model_runtime",
            "recommended_process_parallelism",
            "recommended_supplement_strategy",
        }:
            return False
        if method == "run_action" and isinstance(params, dict) and params.get("action") == "smoke_test":
            return False
        return True

    @staticmethod
    def _startup_timeout_seconds() -> int:
        raw_value = os.getenv("WORDPYCKET_LLM_STARTUP_TIMEOUT", "5")
        try:
            return max(1, int(raw_value))
        except ValueError as error:
            raise RuntimeError("WORDPYCKET_LLM_STARTUP_TIMEOUT 必须是整数秒。") from error

    def _find_existing_model_path(self) -> Path | None:
        if not self._model_dir.exists():
            return None

        models = sorted(
            self._model_dir.glob("*.gguf"),
            key=lambda path: (
                path.name != self.DEFAULT_MODEL_FILENAME,
                -path.stat().st_size,
            ),
        )
        if len(models) > 1:
            names = "、".join(path.name for path in models)
            raise RuntimeError(f"本地模型文件夹中只能保留一个 .gguf 模型文件。当前存在：{names}")
        return models[0] if models else None

    @staticmethod
    def _model_size_mb(model_path: Path | None) -> int:
        if model_path is None:
            return 2048
        return max(1, model_path.stat().st_size // (1024 * 1024))

    def _recommended_batch_generate_size(self) -> int:
        override = os.getenv("WORDPYCKET_LLM_BATCH_SIZE")
        if override is not None:
            try:
                return max(1, min(32, int(override)))
            except ValueError as error:
                raise RuntimeError("WORDPYCKET_LLM_BATCH_SIZE 必须是整数。") from error

        model_path = self._find_existing_model_path()
        model_size_mb = self._model_size_mb(model_path)
        vram_mb = self._cuda_capacity_mb()
        if vram_mb is None:
            return 1
        single_instance_mb = max(2600, int(model_size_mb * 0.65) + 900)
        if vram_mb < single_instance_mb * 5:
            return 8
        return 1

    @staticmethod
    def _threads_per_model() -> int:
        cpu_count = os.cpu_count() or 1
        if cpu_count <= 4:
            return max(1, cpu_count - 1)
        return min(3, cpu_count - 2)

    @classmethod
    def _cuda_capacity_mb(cls) -> int | None:
        if os.getenv("CUDA_VISIBLE_DEVICES") == "-1":
            return None
        total_vram_mb = cls._cuda_total_vram_mb()
        if total_vram_mb is not None:
            return total_vram_mb
        return cls._cuda_free_vram_mb()

    @staticmethod
    def _cuda_total_vram_mb() -> int | None:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            return None
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
        return max(values) if values else None

    @staticmethod
    def _cuda_free_vram_mb() -> int | None:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            return None
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
        return max(values) if values else None

    @classmethod
    def _system_capacity_mb(cls) -> int:
        return cls._system_total_memory_mb() or cls._system_memory_mb()

    @staticmethod
    def _system_total_memory_mb() -> int | None:
        if platform.system() == "Windows":
            try:
                import ctypes

                class MemoryStatus(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                status = MemoryStatus()
                status.dwLength = ctypes.sizeof(status)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
                return max(1, status.ullTotalPhys // (1024 * 1024))
            except Exception:
                return None

        if hasattr(os, "sysconf"):
            try:
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                return max(1, (pages * page_size) // (1024 * 1024))
            except (OSError, ValueError):
                return None
        return None

    @staticmethod
    def _system_memory_mb() -> int:
        if platform.system() == "Windows":
            try:
                import ctypes

                class MemoryStatus(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                status = MemoryStatus()
                status.dwLength = ctypes.sizeof(status)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
                return max(1, status.ullAvailPhys // (1024 * 1024))
            except Exception:
                return 4096

        if hasattr(os, "sysconf"):
            try:
                pages = os.sysconf("SC_AVPHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                return max(1, (pages * page_size) // (1024 * 1024))
            except (OSError, ValueError):
                return 4096
        return 4096

    @staticmethod
    def _request_timeout_seconds() -> int:
        raw_value = os.getenv("WORDPYCKET_LLM_HTTP_TIMEOUT", "5")
        try:
            return max(1, int(raw_value))
        except ValueError as error:
            raise RuntimeError("WORDPYCKET_LLM_HTTP_TIMEOUT 必须是整数秒。") from error

    @staticmethod
    def _operation_timeout_seconds() -> int:
        raw_value = os.getenv("WORDPYCKET_LLM_TIMEOUT", "300")
        try:
            return max(1, int(raw_value))
        except ValueError as error:
            raise RuntimeError("WORDPYCKET_LLM_TIMEOUT 必须是整数秒。") from error

    @staticmethod
    def _poll_interval_seconds() -> float:
        raw_value = os.getenv("WORDPYCKET_LLM_POLL_INTERVAL", "0.25")
        try:
            return max(0.05, float(raw_value))
        except ValueError as error:
            raise RuntimeError("WORDPYCKET_LLM_POLL_INTERVAL 必须是数字秒。") from error

    @staticmethod
    def _parse_base_url(base_url: str) -> tuple[str, int]:
        if not base_url.startswith("http://"):
            raise RuntimeError("WORDPYCKET_LLM_SERVER_URL 目前只支持 http:// 地址。")
        value = base_url.removeprefix("http://").rstrip("/")
        host, _, port_text = value.partition(":")
        return host, int(port_text)

    @classmethod
    def _select_device(cls, requested: str, detected: str, supports_gpu_offload: bool) -> str:
        allowed = {cls._AUTO_DEVICE, cls._CPU_DEVICE, cls._CUDA_DEVICE, cls._MPS_DEVICE}
        if requested not in allowed:
            raise RuntimeError("WORDPYCKET_LLM_DEVICE 只能设置为 auto、cpu、cuda 或 mps。")
        if requested == cls._CPU_DEVICE:
            return cls._CPU_DEVICE
        if requested in {cls._CUDA_DEVICE, cls._MPS_DEVICE}:
            if requested != detected:
                raise RuntimeError(f"没有检测到可用的 {requested.upper()} 加速设备。")
            if not supports_gpu_offload:
                raise RuntimeError(
                    "当前运行环境不支持 GPU 加速。"
                    f"请安装支持 {requested.upper()} 的 wheel，或设置 WORDPYCKET_LLM_DEVICE=cpu。"
                )
            return requested
        if supports_gpu_offload and detected in {cls._CUDA_DEVICE, cls._MPS_DEVICE}:
            return detected
        return cls._CPU_DEVICE

    @classmethod
    def _detect_accelerator(cls) -> str:
        if cls._has_cuda_device():
            return cls._CUDA_DEVICE
        if cls._has_mps_device():
            return cls._MPS_DEVICE
        return cls._CPU_DEVICE

    @staticmethod
    def _has_cuda_device() -> bool:
        if os.getenv("CUDA_VISIBLE_DEVICES") == "-1":
            return False
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            return False
        try:
            result = subprocess.run(
                [nvidia_smi, "-L"],
                capture_output=True,
                check=False,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and "GPU" in result.stdout

    @staticmethod
    def _has_mps_device() -> bool:
        return platform.system() == "Darwin" and platform.machine() in {"arm64", "arm"}

    @staticmethod
    def _entry_payload(entry: WordEntry) -> dict[str, Any]:
        return {
            "word": entry.word,
            "meaning": entry.meaning,
            "source_index": entry.source_index,
            "frequency": entry.frequency,
            "forms": entry.forms,
            "example_sentence": entry.example_sentence,
            "example_sentence_cn": entry.example_sentence_cn,
        }

    @staticmethod
    def _entry_from_payload(data: dict[str, Any]) -> WordEntry:
        return WordEntry(
            word=str(data.get("word", "")),
            meaning=str(data.get("meaning", "")),
            source_index=int(data.get("source_index", 0)),
            frequency=int(data.get("frequency", 0)),
            forms=str(data.get("forms", "")),
            example_sentence=str(data.get("example_sentence", "")),
            example_sentence_cn=str(data.get("example_sentence_cn", "")),
        )

    @staticmethod
    def _model_status_from_payload(data: dict[str, Any]) -> ModelStatus:
        path = data.get("path")
        return ModelStatus(
            path=Path(path) if path else None,
            is_user_model=bool(data.get("is_user_model", False)),
            downloaded=bool(data.get("downloaded", False)),
        )

    @staticmethod
    def _device_status_from_payload(data: dict[str, Any]) -> DeviceStatus:
        selected = data.get("selected")
        supported = data.get("gpu_offload_supported")
        return DeviceStatus(
            requested=str(data.get("requested", "")),
            detected=str(data.get("detected", "")),
            selected=str(selected) if selected is not None else None,
            gpu_offload_supported=bool(supported) if supported is not None else None,
            error=str(data.get("error", "")),
        )


def _main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        entry = LocalLlmExampleGenerator._entry_from_payload(payload["entry"])
        generator = LocalLlmExampleGenerator(Path(str(payload.get("model_dir", "model"))))
        result = generator._rpc(
            "run_action",
            {
                "action": str(payload["action"]),
                "entry": LocalLlmExampleGenerator._entry_payload(entry),
                "scope": str(payload.get("scope", "")),
                "language": str(payload.get("language", "")),
            },
        )
        message = {"type": "result", "data": result}
    except Exception as error:
        message = {"type": "error", "error": str(error)}
    print(json.dumps(message, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    _main()
