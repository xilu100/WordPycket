from __future__ import annotations

import json
import logging
import gc
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from queue import Queue
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable

from llmserver import llama_runtime, model_store, pdf_cleaning, prompts
from llmserver.contracts import (
    DeviceStatus,
    GeneratedCorrection,
    GeneratedExample,
    GeneratedExplanation,
    ModelCheckResult,
    ModelStatus,
    WordEntry,
)

ProgressCallback = Any
ControlCallback = Any
LOGGER = logging.getLogger(__name__)


def _is_llama_cleanup_error(unraisable: Any) -> bool:
    return llama_runtime.is_llama_cleanup_error(unraisable)


def _install_llama_cleanup_error_filter() -> None:
    llama_runtime.install_llama_cleanup_error_filter()


_install_llama_cleanup_error_filter()

class LocalLlmExampleGenerator:
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
        self._model_dir = model_dir
        self._llms: list[Any | None] = []
        self._llm_lock = threading.Lock()
        self._recommended_process_parallelism: int | None = None
        self._parallel_worker_count: int | None = None

    def generate(self, entry: WordEntry, scope: str = "") -> GeneratedExample:
        llm = self._load_model(0)
        prompt = self._build_prompt(entry, scope)
        content = self._call_model(llm, prompt)
        return self._parse_response(content, require_meaning=not entry.meaning.strip())

    def generate_isolated(self, entry: WordEntry, scope: str = "") -> GeneratedExample:
        if os.getenv("WORDPYCKET_LLM_CHILD") == "1":
            return self.generate(entry, scope)
        data = self._run_isolated_worker("generate", entry, scope)
        return GeneratedExample(
            example_sentence=str(data["example_sentence"]),
            example_sentence_cn=str(data["example_sentence_cn"]),
            meaning=str(data.get("meaning", "")),
        )

    def correct_entry(self, entry: WordEntry, scope: str = "") -> GeneratedCorrection:
        llm = self._load_model(0)
        prompt = self._build_correction_prompt(entry, scope)
        content = self._call_model(llm, prompt)
        return self._parse_correction_response(content, entry)

    def correct_entry_isolated(self, entry: WordEntry, scope: str = "") -> GeneratedCorrection:
        if os.getenv("WORDPYCKET_LLM_CHILD") == "1":
            return self.correct_entry(entry, scope)
        data = self._run_isolated_worker("correct", entry, scope)
        return GeneratedCorrection(
            corrected_word=str(data["corrected_word"]),
            note=str(data.get("note", "")),
        )

    def explain_entry(self, entry: WordEntry, scope: str = "", language: str = "") -> GeneratedExplanation:
        llm = self._load_model(0)
        prompt = self._build_explanation_prompt(entry, scope, language)
        content = self._call_model(
            llm,
            prompt,
            system_prompt="You explain target-language vocabulary usage for Chinese learners. Return only valid JSON.",
            max_tokens=self._explanation_max_tokens(),
        )
        return self._parse_explanation_response(content)

    def explain_entry_isolated(self, entry: WordEntry, scope: str = "", language: str = "") -> GeneratedExplanation:
        if os.getenv("WORDPYCKET_LLM_CHILD") == "1":
            return self.explain_entry(entry, scope, language)
        data = self._run_isolated_worker("explain", entry, scope, language)
        return GeneratedExplanation(explanation=str(data["explanation"]))

    def _explain_with_slot(
        self,
        slot: int,
        entry: WordEntry,
        scope: str = "",
        language: str = "",
    ) -> GeneratedExplanation:
        llm = self._load_model(slot)
        prompt = self._build_explanation_prompt(entry, scope, language)
        content = self._call_model(
            llm,
            prompt,
            system_prompt="You explain target-language vocabulary usage for Chinese learners. Return only valid JSON.",
            max_tokens=self._explanation_max_tokens(),
        )
        return self._parse_explanation_response(content)

    def generate_many(
        self,
        entries: list[WordEntry],
        scope: str = "",
        progress: ProgressCallback | None = None,
        control: ControlCallback | None = None,
    ) -> tuple[list[tuple[WordEntry, GeneratedExample]], list[str], int]:
        return self._run_parallel(entries, self._generate_with_slot, scope, progress, control)

    def generate_batch(self, entries: list[WordEntry], scope: str = "") -> list[tuple[WordEntry, GeneratedExample]]:
        return self._generate_batch_with_slot(0, entries, scope)

    def correct_many(
        self,
        entries: list[WordEntry],
        scope: str = "",
        progress: ProgressCallback | None = None,
        control: ControlCallback | None = None,
    ) -> tuple[list[tuple[WordEntry, GeneratedCorrection]], list[str], int]:
        return self._run_parallel(entries, self._correct_with_slot, scope, progress, control)

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
        existing_path = self._find_existing_model_path()
        if existing_path is not None:
            return ModelStatus(
                path=existing_path,
                is_user_model=existing_path.name != self.DEFAULT_MODEL_FILENAME,
            )
        model_path = self._download_default_model()
        return ModelStatus(path=model_path, is_user_model=False, downloaded=True)

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
            selected = self._select_device(supports_gpu_offload)
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
        model = self.ensure_model_available()
        device = self.device_status()
        if device.error:
            raise RuntimeError(f"设备检查失败：{device.error}")
        self.run_smoke_test_isolated()
        return ModelCheckResult(model=model, device=device, smoke_test_passed=True)

    def run_smoke_test_isolated(self) -> None:
        if os.getenv("WORDPYCKET_LLM_CHILD") == "1":
            self._run_smoke_test()
            return
        data = self._run_isolated_worker(
            "smoke_test",
            WordEntry(word="test", meaning="测试"),
            "",
        )
        if data.get("ok") is not True:
            raise RuntimeError(f"模型最小执行测试返回异常：{data}")

    def _run_isolated_worker(
        self,
        action: str,
        entry: WordEntry,
        scope: str,
        language: str = "",
    ) -> dict[str, Any]:
        env = self.isolated_environment()
        payload = self.isolated_payload(action, entry, scope, language)
        timeout = self._isolated_timeout_seconds()
        try:
            result = subprocess.run(
                self.isolated_command(),
                input=payload,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"模型子进程超过 {timeout} 秒未完成。") from error

        return self.parse_isolated_result(result.stdout, result.stderr, result.returncode)

    def isolated_command(self) -> list[str]:
        return [sys.executable, "-m", "llmserver.engine"]

    def isolated_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env["WORDPYCKET_LLM_CHILD"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
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
            "entry": {
                "word": entry.word,
                "meaning": entry.meaning,
                "source_index": entry.source_index,
                "frequency": entry.frequency,
                "forms": entry.forms,
                "example_sentence": entry.example_sentence,
                "example_sentence_cn": entry.example_sentence_cn,
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def isolated_pdf_import_environment(self) -> dict[str, str]:
        env = self.isolated_environment()
        env["WORDPYCKET_PDF_CHILD"] = "1"
        return env

    def isolated_pdf_import_payload(self, pdf_path: Path, csv_path: Path) -> str:
        payload = {
            "pdf_path": str(pdf_path),
            "csv_path": str(csv_path),
            "use_llm_cleanup": True,
            "model_dir": str(self._model_dir),
        }
        return json.dumps(payload, ensure_ascii=False)

    def parse_isolated_result(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> dict[str, Any]:
        if returncode != 0:
            recovered = self._parse_isolated_stdout(stdout)
            if recovered is not None:
                return recovered
            detail = (stderr or stdout).strip()
            if detail:
                raise RuntimeError(f"模型子进程退出代码 {returncode}：{detail}")
            raise RuntimeError(f"模型子进程退出代码 {returncode}。")

        data = self._parse_isolated_stdout(stdout)
        if data is None:
            raise RuntimeError(f"模型子进程返回内容不是有效 JSON：{stdout}")
        return data

    def recommended_process_parallelism(self) -> int:
        override = os.getenv("WORDPYCKET_LLM_PROCESS_PARALLEL")
        if override is not None:
            try:
                return max(1, min(8, int(override)))
            except ValueError as error:
                raise RuntimeError("WORDPYCKET_LLM_PROCESS_PARALLEL 必须是整数。") from error
        if self._recommended_process_parallelism is not None:
            return self._recommended_process_parallelism

        model_path = self._find_existing_model_path()
        model_size_mb = self._model_size_mb(model_path)
        memory_mb = self._system_capacity_mb()
        cpu_count = os.cpu_count() or 1
        accelerator = self._detect_accelerator()

        if accelerator == self._MPS_DEVICE:
            self._recommended_process_parallelism = 1
            return self._recommended_process_parallelism

        if accelerator == self._CUDA_DEVICE:
            vram_mb = self._cuda_capacity_mb()
            if vram_mb is not None:
                vram_per_worker_mb = max(2600, int(model_size_mb * 0.65) + 900)
                ram_per_worker_mb = max(2600, int(model_size_mb * 0.9) + 900)
                vram_workers = max(1, vram_mb // vram_per_worker_mb)
                ram_workers = max(1, memory_mb // ram_per_worker_mb)
                self._recommended_process_parallelism = max(1, min(4, vram_workers, ram_workers))
                return self._recommended_process_parallelism
            self._recommended_process_parallelism = 1
            return self._recommended_process_parallelism

        per_worker_mb = max(3600, int(model_size_mb * 2.0) + 1400)
        memory_workers = max(1, memory_mb // per_worker_mb)
        cpu_budget = max(1, cpu_count - 2)
        cpu_workers = max(1, cpu_budget // self._threads_per_model())
        self._recommended_process_parallelism = max(1, min(2, memory_workers, cpu_workers))
        return self._recommended_process_parallelism

    def recommended_supplement_strategy(self) -> dict[str, int | str]:
        batch_size = self._recommended_batch_generate_size()
        if batch_size > 1:
            return {"mode": "batch", "parallelism": 1, "batch_size": batch_size}
        return {
            "mode": "parallel",
            "parallelism": self.recommended_process_parallelism(),
            "batch_size": 1,
        }

    def _recommended_batch_generate_size(self) -> int:
        override = os.getenv("WORDPYCKET_LLM_BATCH_SIZE")
        if override is not None:
            try:
                return max(1, min(32, int(override)))
            except ValueError as error:
                raise RuntimeError("WORDPYCKET_LLM_BATCH_SIZE 必须是整数。") from error

        model_path = self._find_existing_model_path()
        model_size_mb = self._model_size_mb(model_path)
        if self._detect_accelerator() == self._CUDA_DEVICE:
            vram_mb = self._cuda_capacity_mb()
            if vram_mb is None:
                return 1
            single_instance_mb = max(2600, int(model_size_mb * 0.65) + 900)
            if vram_mb < single_instance_mb * 5:
                return 8
            return 1
        return 1

    @staticmethod
    def _parse_isolated_stdout(stdout: str) -> dict[str, Any] | None:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return data

    @staticmethod
    def _isolated_timeout_seconds() -> int:
        raw_value = os.getenv("WORDPYCKET_LLM_TIMEOUT", "300")
        try:
            return max(1, int(raw_value))
        except ValueError as error:
            raise RuntimeError("WORDPYCKET_LLM_TIMEOUT 必须是整数秒。") from error

    def _load_model(self, slot: int):
        with self._llm_lock:
            while len(self._llms) <= slot:
                self._llms.append(None)
            if self._llms[slot] is not None:
                return self._llms[slot]

            model_path = self._ensure_model_path()
            if model_path is None:
                raise RuntimeError("model 目录中没有找到 .gguf 模型文件。")

            try:
                from llama_cpp import Llama
                from llama_cpp import llama_supports_gpu_offload
            except ImportError as error:
                raise RuntimeError(
                    "缺少 llama-cpp-python。请先安装后再使用补充例句功能。"
                ) from error

            device = self._select_device(llama_supports_gpu_offload())
            model_options = {
                "model_path": str(model_path),
                "n_ctx": self._context_size(model_path),
                "n_threads": self._threads_per_model(),
                "chat_format": "chatml",
                "verbose": False,
            }
            if device in {self._CUDA_DEVICE, self._MPS_DEVICE}:
                model_options["n_gpu_layers"] = self._gpu_layers()

            try:
                self._llms[slot] = self._create_llama(Llama, model_options)
            except OSError as error:
                if "0xc000001d" in str(error) or "-1073741795" in str(error):
                    raise RuntimeError(
                        "llama-cpp-python 当前 wheel 与这台机器的 CPU 指令集不兼容。"
                        "请安装适合本机 CPU 的 no-AVX/basic 版本 wheel；"
                        "如果有 NVIDIA CUDA，也可以安装 llama-cpp-python 的 CUDA wheel。"
                    ) from error
                raise
            return self._llms[slot]

    def _run_parallel(
        self,
        entries: list[WordEntry],
        worker: Any,
        scope: str,
        progress: ProgressCallback | None,
        control: ControlCallback | None,
    ):
        if not entries:
            return [], [], 0

        worker_count = min(len(entries), self._parallel_workers())
        slots: Queue[int] = Queue()
        for slot in range(worker_count):
            slots.put(slot)

        results = []
        errors: list[str] = []
        done = 0
        next_index = 0
        active = set()

        def wait_if_paused() -> bool:
            if control is None:
                return True
            while True:
                state = control()
                if state == "stopped":
                    return False
                if state != "paused":
                    return True
                threading.Event().wait(0.2)

        def submit_next(executor: ThreadPoolExecutor) -> bool:
            nonlocal next_index
            if next_index >= len(entries):
                return False
            if not wait_if_paused():
                return False
            entry = entries[next_index]
            next_index += 1
            active.add(executor.submit(self._run_with_slot, slots, worker, entry, scope))
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
                        progress(done, len(entries), worker_count)

                if control is not None and control() == "stopped":
                    break

                while len(active) < worker_count and submit_next(executor):
                    pass

        return results, errors, worker_count

    def _run_with_slot(
        self,
        slots: Queue[int],
        worker: Any,
        entry: WordEntry,
        scope: str = "",
    ):
        slot = slots.get()
        try:
            return worker(slot, entry, scope)
        except Exception as error:
            raise RuntimeError(f"{entry.word}: {error}") from error
        finally:
            slots.put(slot)

    def _generate_with_slot(
        self,
        slot: int,
        entry: WordEntry,
        scope: str = "",
    ) -> tuple[WordEntry, GeneratedExample]:
        llm = self._load_model(slot)
        prompt = self._build_prompt(entry, scope)
        content = self._call_model(llm, prompt)
        return entry, self._parse_response(content, require_meaning=not entry.meaning.strip())

    def _generate_batch_with_slot(
        self,
        slot: int,
        entries: list[WordEntry],
        scope: str = "",
    ) -> list[tuple[WordEntry, GeneratedExample]]:
        llm = self._load_model(slot)
        return self._generate_batch_with_llm(llm, entries, scope)

    def _generate_batch_with_llm(
        self,
        llm: Any,
        entries: list[WordEntry],
        scope: str = "",
    ) -> list[tuple[WordEntry, GeneratedExample]]:
        prompt = self._build_batch_prompt(entries, scope)
        content = self._call_model(llm, prompt, max_tokens=self._batch_max_tokens(len(entries)))
        try:
            return self._parse_batch_response(content, entries)
        except Exception:
            if len(entries) <= 1:
                raise
            midpoint = max(1, len(entries) // 2)
            return [
                *self._generate_batch_with_llm(llm, entries[:midpoint], scope),
                *self._generate_batch_with_llm(llm, entries[midpoint:], scope),
            ]

    def _correct_with_slot(
        self,
        slot: int,
        entry: WordEntry,
        scope: str = "",
    ) -> tuple[WordEntry, GeneratedCorrection]:
        llm = self._load_model(slot)
        prompt = self._build_correction_prompt(entry, scope)
        content = self._call_model(llm, prompt)
        return entry, self._parse_correction_response(content, entry)

    def _parallel_workers(self) -> int:
        override = os.getenv("WORDPYCKET_LLM_PARALLEL")
        if override is not None:
            try:
                return max(1, int(override))
            except ValueError as error:
                raise RuntimeError("WORDPYCKET_LLM_PARALLEL 必须是整数。") from error
        if self._parallel_worker_count is not None:
            return self._parallel_worker_count

        model_path = self._find_existing_model_path()
        model_size_mb = self._model_size_mb(model_path)
        if self._detect_accelerator() == self._CUDA_DEVICE:
            vram_mb = self._cuda_capacity_mb()
            if vram_mb is not None:
                per_worker_mb = max(2200, int(model_size_mb * 0.95) + 500)
                self._parallel_worker_count = max(1, min(5, vram_mb // per_worker_mb))
                return self._parallel_worker_count
            self._parallel_worker_count = 1
            return self._parallel_worker_count

        if self._has_mps_device():
            memory_mb = self._system_capacity_mb()
            per_worker_mb = max(3200, int(model_size_mb * 1.8) + 1200)
            self._parallel_worker_count = max(1, min(3, memory_mb // per_worker_mb))
            return self._parallel_worker_count

        memory_mb = self._system_capacity_mb()
        cpu_count = os.cpu_count() or 1
        per_worker_mb = max(3200, int(model_size_mb * 1.8) + 1200)
        cpu_budget = max(1, cpu_count - 2)
        self._parallel_worker_count = max(1, min(2, cpu_budget // self._threads_per_model(), memory_mb // per_worker_mb))
        return self._parallel_worker_count

    @staticmethod
    def _threads_per_model() -> int:
        cpu_count = os.cpu_count() or 1
        if cpu_count <= 4:
            return max(1, cpu_count - 1)
        return min(3, cpu_count - 2)

    @staticmethod
    def _model_size_mb(model_path: Path | None) -> int:
        if model_path is None:
            return 2048
        return max(1, model_path.stat().st_size // (1024 * 1024))

    @classmethod
    def _cuda_capacity_mb(cls) -> int | None:
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
        values = [
            int(line.strip())
            for line in result.stdout.splitlines()
            if line.strip().isdigit()
        ]
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
        values = [
            int(line.strip())
            for line in result.stdout.splitlines()
            if line.strip().isdigit()
        ]
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
    def _create_llama(llama_class: Any, model_options: dict[str, Any]) -> Any:
        original_unraisablehook = sys.unraisablehook

        def suppress_llama_cleanup_error(unraisable: Any) -> None:
            if not _is_llama_cleanup_error(unraisable):
                original_unraisablehook(unraisable)

        sys.unraisablehook = suppress_llama_cleanup_error
        try:
            return llama_class(**model_options)
        except BaseException:
            gc.collect()
            raise
        finally:
            sys.unraisablehook = original_unraisablehook

    @classmethod
    def _select_device(cls, supports_gpu_offload: bool) -> str:
        requested = os.getenv("WORDPYCKET_LLM_DEVICE", cls._AUTO_DEVICE).lower()
        allowed = {cls._AUTO_DEVICE, cls._CPU_DEVICE, cls._CUDA_DEVICE, cls._MPS_DEVICE}
        if requested not in allowed:
            raise RuntimeError(
                "WORDPYCKET_LLM_DEVICE 只能设置为 auto、cpu、cuda 或 mps。"
            )
        if requested == cls._CPU_DEVICE:
            return cls._CPU_DEVICE

        detected = cls._detect_accelerator()
        if requested in {cls._CUDA_DEVICE, cls._MPS_DEVICE}:
            if requested != detected:
                raise RuntimeError(f"没有检测到可用的 {requested.upper()} 加速设备。")
            if not supports_gpu_offload:
                raise RuntimeError(
                    "当前 llama-cpp-python 不支持 GPU offload。"
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

    @classmethod
    def _context_size(cls, model_path: Path | None = None) -> int:
        raw_value = os.getenv("WORDPYCKET_LLM_CONTEXT")
        if raw_value is None:
            return cls._auto_context_size(model_path)
        try:
            return max(2048, min(32768, int(raw_value)))
        except ValueError as error:
            raise RuntimeError("WORDPYCKET_LLM_CONTEXT 必须是整数。") from error

    @classmethod
    def _auto_context_size(cls, model_path: Path | None = None) -> int:
        memory_mb = cls._system_capacity_mb()
        model_size_mb = cls._model_size_mb(model_path)
        cpu_count = os.cpu_count() or 1

        if cls._detect_accelerator() == cls._CUDA_DEVICE:
            vram_mb = cls._cuda_capacity_mb()
            if vram_mb is not None:
                budget_mb = min(memory_mb, max(0, vram_mb - max(1200, model_size_mb // 3)))
                if budget_mb >= 15000:
                    return 32768
                if budget_mb >= 8500:
                    return 16384
                if budget_mb >= 4200:
                    return 8192
                return 4096 if budget_mb >= 2500 else 2048

        if cls._has_mps_device():
            if memory_mb >= 48000:
                return 32768
            if memory_mb >= 24000:
                return 16384
            if memory_mb >= 12000:
                return 8192
            return 4096

        if memory_mb >= 48000 and cpu_count >= 12:
            return 16384
        if memory_mb >= 24000 and cpu_count >= 8:
            return 8192
        return 4096 if memory_mb >= 8000 else 2048

    @staticmethod
    def _gpu_layers() -> int:
        raw_value = os.getenv("WORDPYCKET_LLM_GPU_LAYERS")
        if raw_value is None:
            return -1
        try:
            return int(raw_value)
        except ValueError as error:
            raise RuntimeError("WORDPYCKET_LLM_GPU_LAYERS 必须是整数。") from error

    def clean_pdf_vocabulary_entries(
        self,
        entries: list[WordEntry],
        language: str,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> list[WordEntry]:
        if not entries:
            return []
        removed_indexes: set[int] = pdf_cleaning.deterministic_pdf_remove_indexes(entries)
        protected_indexes = pdf_cleaning.protected_pdf_vocabulary_indexes(entries)
        batches = self._pdf_clean_batches(entries)
        total_batches = len(batches)
        worker_count = min(total_batches, self._parallel_workers())
        slots: Queue[int] = Queue()
        for slot in range(worker_count):
            slots.put(slot)

        next_batch = 0
        completed = 0
        active = {}
        batch_errors: list[str] = []

        def submit_next(executor: ThreadPoolExecutor) -> bool:
            nonlocal next_batch
            if next_batch >= total_batches:
                return False
            batch = batches[next_batch]
            batch_number = next_batch + 1
            next_batch += 1
            future = executor.submit(self._clean_pdf_batch_with_slot, slots, batch, language)
            active[future] = batch_number
            return True

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for _ in range(worker_count):
                if not submit_next(executor):
                    break

            while active:
                finished, _pending = wait(set(active), return_when=FIRST_COMPLETED)
                for future in finished:
                    batch_number = active.pop(future)
                    try:
                        removed_indexes.update(future.result() - protected_indexes)
                    except Exception as error:
                        message = f"AI 审阅 CSV：第 {batch_number}/{total_batches} 批失败，已保留该批原始词条。{error}"
                        batch_errors.append(message)
                        LOGGER.exception(message)
                    completed += 1
                    if progress_callback is not None:
                        percent = 40 + int(completed / max(1, total_batches) * 40)
                        progress_callback(f"AI 审阅 CSV：已完成 {completed}/{total_batches}", percent)
                while len(active) < worker_count and submit_next(executor):
                    pass
        if progress_callback is not None:
            if batch_errors:
                progress_callback(f"AI 审阅 CSV 完成，{len(batch_errors)} 批失败并已保留", 80)
            else:
                progress_callback("AI 审阅 CSV 完成", 80)
        return [
            WordEntry(
                word=entry.word,
                meaning=entry.meaning,
                source_index=index,
                frequency=entry.frequency,
                forms=entry.forms,
                example_sentence=entry.example_sentence,
                example_sentence_cn=entry.example_sentence_cn,
            )
            for index, entry in enumerate(
                (entry for entry in entries if entry.source_index not in removed_indexes),
                start=1,
            )
        ]

    def _clean_pdf_batch_with_slot(
        self,
        slots: Queue[int],
        batch: list[WordEntry],
        language: str,
    ) -> set[int]:
        slot = slots.get()
        try:
            llm = self._load_model(slot)
            prompt = self._build_pdf_vocabulary_cleaning_prompt(batch, language)
            content = self._call_model(
                llm,
                prompt,
                system_prompt="You review generated vocabulary CSV rows. Return only valid JSON.",
                max_tokens=256,
            )
            return self._parse_pdf_vocabulary_cleaning_response(content, batch)
        finally:
            slots.put(slot)

    @classmethod
    def _pdf_clean_batches(cls, entries: list[WordEntry]) -> list[list[WordEntry]]:
        return pdf_cleaning.pdf_clean_batches(
            entries,
            cls._pdf_clean_batch_size(),
            cls._pdf_clean_prompt_char_budget(),
        )

    @staticmethod
    def _pdf_clean_batch_size() -> int:
        return pdf_cleaning.pdf_clean_batch_size(LocalLlmExampleGenerator._auto_pdf_clean_batch_size())

    @staticmethod
    def _pdf_clean_prompt_char_budget() -> int:
        return pdf_cleaning.pdf_clean_prompt_char_budget(
            LocalLlmExampleGenerator._auto_pdf_clean_prompt_char_budget()
        )

    @classmethod
    def _auto_pdf_clean_batch_size(cls) -> int:
        context = cls._context_size()
        memory_mb = cls._system_capacity_mb()
        cpu_count = os.cpu_count() or 1
        if context >= 16384:
            rows = 100
        elif context >= 8192:
            rows = 80
        elif context >= 4096:
            rows = 50
        else:
            rows = 25
        if memory_mb < 6000:
            rows = min(rows, 25)
        elif memory_mb < 10000:
            rows = min(rows, 40)
        if cpu_count <= 4:
            rows = min(rows, 35)
        return max(10, min(100, rows))

    @classmethod
    def _auto_pdf_clean_prompt_char_budget(cls) -> int:
        context = cls._context_size()
        if context >= 32768:
            budget = 12000
        elif context >= 16384:
            budget = 10000
        elif context >= 8192:
            budget = 7500
        elif context >= 4096:
            budget = 4500
        else:
            budget = 2500
        if cls._system_capacity_mb() < 6000:
            budget = min(budget, 2500)
        return max(1000, min(12000, budget))

    @staticmethod
    def _csv_review_row(entry: WordEntry) -> list[int | str]:
        return pdf_cleaning.csv_review_row(entry)

    @staticmethod
    def _truncate_for_prompt(value: str, limit: int) -> str:
        return pdf_cleaning.truncate_for_prompt(value, limit)

    @staticmethod
    def _build_pdf_vocabulary_cleaning_prompt(entries: list[WordEntry], language: str) -> str:
        return pdf_cleaning.build_pdf_vocabulary_cleaning_prompt(entries, language)

    @staticmethod
    def _parse_pdf_vocabulary_cleaning_response(content: str, batch: list[WordEntry]) -> set[int]:
        return pdf_cleaning.parse_pdf_vocabulary_cleaning_response(content, batch)

    @staticmethod
    def _parse_pdf_cleaning_numbers(content: str, batch: list[WordEntry]) -> set[int]:
        return pdf_cleaning.parse_pdf_cleaning_numbers(content, batch)

    @staticmethod
    def _coerce_pdf_clean_index(
        value: Any,
        batch: list[WordEntry],
        valid_indexes: set[int],
    ) -> int | None:
        return pdf_cleaning.coerce_pdf_clean_index(value, batch, valid_indexes)

    @staticmethod
    def _call_model(
        llm: Any,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 200,
    ) -> str:
        return llama_runtime.call_model(llm, prompt, system_prompt, max_tokens)

    def _run_smoke_test(self) -> None:
        llm = self._load_model(0)
        content = self._call_smoke_model(llm)
        try:
            data = json.loads(content)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"模型最小执行测试返回内容不是有效 JSON：{content}") from error
        if not isinstance(data, dict) or data.get("ok") is not True:
            raise RuntimeError(f"模型最小执行测试返回异常：{content}")

    @staticmethod
    def _call_smoke_model(llm: Any) -> str:
        return llama_runtime.call_smoke_model(llm)

    @staticmethod
    def _extract_chat_content(response: Any) -> str:
        return llama_runtime.extract_chat_content(response)

    @staticmethod
    def _extract_completion_text(response: Any) -> str:
        return llama_runtime.extract_completion_text(response)

    @staticmethod
    def _build_completion_prompt(prompt: str, system_prompt: str | None = None) -> str:
        return llama_runtime.build_completion_prompt(prompt, system_prompt)

    def _find_existing_model_path(self) -> Path | None:
        return model_store.find_existing_model_path(self._model_dir, self.DEFAULT_MODEL_FILENAME)

    def _ensure_model_path(self) -> Path | None:
        return model_store.ensure_model_path(
            self._model_dir,
            self.DEFAULT_MODEL_FILENAME,
            self.DEFAULT_MODEL_REPO,
            self.DEFAULT_MODEL_URL,
        )

    def _download_default_model(self) -> Path:
        return model_store.download_default_model(
            self._model_dir,
            self.DEFAULT_MODEL_FILENAME,
            self.DEFAULT_MODEL_REPO,
            self.DEFAULT_MODEL_URL,
        )

    @staticmethod
    def _acquire_download_lock(lock_path: Path, target_path: Path) -> int | None:
        return model_store.acquire_download_lock(lock_path, target_path)

    @staticmethod
    def _build_prompt(entry: WordEntry, scope: str = "") -> str:
        return prompts.build_prompt(entry, scope)

    @staticmethod
    def _build_batch_prompt(entries: list[WordEntry], scope: str = "") -> str:
        return prompts.build_batch_prompt(entries, scope)

    @staticmethod
    def _build_correction_prompt(entry: WordEntry, scope: str = "") -> str:
        return prompts.build_correction_prompt(entry, scope)

    @staticmethod
    def _build_explanation_prompt(entry: WordEntry, scope: str = "", language: str = "") -> str:
        return prompts.build_explanation_prompt(entry, scope, language)

    @staticmethod
    def _build_scope_text(scope: str) -> str:
        return prompts.build_scope_text(scope)

    @staticmethod
    def _normalize_english_term(text: str) -> str:
        return prompts.normalize_english_term(text)

    @staticmethod
    def _parse_response(content: str, require_meaning: bool = False) -> GeneratedExample:
        return prompts.parse_response(content, require_meaning)

    @staticmethod
    def _parse_batch_response(content: str, entries: list[WordEntry]) -> list[tuple[WordEntry, GeneratedExample]]:
        return prompts.parse_batch_response(content, entries)

    @staticmethod
    def _batch_max_tokens(entry_count: int) -> int:
        return max(512, min(4096, 220 * max(1, entry_count)))

    @staticmethod
    def _explanation_max_tokens() -> int:
        raw_value = os.getenv("WORDPYCKET_LLM_EXPLAIN_TOKENS")
        if raw_value is not None:
            try:
                return max(256, min(4096, int(raw_value)))
            except ValueError as error:
                raise RuntimeError("WORDPYCKET_LLM_EXPLAIN_TOKENS 必须是整数。") from error
        return 768

    @staticmethod
    def _parse_correction_response(
        content: str,
        original: WordEntry,
    ) -> GeneratedCorrection:
        return prompts.parse_correction_response(content, original)

    @staticmethod
    def _parse_explanation_response(content: str) -> GeneratedExplanation:
        return prompts.parse_explanation_response(content)


def _main() -> None:
    payload = json.loads(sys.stdin.read())
    entry_data = payload["entry"]
    entry = WordEntry(
        word=entry_data["word"],
        meaning=entry_data["meaning"],
        source_index=int(entry_data.get("source_index", 0)),
        frequency=int(entry_data.get("frequency", 0)),
        forms=str(entry_data.get("forms", "")),
        example_sentence=str(entry_data.get("example_sentence", "")),
        example_sentence_cn=str(entry_data.get("example_sentence_cn", "")),
    )
    generator = LocalLlmExampleGenerator(Path(payload["model_dir"]))
    action = payload["action"]
    scope = str(payload.get("scope", ""))
    language = str(payload.get("language", ""))
    if action == "generate":
        generated = generator.generate(entry, scope)
        result = {
            "example_sentence": generated.example_sentence,
            "example_sentence_cn": generated.example_sentence_cn,
        }
        if generated.meaning:
            result["meaning"] = generated.meaning
    elif action == "correct":
        corrected = generator.correct_entry(entry, scope)
        result = {
            "corrected_word": corrected.corrected_word,
            "note": corrected.note,
        }
    elif action == "explain":
        explained = generator.explain_entry(entry, scope, language)
        result = {"explanation": explained.explanation}
    elif action == "smoke_test":
        generator._run_smoke_test()
        result = {"ok": True}
    else:
        raise RuntimeError(f"未知智能任务：{action}")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    _main()
