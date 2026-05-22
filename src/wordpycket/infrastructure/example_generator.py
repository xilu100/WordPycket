from __future__ import annotations

import gc
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from queue import Queue
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from wordpycket.domain.entities import WordEntry


ProgressCallback = Any
ControlCallback = Any


def _is_llama_cleanup_error(unraisable: Any) -> bool:
    return (
        unraisable.exc_type is AttributeError
        and "'LlamaModel' object has no attribute 'sampler'"
        in str(unraisable.exc_value)
    )


def _install_llama_cleanup_error_filter() -> None:
    current_hook = sys.unraisablehook
    if getattr(current_hook, "_wordpycket_llama_filter", False):
        return

    def filter_llama_cleanup_error(unraisable: Any) -> None:
        if not _is_llama_cleanup_error(unraisable):
            current_hook(unraisable)

    filter_llama_cleanup_error._wordpycket_llama_filter = True  # type: ignore[attr-defined]
    sys.unraisablehook = filter_llama_cleanup_error


_install_llama_cleanup_error_filter()


@dataclass(frozen=True)
class GeneratedExample:
    example_sentence: str
    example_sentence_cn: str


@dataclass(frozen=True)
class GeneratedCorrection:
    corrected_word: str
    note: str = ""


class LocalLlmExampleGenerator:
    _AUTO_DEVICE = "auto"
    _CPU_DEVICE = "cpu"
    _CUDA_DEVICE = "cuda"
    _MPS_DEVICE = "mps"

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._llms: list[Any | None] = []
        self._llm_lock = threading.Lock()

    def generate(self, entry: WordEntry, scope: str = "") -> GeneratedExample:
        llm = self._load_model(0)
        prompt = self._build_prompt(entry, scope)
        content = self._call_model(llm, prompt)
        return self._parse_response(content)

    def correct_entry(self, entry: WordEntry, scope: str = "") -> GeneratedCorrection:
        llm = self._load_model(0)
        prompt = self._build_correction_prompt(entry, scope)
        content = self._call_model(llm, prompt)
        return self._parse_correction_response(content, entry)

    def generate_many(
        self,
        entries: list[WordEntry],
        scope: str = "",
        progress: ProgressCallback | None = None,
        control: ControlCallback | None = None,
    ) -> tuple[list[tuple[WordEntry, GeneratedExample]], list[str], int]:
        return self._run_parallel(entries, self._generate_with_slot, scope, progress, control)

    def correct_many(
        self,
        entries: list[WordEntry],
        scope: str = "",
        progress: ProgressCallback | None = None,
        control: ControlCallback | None = None,
    ) -> tuple[list[tuple[WordEntry, GeneratedCorrection]], list[str], int]:
        return self._run_parallel(entries, self._correct_with_slot, scope, progress, control)

    def is_available(self) -> bool:
        return self._find_model_path() is not None

    def _load_model(self, slot: int):
        with self._llm_lock:
            while len(self._llms) <= slot:
                self._llms.append(None)
            if self._llms[slot] is not None:
                return self._llms[slot]

            model_path = self._find_model_path()
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
                "n_ctx": 2048,
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
        return entry, self._parse_response(content)

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

        model_path = self._find_model_path()
        model_size_mb = self._model_size_mb(model_path)
        if self._detect_accelerator() == self._CUDA_DEVICE:
            free_vram_mb = self._cuda_free_vram_mb()
            if free_vram_mb is not None:
                per_worker_mb = max(2200, int(model_size_mb * 0.95) + 500)
                return max(1, min(5, free_vram_mb // per_worker_mb))
            return 1

        if self._has_mps_device():
            memory_mb = self._system_memory_mb()
            per_worker_mb = max(3200, int(model_size_mb * 1.8) + 1200)
            return max(1, min(3, memory_mb // per_worker_mb))

        memory_mb = self._system_memory_mb()
        cpu_count = os.cpu_count() or 1
        per_worker_mb = max(3200, int(model_size_mb * 1.8) + 1200)
        return max(1, min(3, cpu_count // self._threads_per_model(), memory_mb // per_worker_mb))

    @staticmethod
    def _threads_per_model() -> int:
        return min(4, os.cpu_count() or 1)

    @staticmethod
    def _model_size_mb(model_path: Path | None) -> int:
        if model_path is None:
            return 2048
        return max(1, model_path.stat().st_size // (1024 * 1024))

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

    @staticmethod
    def _gpu_layers() -> int:
        raw_value = os.getenv("WORDPYCKET_LLM_GPU_LAYERS")
        if raw_value is None:
            return -1
        try:
            return int(raw_value)
        except ValueError as error:
            raise RuntimeError("WORDPYCKET_LLM_GPU_LAYERS 必须是整数。") from error

    @staticmethod
    def _call_model(llm: Any, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You generate concise English vocabulary examples. "
                    "Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response = llm.create_chat_completion(
                messages=messages,
                temperature=0.4,
                max_tokens=160,
                response_format={"type": "json_object"},
            )
            return LocalLlmExampleGenerator._extract_chat_content(response)
        except (KeyError, TypeError, ValueError, AttributeError):
            response = llm.create_completion(
                prompt=LocalLlmExampleGenerator._build_completion_prompt(prompt),
                temperature=0.4,
                max_tokens=160,
                stop=["\n\n"],
            )
            return LocalLlmExampleGenerator._extract_completion_text(response)

    @staticmethod
    def _extract_chat_content(response: Any) -> str:
        choices = response["choices"]
        message = choices[0]["message"]
        content = message["content"]
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        if isinstance(content, list):
            return "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            )
        return str(content)

    @staticmethod
    def _extract_completion_text(response: Any) -> str:
        return str(response["choices"][0]["text"])

    @staticmethod
    def _build_completion_prompt(prompt: str) -> str:
        return (
            "You generate concise English vocabulary examples. "
            "Return only valid JSON.\n\n"
            f"{prompt}\n\nJSON:"
        )

    def _find_model_path(self) -> Path | None:
        if not self._model_dir.exists():
            return None

        models = sorted(
            self._model_dir.glob("*.gguf"),
            key=lambda path: path.stat().st_size,
            reverse=True,
        )
        return models[0] if models else None

    @staticmethod
    def _build_prompt(entry: WordEntry, scope: str = "") -> str:
        forms = f"\nWord forms: {entry.forms}" if entry.forms else ""
        scope_text = LocalLlmExampleGenerator._build_scope_text(scope)
        return (
            "Generate one natural, short English sentence for this vocabulary word, "
            "and provide a fluent Chinese translation.\n"
            f"{scope_text}"
            f"Word: {entry.word}\n"
            f"Chinese meaning: {entry.meaning}"
            f"{forms}\n"
            "Requirements:\n"
            "- The English sentence must include the word or a common form of it.\n"
            "- Interpret the word according to the scope above when choosing meanings and translations.\n"
            "- Keep the sentence under 16 English words.\n"
            "- Do not explain anything.\n"
            'Return JSON exactly like: {"example_sentence": "...", '
            '"example_sentence_cn": "..."}'
        )

    @staticmethod
    def _build_correction_prompt(entry: WordEntry, scope: str = "") -> str:
        normalized_word = LocalLlmExampleGenerator._normalize_english_term(entry.word)
        scope_text = LocalLlmExampleGenerator._build_scope_text(scope)
        return (
            "Correct this vocabulary record.\n"
            f"{scope_text}"
            f"Original English: {entry.word}\n"
            f"Normalized English candidate: {normalized_word}\n"
            "Requirements:\n"
            "- Fix English formatting errors such as underscores used as spaces.\n"
            "- If the original English contains underscores, replace them with spaces, not hyphens.\n"
            "- Keep proper compounds/hyphenation only when standard English requires them.\n"
            "- Do not correct, translate, validate, or comment on the Chinese meaning.\n"
            "- Do not correct, validate, or comment on word forms.\n"
            "- Do not explain anything outside JSON.\n"
            'Return JSON exactly like: {"corrected_word": "...", "note": "..."}'
        )

    @staticmethod
    def _build_scope_text(scope: str) -> str:
        cleaned = scope.strip()
        if not cleaned:
            return ""
        return (
            f"Scope/domain: {cleaned}\n"
            "Use this scope to resolve ambiguous English terms. "
            "For example, decide whether an English term should stay hyphenated, "
            "spaced, or joined based on this scope.\n"
        )

    @staticmethod
    def _normalize_english_term(text: str) -> str:
        normalized = re.sub(r"[_]+", " ", text.strip())
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    @staticmethod
    def _parse_response(content: str) -> GeneratedExample:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"模型返回内容不是有效 JSON：{content}") from error

        example_sentence = str(data.get("example_sentence", "")).strip()
        example_sentence_cn = str(data.get("example_sentence_cn", "")).strip()
        if not example_sentence or not example_sentence_cn:
            raise RuntimeError(f"模型返回缺少例句字段：{content}")

        return GeneratedExample(
            example_sentence=example_sentence,
            example_sentence_cn=example_sentence_cn,
        )

    @staticmethod
    def _parse_correction_response(
        content: str,
        original: WordEntry,
    ) -> GeneratedCorrection:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"模型返回内容不是有效 JSON：{content}") from error

        corrected_word = str(data.get("corrected_word", "")).strip()
        note = str(data.get("note", "")).strip()
        normalized_word = LocalLlmExampleGenerator._normalize_english_term(original.word)
        if "_" in original.word:
            corrected_word = normalized_word
        if not corrected_word:
            corrected_word = normalized_word

        return GeneratedCorrection(
            corrected_word=corrected_word,
            note=note,
        )
