from __future__ import annotations

import argparse
import concurrent.futures
import inspect
import json
import queue
import sys
import threading
import uuid
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from llmserver.engine import LocalLlmExampleGenerator, WordEntry


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


def _entry_to_payload(entry: WordEntry) -> dict[str, Any]:
    return {
        "word": entry.word,
        "meaning": entry.meaning,
        "source_index": entry.source_index,
        "frequency": entry.frequency,
        "forms": entry.forms,
        "example_sentence": entry.example_sentence,
        "example_sentence_cn": entry.example_sentence_cn,
    }


def _call_with_optional_language(method: Any, *args: Any, language: str = "") -> Any:
    signature = inspect.signature(method)
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values()):
        return method(*args, language)
    positional_count = sum(
        1
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    )
    if positional_count > len(args):
        return method(*args, language)
    return method(*args)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


class JobStore:
    def __init__(self, max_workers: int) -> None:
        self._max_workers = 1
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        self._slots: queue.LifoQueue[int] = queue.LifoQueue()
        for slot in reversed(range(self._max_workers)):
            self._slots.put(slot)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def submit(self, target: Callable[[Callable[[str, int], None], int], Any]) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "state": "queued",
                "stage": "queued",
                "progress": {"message": "任务已排队", "percent": 0},
                "result": None,
                "error": "",
            }
        self._executor.submit(self._run, job_id, target)
        return job_id

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise RuntimeError(f"未知 AI 任务：{job_id}")
            return _jsonable(dict(job))

    def _run(self, job_id: str, target: Callable[[Callable[[str, int], None], int], Any]) -> None:
        self._update(job_id, state="running", stage="running", progress={"message": "AI 正在处理", "percent": 1})

        def progress(message: str, percent: int) -> None:
            self._update(
                job_id,
                stage="progress",
                progress={"message": message, "percent": max(0, min(100, int(percent)))},
            )

        slot = self._slots.get()
        try:
            result = target(progress, slot)
        except Exception as error:
            self._update(
                job_id,
                state="failed",
                stage="failed",
                progress={"message": "AI 任务失败", "percent": 100},
                error=str(error),
            )
            return
        finally:
            self._slots.put(slot)
        self._update(
            job_id,
            state="completed",
            stage="completed",
            progress={"message": "AI 任务完成", "percent": 100},
            result=_jsonable(result),
        )

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(changes)


class LlmRpcHandler(BaseHTTPRequestHandler):
    generator: LocalLlmExampleGenerator
    jobs: JobStore

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json({"ok": True})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/rpc":
            self.send_error(404)
            return
        try:
            payload = self._read_json()
            result = self._dispatch(payload)
            self._write_json({"ok": True, "result": _jsonable(result)})
        except Exception as error:
            self._write_json({"ok": False, "error": str(error)}, status=500)

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length).decode("utf-8")
        payload = json.loads(data or "{}")
        if not isinstance(payload, dict):
            raise RuntimeError("RPC payload must be a JSON object.")
        return payload

    def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _dispatch(self, payload: dict[str, Any]) -> Any:
        method = str(payload.get("method", ""))
        params = payload.get("params", {})
        if not isinstance(params, dict):
            raise RuntimeError("RPC params must be a JSON object.")

        if method == "submit_job":
            target_method = str(params.get("method", ""))
            target_params = params.get("params", {})
            if not isinstance(target_params, dict):
                raise RuntimeError("Job params must be a JSON object.")
            job_id = self.jobs.submit(
                lambda progress, slot: self._dispatch_method(target_method, target_params, progress, slot)
            )
            return {"job_id": job_id, "state": "queued"}
        if method == "job_status":
            return self.jobs.status(str(params.get("job_id", "")))
        return self._dispatch_method(method, params, None)

    def _dispatch_method(
        self,
        method: str,
        params: dict[str, Any],
        progress_callback: Callable[[str, int], None] | None,
        slot: int = 0,
    ) -> Any:
        if method == "generate":
            entry = _entry_from_payload(params["entry"])
            return _call_with_optional_language(
                self.generator._generate_with_slot,
                slot,
                entry,
                str(params.get("scope", "")),
                language=str(params.get("language", "")),
            )[1]
        if method == "generate_batch":
            entries = [_entry_from_payload(item) for item in params.get("entries", [])]
            generated = _call_with_optional_language(
                self.generator._generate_batch_with_slot,
                0,
                entries,
                str(params.get("scope", "")),
                language=str(params.get("language", "")),
            )
            return [
                {
                    "entry": _entry_to_payload(item_entry),
                    "example_sentence": example.example_sentence,
                    "example_sentence_cn": example.example_sentence_cn,
                    "meaning": example.meaning,
                }
                for item_entry, example in generated
            ]
        if method == "correct_entry":
            entry = _entry_from_payload(params["entry"])
            return _call_with_optional_language(
                self.generator._correct_with_slot,
                slot,
                entry,
                str(params.get("scope", "")),
                language=str(params.get("language", "")),
            )[1]
        if method == "explain_entry":
            return self.generator._explain_with_slot(
                slot,
                _entry_from_payload(params["entry"]),
                str(params.get("scope", "")),
                str(params.get("language", "")),
            )
        if method == "run_action":
            return self._run_action(params, slot)
        if method == "is_available":
            return self.generator.is_available()
        if method == "uses_user_model":
            return self.generator.uses_user_model()
        if method == "model_status":
            return self.generator.model_status()
        if method == "ensure_model_available":
            return self.generator.ensure_model_available()
        if method == "device_status":
            return self.generator.device_status()
        if method == "check_model_runtime":
            return self.generator.check_model_runtime()
        if method == "recommended_process_parallelism":
            return self.generator.recommended_process_parallelism()
        if method == "recommended_supplement_strategy":
            return self.generator.recommended_supplement_strategy()
        if method == "clean_pdf_vocabulary_entries":
            entries = [_entry_from_payload(item) for item in params.get("entries", [])]
            cleaned = self.generator.clean_pdf_vocabulary_entries(
                entries,
                str(params.get("language", "")),
                progress_callback=progress_callback,
            )
            return [_entry_to_payload(entry) for entry in cleaned]
        raise RuntimeError(f"未知 AI 方法：{method}")

    def _run_action(self, params: dict[str, Any], slot: int = 0) -> dict[str, Any]:
        action = str(params.get("action", ""))
        scope = str(params.get("scope", ""))
        language = str(params.get("language", ""))
        if action == "generate_batch":
            entries = [_entry_from_payload(item) for item in params.get("entries", [])]
            generated = _call_with_optional_language(
                self.generator._generate_batch_with_slot,
                0,
                entries,
                scope,
                language=language,
            )
            return {
                "items": [
                    {
                        "entry": _entry_to_payload(item_entry),
                        "example_sentence": example.example_sentence,
                        "example_sentence_cn": example.example_sentence_cn,
                        "meaning": example.meaning,
                    }
                    for item_entry, example in generated
                ]
            }
        entry = _entry_from_payload(params["entry"])
        if action == "generate":
            generated = _call_with_optional_language(
                self.generator._generate_with_slot,
                slot,
                entry,
                scope,
                language=language,
            )[1]
            result = {
                "example_sentence": generated.example_sentence,
                "example_sentence_cn": generated.example_sentence_cn,
            }
            if generated.meaning:
                result["meaning"] = generated.meaning
            return result
        if action == "correct":
            corrected = _call_with_optional_language(
                self.generator._correct_with_slot,
                slot,
                entry,
                scope,
                language=language,
            )[1]
            return {
                "corrected_word": corrected.corrected_word,
                "note": corrected.note,
                "should_update": corrected.should_update,
            }
        if action == "explain":
            explained = self.generator._explain_with_slot(slot, entry, scope, language)
            return {"explanation": explained.explanation}
        if action == "smoke_test":
            self.generator._run_smoke_test()
            return {"ok": True}
        raise RuntimeError(f"未知 AI 任务：{action}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    LlmRpcHandler.generator = LocalLlmExampleGenerator(Path(args.model_dir))
    LlmRpcHandler.jobs = JobStore(LlmRpcHandler.generator.recommended_process_parallelism())
    server = ThreadingHTTPServer((args.host, args.port), LlmRpcHandler)
    status = LlmRpcHandler.generator.model_status()
    print(
        json.dumps(
            {
                "type": "ready",
                "host": server.server_address[0],
                "port": server.server_address[1],
                "model_available": status.path is not None,
                "model_path": str(status.path) if status.path is not None else None,
                "is_user_model": status.is_user_model,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
