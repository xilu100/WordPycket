from __future__ import annotations

import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path


def find_existing_model_path(model_dir: Path, default_filename: str) -> Path | None:
    if not model_dir.exists():
        return None
    models = sorted(
        model_dir.glob("*.gguf"),
        key=lambda path: (
            path.name != default_filename,
            -path.stat().st_size,
        ),
    )
    if len(models) > 1:
        names = "、".join(path.name for path in models)
        raise RuntimeError(f"model 目录中只能存在一个 .gguf 模型文件。当前存在：{names}")
    return models[0] if models else None


def ensure_model_path(model_dir: Path, default_filename: str, default_repo: str, default_url: str) -> Path:
    model_path = find_existing_model_path(model_dir, default_filename)
    if model_path is not None:
        return model_path
    return download_default_model(model_dir, default_filename, default_repo, default_url)


def download_default_model(model_dir: Path, default_filename: str, default_repo: str, default_url: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    target_path = model_dir / default_filename
    lock_path = model_dir / f"{default_filename}.lock"
    partial_path = model_dir / f"{default_filename}.part"

    lock_fd = acquire_download_lock(lock_path, target_path)
    if lock_fd is None:
        return target_path

    try:
        if target_path.exists():
            return target_path
        if partial_path.exists():
            partial_path.unlink()
        print(
            f"正在从 Hugging Face 下载默认模型 {default_repo}/{default_filename} ...",
            file=sys.stderr,
            flush=True,
        )
        with urllib.request.urlopen(default_url, timeout=60) as response:
            with partial_path.open("wb") as file:
                shutil.copyfileobj(response, file, length=1024 * 1024)
        partial_path.replace(target_path)
        return target_path
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def acquire_download_lock(lock_path: Path, target_path: Path) -> int | None:
    deadline = time.monotonic() + 3600
    while True:
        if target_path.exists():
            return None
        try:
            return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError("等待默认模型下载超时。")
            time.sleep(1)
