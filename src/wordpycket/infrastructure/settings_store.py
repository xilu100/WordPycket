from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonSettingsStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def get_string(self, key: str, default: str = "") -> str:
        value = self._read().get(key, default)
        return value if isinstance(value, str) else default

    def set_string(self, key: str, value: str) -> None:
        data = self._read()
        data[key] = value
        self._write(data)

    def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
