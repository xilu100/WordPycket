from __future__ import annotations

import argparse
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACY_MODEL_PACKAGES = (
    "en_core_web_sm",
    "de_core_news_sm",
)


def main() -> int:
    args = parse_args()
    ensure_pyinstaller()
    command = pyinstaller_command(args)
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"\nPortable build ready: {PROJECT_ROOT / 'dist' / args.name}")
    print("Distribute the whole folder, not only the executable.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portable WordPycket desktop folder.")
    parser.add_argument("--name", default="run", help="Executable/folder name. Default: run")
    parser.add_argument("--include-model", action="store_true", help="Bundle the local model/ directory if it exists.")
    parser.add_argument("--include-input", action="store_true", help="Bundle the local input/ directory if it exists.")
    parser.add_argument("--clean", action="store_true", help="Pass --clean to PyInstaller.")
    return parser.parse_args()


def ensure_pyinstaller() -> None:
    if importlib.util.find_spec("PyInstaller") is not None:
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller>=6.0"], check=True)


def pyinstaller_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name",
        args.name,
        "--paths",
        str(PROJECT_ROOT / "src"),
        "--hidden-import",
        "PySide6.QtPdf",
        "--collect-all",
        "PySide6",
        "--collect-all",
        "spacy",
        "--collect-all",
        "nltk",
        "--collect-all",
        "llama_cpp",
    ]
    if args.clean:
        command.append("--clean")

    for package in SPACY_MODEL_PACKAGES:
        if importlib.util.find_spec(package) is not None:
            command.extend(["--collect-all", package])

    add_data(command, PROJECT_ROOT / "README.md", ".")
    if args.include_input:
        add_data(command, PROJECT_ROOT / "input", "input")
    if args.include_model:
        add_data(command, PROJECT_ROOT / "model", "model")

    command.append(str(PROJECT_ROOT / "src" / "wordpycket" / "main.py"))
    return command


def add_data(command: list[str], source: Path, target: str) -> None:
    if not source.exists():
        return
    separator = ";" if platform.system() == "Windows" else ":"
    command.extend(["--add-data", f"{source}{separator}{target}"])


if __name__ == "__main__":
    raise SystemExit(main())
