from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = PROJECT_ROOT / ".venv"


def main() -> int:
    args = parse_args()
    run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "setup_env.py"),
            "--venv",
            str(args.venv),
            "--device",
            args.device,
            "--spacy-models",
            args.spacy_models,
        ]
        + (["--strict-accel"] if args.strict_accel else [])
    )
    python = venv_python(args.venv)
    build_command = [
        str(python),
        str(PROJECT_ROOT / "scripts" / "build_portable.py"),
        "--name",
        args.name,
        "--include-input",
        "--include-model",
    ]
    if args.clean:
        build_command.append("--clean")
    run(build_command)
    print(f"\nRelease folder: {PROJECT_ROOT / 'dist' / args.name}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/update the build venv and build the portable release folder.")
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV, help="Build virtual environment path. Default: .venv")
    parser.add_argument("--name", default="run", help="Portable executable/folder name. Default: run")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--strict-accel", action="store_true")
    parser.add_argument("--spacy-models", default="all")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def venv_python(venv_path: Path) -> Path:
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def run(command: list[str]) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
