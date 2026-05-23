from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import venv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = PROJECT_ROOT / ".venv"
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
}


def main() -> int:
    args = parse_args()
    venv_path = args.venv.resolve()
    configure_local_runtime()
    create_venv(venv_path)
    python = venv_python(venv_path)

    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    install_base_dependencies(python)
    install_llama_cpp(python, select_device(args.device), args.strict_accel)
    install_nltk_data(python)
    install_spacy_models(python, args.spacy_models)
    install_project(python)
    print_activation_help(venv_path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and configure a local WordPycket virtual environment.")
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV, help="Virtual environment path. Default: .venv")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="llama.cpp acceleration target. Default: auto",
    )
    parser.add_argument(
        "--strict-accel",
        action="store_true",
        help="Fail instead of falling back to CPU when CUDA/Metal llama-cpp-python build fails.",
    )
    parser.add_argument(
        "--spacy-models",
        default="all",
        help="Comma-separated spaCy model languages to install, e.g. en,de. Default: all",
    )
    return parser.parse_args()


def create_venv(venv_path: Path) -> None:
    python = venv_python(venv_path)
    if python.exists():
        print(f"Using existing virtual environment: {venv_path}")
        ensure_pip(python)
        return
    print(f"Creating virtual environment: {venv_path}")
    venv.EnvBuilder(with_pip=True, clear=False).create(venv_path)
    ensure_pip(venv_python(venv_path))


def ensure_pip(python: Path) -> None:
    result = subprocess.run([str(python), "-m", "pip", "--version"], cwd=PROJECT_ROOT, check=False)
    if result.returncode == 0:
        return
    run([str(python), "-m", "ensurepip", "--upgrade"])


def configure_local_runtime() -> None:
    runtime_dir = PROJECT_ROOT / "runtime"
    cache_dir = runtime_dir / "cache"
    nltk_data = runtime_dir / "nltk_data"
    for path in (cache_dir / "pip", cache_dir / "huggingface", nltk_data):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["PIP_CACHE_DIR"] = str(cache_dir / "pip")
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    os.environ["NLTK_DATA"] = str(nltk_data)


def venv_python(venv_path: Path) -> Path:
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def install_base_dependencies(python: Path) -> None:
    requirement_lines = [dependency for dependency in project_dependencies() if "llama-cpp-python" not in dependency]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as file:
        file.write("\n".join(requirement_lines))
        filtered_requirements = Path(file.name)
    try:
        run([str(python), "-m", "pip", "install", "-r", str(filtered_requirements)])
    finally:
        try:
            filtered_requirements.unlink()
        except FileNotFoundError:
            pass


def install_project(python: Path) -> None:
    run([str(python), "-m", "pip", "install", "--no-deps", "-e", str(PROJECT_ROOT)])


def project_dependencies() -> list[str]:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject.get("project", {}).get("dependencies", [])
    return [dependency for dependency in dependencies if isinstance(dependency, str)]


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if has_cuda_device():
        return "cuda"
    if has_mps_device():
        return "mps"
    return "cpu"


def has_cuda_device() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    result = subprocess.run([nvidia_smi, "-L"], capture_output=True, text=True, check=False)
    return result.returncode == 0 and "GPU" in result.stdout


def cuda_wheel_tags() -> list[str]:
    detected = detected_cuda_version()
    supported = ["cu129", "cu128", "cu126", "cu125", "cu124", "cu123", "cu122", "cu121", "cu120"]
    if detected is None:
        return supported
    major, minor = detected
    detected_tag = f"cu{major}{minor}"
    if detected_tag in supported:
        start = supported.index(detected_tag)
        return supported[start:] + supported[:start]
    compatible = [tag for tag in supported if int(tag[2:]) <= major * 10 + minor]
    return compatible or supported


def detected_cuda_version() -> tuple[int, int] | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    result = subprocess.run(
        [nvidia_smi],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def has_mps_device() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "arm"}


def install_llama_cpp(python: Path, device: str, strict_accel: bool) -> None:
    if device == "cpu":
        print("Installing CPU llama-cpp-python wheel.")
        install_llama_cpu(python)
        return

    if device == "cuda":
        if not has_cuda_device():
            message = "CUDA requested, but no NVIDIA GPU/driver was detected. A project venv cannot provide CUDA without hardware and drivers."
            if strict_accel:
                raise RuntimeError(message)
            print(f"{message} Falling back to CPU.")
            install_llama_cpu(python)
            return
        if install_llama_cuda_prebuilt(python):
            return
        message = "No compatible prebuilt CUDA llama-cpp-python wheel was installed."
        if strict_accel:
            raise RuntimeError(message)
        print(f"{message} Falling back to CPU.")
        install_llama_cpu(python)
        return
    elif device == "mps":
        if not has_mps_device():
            message = "Metal/MPS requested, but this is not an Apple Silicon macOS system."
            if strict_accel:
                raise RuntimeError(message)
            print(f"{message} Falling back to CPU.")
            install_llama_cpu(python)
            return
        if install_llama_metal_prebuilt(python):
            return
        message = "No compatible prebuilt Metal llama-cpp-python wheel was installed."
        if strict_accel:
            raise RuntimeError(message)
        print(f"{message} Falling back to CPU.")
        install_llama_cpu(python)
        return
    else:
        raise RuntimeError(f"Unknown device target: {device}")


def install_llama_cpu(python: Path) -> None:
    run([str(python), "-m", "pip", "install", "--upgrade", "--force-reinstall", "llama-cpp-python>=0.3.0"])


def install_llama_cuda_prebuilt(python: Path) -> bool:
    for tag in cuda_wheel_tags():
        wheel_index = f"https://abetlen.github.io/llama-cpp-python/whl/{tag}"
        command = llama_prebuilt_command(python, wheel_index)
        print(f"Trying prebuilt CUDA llama-cpp-python wheel: {tag}")
        if run_optional(command):
            return True
    return False


def install_llama_metal_prebuilt(python: Path) -> bool:
    wheel_index = "https://abetlen.github.io/llama-cpp-python/whl/metal"
    print("Trying prebuilt Metal llama-cpp-python wheel.")
    return run_optional(llama_prebuilt_command(python, wheel_index))


def llama_prebuilt_command(python: Path, wheel_index: str) -> list[str]:
    return [
        str(python),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-cache-dir",
        "--prefer-binary",
        "--only-binary",
        "llama-cpp-python",
        "--extra-index-url",
        wheel_index,
        "llama-cpp-python>=0.3.0",
    ]


def install_nltk_data(python: Path) -> None:
    nltk_data = PROJECT_ROOT / "runtime" / "nltk_data"
    run([str(python), "-m", "nltk.downloader", "-d", str(nltk_data), "wordnet", "omw-1.4"])


def install_spacy_models(python: Path, model_selection: str) -> None:
    if model_selection.strip().lower() == "none":
        return
    if model_selection.strip().lower() == "all":
        languages = list(SPACY_MODELS)
    else:
        languages = [item.strip().lower() for item in model_selection.split(",") if item.strip()]

    for language in languages:
        model = SPACY_MODELS.get(language)
        if model is None:
            raise RuntimeError(f"Unsupported spaCy model language: {language}")
        run([str(python), "-m", "spacy", "download", model])


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def run_optional(command: list[str]) -> bool:
    print(f"$ {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return result.returncode == 0


def print_activation_help(venv_path: Path) -> None:
    if platform.system() == "Windows":
        activate = venv_path / "Scripts" / "Activate.ps1"
        python = venv_path / "Scripts" / "python.exe"
    else:
        activate = venv_path / "bin" / "activate"
        python = venv_path / "bin" / "python"
    print("\nEnvironment ready.")
    print(f"Activate: {activate}")
    print(f"Run: {python} -m wordpycket.main")


if __name__ == "__main__":
    raise SystemExit(main())
