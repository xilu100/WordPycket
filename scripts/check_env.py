from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SPACY_MODELS = (
    "en_core_web_sm",
    "de_core_news_sm",
    "fr_core_news_sm",
    "es_core_news_sm",
    "it_core_news_sm",
    "pt_core_news_sm",
    "nl_core_news_sm",
)


def main() -> int:
    configure_runtime()
    failures: list[str] = []
    failures.extend(check_project_dependencies())
    failures.extend(check_imports())
    failures.extend(check_nltk_data())
    failures.extend(check_spacy_models())

    if failures:
        print("Environment check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Environment check passed.")
    return 0


def configure_runtime() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    runtime_dir = PROJECT_ROOT / "runtime"
    cache_dir = runtime_dir / "cache"
    nltk_data = runtime_dir / "nltk_data"
    for path in (cache_dir / "pip", cache_dir / "huggingface", nltk_data):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PIP_CACHE_DIR", str(cache_dir / "pip"))
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("NLTK_DATA", str(nltk_data))


def check_project_dependencies() -> list[str]:
    try:
        from pip._vendor.packaging.requirements import Requirement
    except Exception as error:
        return [f"pip packaging support is unavailable: {error}"]

    failures = []
    for dependency in project_dependencies():
        try:
            requirement = Requirement(dependency)
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"Missing Python package: {dependency}")
            continue
        except Exception as error:
            failures.append(f"Could not inspect dependency {dependency}: {error}")
            continue
        if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
            failures.append(f"Package {requirement.name} has {version}, expected {requirement.specifier}")
    return failures


def project_dependencies() -> list[str]:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject.get("project", {}).get("dependencies", [])
    return [dependency for dependency in dependencies if isinstance(dependency, str)]


def check_imports() -> list[str]:
    modules = (
        "argostranslate",
        "wordpycket",
        "llmserver",
        "llama_cpp",
        "nltk",
        "pandas",
        "PySide6",
        "regex",
        "spacy",
        "wordfreq",
    )
    failures = []
    for module in modules:
        if importlib.util.find_spec(module) is None:
            failures.append(f"Could not import module: {module}")
    return failures


def check_nltk_data() -> list[str]:
    try:
        import nltk.data
    except Exception as error:
        return [f"Could not import nltk: {error}"]

    failures = []
    for resource in ("corpora/wordnet", "corpora/omw-1.4"):
        try:
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.data.find(f"{resource}.zip")
        except LookupError:
            failures.append(f"Missing NLTK data: {resource}")
    return failures


def check_spacy_models() -> list[str]:
    try:
        import spacy.util
    except Exception as error:
        return [f"Could not import spaCy: {error}"]

    failures = []
    for model in SPACY_MODELS:
        if not spacy.util.is_package(model):
            failures.append(f"Missing spaCy model: {model}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
