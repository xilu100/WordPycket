from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "wordpycket"


def _imports_for(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_domain_does_not_depend_on_outer_layers() -> None:
    forbidden = ("wordpycket.application", "wordpycket.infrastructure", "wordpycket.presentation")
    for path in (SOURCE_ROOT / "domain").glob("*.py"):
        assert not any(module.startswith(forbidden) for module in _imports_for(path)), path


def test_application_does_not_depend_on_infrastructure_or_presentation() -> None:
    forbidden = ("wordpycket.infrastructure", "wordpycket.presentation")
    for path in (SOURCE_ROOT / "application").glob("*.py"):
        assert not any(module.startswith(forbidden) for module in _imports_for(path)), path


def test_inner_layers_do_not_import_qt() -> None:
    forbidden = ("PySide6", "PyQt6", "PyQt5")
    for layer in ("domain", "application"):
        for path in (SOURCE_ROOT / layer).glob("*.py"):
            assert not any(module.startswith(forbidden) for module in _imports_for(path)), path


def test_qt_app_keeps_workers_and_ports_out_of_window_module() -> None:
    source = (SOURCE_ROOT / "presentation" / "qt_app.py").read_text(encoding="utf-8")
    forbidden_definitions = (
        "class ExampleGenerator",
        "class CsvImportResult",
        "class PdfImportResult",
        "class CsvDatasetResult",
        "class BatchWorker",
        "class PdfImportWorker",
        "class BackgroundTaskWorker",
        "class UiThreadBridge",
    )
    assert not any(definition in source for definition in forbidden_definitions)
