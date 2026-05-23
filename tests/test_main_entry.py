from __future__ import annotations

from wordpycket import main


def test_main_entry_uses_qt_gui_by_default() -> None:
    assert main.WordPycketApp.__module__ == "wordpycket.presentation.qt_app"
