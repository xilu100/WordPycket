from __future__ import annotations

import re
import threading
from functools import lru_cache

from wordpycket.domain.entities import WordEntry


class ArgosMeaningTranslator:
    """Local word/phrase translator backed by Argos Translate."""

    _TARGET_CODE = "zh"

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def translate_meaning(self, entry: WordEntry, scope: str = "", language: str = "") -> str:
        source_code = self._source_code(language)
        text = self._translation_text(entry)
        with self._lock:
            return self._translate_cached(source_code, self._TARGET_CODE, text).strip()

    @classmethod
    def _source_code(cls, language: str) -> str:
        normalized = language.strip().casefold()
        if normalized in {"德语", "german", "de", "deutsch"}:
            return "de"
        if normalized in {"法语", "french", "fr", "francais", "français"}:
            return "fr"
        if normalized in {"西班牙语", "spanish", "es", "espanol", "español"}:
            return "es"
        if normalized in {"意大利语", "italian", "it"}:
            return "it"
        if normalized in {"葡萄牙语", "portuguese", "pt"}:
            return "pt"
        if normalized in {"荷兰语", "dutch", "nl"}:
            return "nl"
        return "en"

    @staticmethod
    def _translation_text(entry: WordEntry) -> str:
        text = re.sub(r"[_]+", " ", entry.word.strip())
        return re.sub(r"\s+", " ", text)

    @staticmethod
    @lru_cache(maxsize=4096)
    def _translate_cached(source_code: str, target_code: str, text: str) -> str:
        if not text:
            return ""
        translation = _installed_translation(source_code, target_code)
        if translation is None:
            _install_translation_package(source_code, target_code)
            translation = _installed_translation(source_code, target_code)
        if translation is None:
            raise RuntimeError(f"Argos Translate 缺少 {source_code}->{target_code} 翻译包。")
        return str(translation.translate(text))


def _installed_translation(source_code: str, target_code: str):
    from argostranslate import translate

    for source_language in translate.get_installed_languages():
        if source_language.code != source_code:
            continue
        for translation in source_language.translations_from:
            if translation.to_lang.code == target_code:
                return translation
    return None


def _install_translation_package(source_code: str, target_code: str) -> None:
    from argostranslate import package

    package.update_package_index()
    available_packages = package.get_available_packages()
    selected = next(
        (
            item
            for item in available_packages
            if item.from_code == source_code and item.to_code == target_code
        ),
        None,
    )
    if selected is None:
        raise RuntimeError(f"Argos Translate 没有可用的 {source_code}->{target_code} 翻译包。")
    package.install_from_path(selected.download())
