from wordpycket.domain.entities import WordEntry
from wordpycket.infrastructure import meaning_translator
from wordpycket.infrastructure.meaning_translator import ArgosMeaningTranslator


def test_german_meaning_translation_uses_german_to_chinese_directly(monkeypatch) -> None:
    calls = []

    def fake_translate_direct(source_code: str, target_code: str, text: str) -> str:
        calls.append((source_code, target_code, text))
        return "模型"

    ArgosMeaningTranslator._translate_cached.cache_clear()
    monkeypatch.setattr(meaning_translator, "_translate_direct", fake_translate_direct)

    translated = ArgosMeaningTranslator().translate_meaning(
        WordEntry(word="das Modell", meaning=""),
        language="德语",
    )

    assert translated == "模型"
    assert calls == [("de", "zh", "das Modell")]


def test_english_meaning_translation_uses_english_to_chinese_directly(monkeypatch) -> None:
    calls = []

    def fake_translate_direct(source_code: str, target_code: str, text: str) -> str:
        calls.append((source_code, target_code, text))
        return "模型"

    ArgosMeaningTranslator._translate_cached.cache_clear()
    monkeypatch.setattr(meaning_translator, "_translate_direct", fake_translate_direct)

    translated = ArgosMeaningTranslator().translate_meaning(
        WordEntry(word="model", meaning=""),
        language="英语",
    )

    assert translated == "模型"
    assert calls == [("en", "zh", "model")]
