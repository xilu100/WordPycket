from __future__ import annotations

import sys
from types import SimpleNamespace

from wordpycket.infrastructure.csv_importer import WordFrequencyCsvImporter
from wordpycket.infrastructure.pdf_vocabulary_importer import PdfVocabularyImporter


def test_pdf_vocabulary_importer_detects_english_and_counts_phrases(tmp_path) -> None:
    text = (
        "Machine learning improves language models. "
        "Machine learning improves search systems. "
        "Language models use training data."
    )

    entries, language, schema = PdfVocabularyImporter.entries_from_text(text)

    assert language == "英语"
    assert schema.columns == ("Index", "English", "Chinese", "Frequency", "Forms")
    terms = {entry.word: entry.frequency for entry in entries}
    assert terms["machine learning"] == 2
    assert "machine" not in terms
    assert "learning" not in terms


def test_pdf_vocabulary_importer_detects_german(tmp_path, monkeypatch) -> None:
    text = (
        "Der Vektor und die Matrix sind wichtig. "
        "Der Vektor ist Teil der linearen Algebra."
    )

    monkeypatch.setattr(
        PdfVocabularyImporter,
        "_spacy_lemma_counts",
        classmethod(lambda _cls, _language, counts: [(word, count, "") for word, count in counts.items()]),
    )

    entries, language, schema = PdfVocabularyImporter.entries_from_text(text)

    assert language == "德语"
    assert schema.columns == ("Index", "Deutsch", "Chinesisch", "Häufigkeit", "Formen")
    assert any(entry.word == "vektor" for entry in entries)


def test_pdf_vocabulary_importer_uses_dominant_language_not_first_language() -> None:
    text = (
        "これは短い日本語です。\n"
        "The model improves language translation. "
        "The language model improves retrieval. "
        "The model learns from data."
    )

    _entries, language, schema = PdfVocabularyImporter.entries_from_text(text)

    assert language == "英语"
    assert schema.columns == ("Index", "English", "Chinese", "Frequency", "Forms")


def test_pdf_vocabulary_importer_writes_csv_compatible_with_csv_importer(tmp_path) -> None:
    csv_path = tmp_path / "word_frequency.csv"
    entries, _language, schema = PdfVocabularyImporter.entries_from_text(
        "Machine learning helps machine translation."
    )

    PdfVocabularyImporter.write_csv(entries[:5], schema, csv_path)
    result = WordFrequencyCsvImporter(csv_path).load_with_metadata()

    assert result.language == "英语"
    assert result.entries
    assert result.entries[0].source_index == 1
    assert result.entries[0].frequency > 0


def test_pdf_vocabulary_importer_cleans_with_llm_and_reindexes(tmp_path, monkeypatch) -> None:
    class FakeCleaner:
        def clean_pdf_vocabulary_entries(self, entries, language, progress_callback=None):
            assert language == "英语"
            if progress_callback is not None:
                progress_callback("fake clean", 80)
            return [entry for entry in entries if entry.word != "john"]

    pdf_path = tmp_path / "source.pdf"
    csv_path = tmp_path / "word_frequency.csv"
    monkeypatch.setattr(
        PdfVocabularyImporter,
        "extract_text",
        staticmethod(lambda _path: "John wrote code. Machine learning improves systems."),
    )
    monkeypatch.setattr(
        PdfVocabularyImporter,
        "_nltk_lemma_counts",
        classmethod(lambda _cls, counts: [(word, count, "") for word, count in counts.items()]),
    )

    result = PdfVocabularyImporter(pdf_path, csv_path, vocabulary_cleaner=FakeCleaner()).build()

    assert "john" not in {entry.word for entry in result.entries}
    assert [entry.source_index for entry in result.entries] == list(range(1, len(result.entries) + 1))


def test_pdf_vocabulary_importer_ignores_formulas_urls_and_noise() -> None:
    text = """
    E = mc^2 + \\alpha / \\beta
    https://example.com/paper?id=42
    [12, 14-16]
    x_i = \\sum_j W_{ij} h_j
    Machine learning models improve translation.
    Machine learning models improve retrieval.
    """

    entries, _language, _schema = PdfVocabularyImporter.entries_from_text(text)

    terms = {entry.word for entry in entries}
    assert "machine learning" in terms
    assert "machine learning models" not in terms
    assert "https" not in terms
    assert "example" not in terms
    assert "mc" not in terms
    assert "sum" not in terms
    assert "ij" not in terms


def test_pdf_vocabulary_importer_ignores_obvious_code_lines() -> None:
    text = """
    import numpy as np
    def train_model(x):
        return model.predict(x)
    const value = items.map(item => item.score);
    for (int i = 0; i < n; i++) { result += values[i]; }
    Machine learning improves translation quality.
    Machine learning improves retrieval quality.
    """

    entries, _language, _schema = PdfVocabularyImporter.entries_from_text(text)

    terms = {entry.word for entry in entries}
    assert "machine learning" in terms
    assert "import" not in terms
    assert "numpy" not in terms
    assert "predict" not in terms
    assert "const" not in terms
    assert "result" not in terms


def test_pdf_vocabulary_importer_groups_english_forms_under_lemma() -> None:
    text = (
        "Use tools. The system uses tools. "
        "People are using tools. It was used before. "
        "They are being tested and were useful."
    )

    entries, language, _schema = PdfVocabularyImporter.entries_from_text(text)

    assert language == "英语"
    by_word = {entry.word: entry for entry in entries}
    assert by_word["use"].frequency == 4
    assert by_word["use"].forms == "used(1), uses(1), using(1)"
    assert by_word["be"].frequency == 5
    assert by_word["be"].forms == "are(2), being(1), was(1), were(1)"


def test_pdf_vocabulary_importer_keeps_fixed_phrase_and_skips_generic_extension() -> None:
    text = (
        "The random forest model improves predictions. "
        "A random forest model handles features. "
        "Random forest methods are robust."
    )

    entries, language, _schema = PdfVocabularyImporter.entries_from_text(text)

    assert language == "英语"
    terms = {entry.word: entry.frequency for entry in entries}
    assert terms["random forest"] == 3
    assert "random" not in terms
    assert "forest" not in terms
    assert "random forest model" not in terms


def test_pdf_vocabulary_importer_keeps_eponym_technical_bigram() -> None:
    text = (
        "Laplace law describes membrane tension. "
        "Laplace law appears in physiology. "
        "The Fourier transform analyzes signals."
    )

    entries, language, _schema = PdfVocabularyImporter.entries_from_text(text)

    assert language == "英语"
    terms = {entry.word: entry.frequency for entry in entries}
    assert terms["laplace law"] == 2
    assert terms["fourier transform"] == 1
    assert "laplace" not in terms
    assert "law" not in terms


def test_pdf_vocabulary_importer_keeps_common_technical_bigrams_without_ai() -> None:
    text = (
        "Surface tension changes pressure. "
        "Surface tension affects droplets. "
        "Linear equation systems are solved. "
        "A normal distribution models noise. "
        "The paper result appears in a table."
    )

    entries, language, _schema = PdfVocabularyImporter.entries_from_text(text)

    assert language == "英语"
    terms = {entry.word: entry.frequency for entry in entries}
    assert terms["surface tension"] == 2
    assert terms["linear equation"] == 1
    assert terms["normal distribution"] == 1
    assert "paper result" not in terms


def test_pdf_vocabulary_importer_groups_german_forms_under_lemma(monkeypatch) -> None:
    text = (
        "Der Vektor ist wichtig. Die Vektoren sind wichtig. "
        "Der Vektor war zentral."
    )

    monkeypatch.setattr(
        PdfVocabularyImporter,
        "_spacy_lemma_counts",
        classmethod(
            lambda _cls, _language, _counts: [
                ("vektor", 3, "vektoren(1)"),
                ("sein", 3, "ist(1), sind(1), war(1)"),
                ("wichtig", 2, ""),
                ("zentral", 1, ""),
            ]
        ),
    )

    entries, language, _schema = PdfVocabularyImporter.entries_from_text(text)

    assert language == "德语"
    by_word = {entry.word: entry for entry in entries}
    assert by_word["vektor"].frequency == 3
    assert by_word["vektor"].forms == "vektoren(1)"
    assert by_word["sein"].frequency == 3
    assert by_word["sein"].forms == "ist(1), sind(1), war(1)"


def test_pdf_vocabulary_importer_groups_spanish_forms_under_lemma(monkeypatch) -> None:
    text = (
        "El modelo mejora resultados. Los modelos mejoran sistemas. "
        "Este modelo mejora traducciones."
    )

    monkeypatch.setattr(
        PdfVocabularyImporter,
        "_spacy_lemma_counts",
        classmethod(
            lambda _cls, _language, _counts: [
                ("modelo", 3, "modelos(1)"),
                ("mejora", 2, ""),
                ("resultado", 1, "resultados(1)"),
                ("sistema", 1, "sistemas(1)"),
                ("traducción", 1, "traducciones(1)"),
            ]
        ),
    )

    entries, language, _schema = PdfVocabularyImporter.entries_from_text(text)

    assert language == "西班牙语"
    by_word = {entry.word: entry for entry in entries}
    assert by_word["modelo"].frequency == 3
    assert by_word["modelo"].forms == "modelos(1)"


def test_pdf_vocabulary_importer_default_max_entries_is_10000(tmp_path) -> None:
    importer = PdfVocabularyImporter(tmp_path / "source.pdf", tmp_path / "words.csv")

    assert importer._max_entries == 10000


def test_pdf_vocabulary_importer_prefers_nltk_when_available(monkeypatch) -> None:
    class FakeWordNetLemmatizer:
        def lemmatize(self, word: str, pos: str = "n") -> str:
            if word in {"uses", "using", "used"}:
                return "use"
            return word

    fake_nltk = SimpleNamespace()
    fake_stem = SimpleNamespace(WordNetLemmatizer=FakeWordNetLemmatizer)
    fake_corpus = SimpleNamespace(wordnet=SimpleNamespace(ensure_loaded=lambda: None))
    monkeypatch.setitem(sys.modules, "nltk", fake_nltk)
    monkeypatch.setitem(sys.modules, "nltk.stem", fake_stem)
    monkeypatch.setitem(sys.modules, "nltk.corpus", fake_corpus)

    entries, language, _schema = PdfVocabularyImporter.entries_from_text("Use uses using used.")

    assert language == "英语"
    by_word = {entry.word: entry for entry in entries}
    assert by_word["use"].frequency == 4
    assert by_word["use"].forms == "used(1), uses(1), using(1)"


def test_pdf_vocabulary_importer_prefers_spacy_when_model_is_available(monkeypatch) -> None:
    class FakeToken:
        def __init__(self, lemma: str) -> None:
            self.lemma_ = lemma

    class FakeNlp:
        def __call__(self, word: str):
            lemmas = {"vektoren": "vektor", "vektor": "vektor"}
            return [FakeToken(lemmas.get(word, word))]

    fake_spacy = SimpleNamespace(load=lambda *_args, **_kwargs: FakeNlp())
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    monkeypatch.setattr(PdfVocabularyImporter, "_SPACY_PIPELINES", {})
    monkeypatch.setattr(PdfVocabularyImporter, "_nltk_lemma_counts", classmethod(lambda *_args: None))

    entries, language, _schema = PdfVocabularyImporter.entries_from_text(
        "Der Vektor ist wichtig. Die Vektoren sind wichtig."
    )

    assert language == "德语"
    by_word = {entry.word: entry for entry in entries}
    assert by_word["vektor"].frequency == 2
    assert by_word["vektor"].forms == "vektoren(1)"
