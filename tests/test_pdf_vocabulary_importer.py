from __future__ import annotations

import json
from pathlib import Path

from wordpycket.infrastructure.csv_importer import WordFrequencyCsvImporter
from wordpycket.infrastructure.pdf_vocabulary_importer import (
    DeterministicMultilingualFrequencyEngine,
    PdfVocabularyImporter,
    run_pdf_import_isolated,
)


def test_pdf_vocabulary_importer_outputs_strict_csv_schema(tmp_path) -> None:
    csv_path = tmp_path / "word_frequency.csv"
    entries, language, schema = PdfVocabularyImporter.entries_from_text(
        "Machine learning improves language models. "
        "Machine learning improves search systems."
    )

    PdfVocabularyImporter.write_csv(entries, schema, csv_path)

    assert language == "英语"
    assert schema.columns == ("Index", "English", "Chinese", "Frequency", "Forms")
    assert csv_path.read_text(encoding="utf-8-sig").splitlines()[0] == "Index,English,Chinese,Frequency,Forms"


def test_pdf_vocabulary_importer_counts_lemmas_forms_and_phrases() -> None:
    entries, language, _schema = PdfVocabularyImporter.entries_from_text(
        "Use tools. The system uses tools. "
        "People are using tools. It was used before. "
        "Machine learning improves systems. Machine learning improves search."
    )

    assert language == "英语"
    by_word = {entry.word: entry for entry in entries}
    assert by_word["use"].frequency == 4
    assert by_word["use"].forms == "use;used;uses;using"
    assert by_word["machine learning"].frequency == 2


def test_pdf_vocabulary_importer_detects_german_with_fixed_schema() -> None:
    entries, language, schema = PdfVocabularyImporter.entries_from_text(
        "Der Vektor ist wichtig. Die Vektoren sind wichtig."
    )

    assert language == "德语"
    assert schema.columns == ("Index", "English", "Chinese", "Frequency", "Forms")
    by_word = {entry.word: entry for entry in entries}
    assert by_word["vektor"].frequency == 2
    assert by_word["vektor"].forms == "vektor;vektoren"


def test_pdf_vocabulary_importer_segments_chinese_without_model() -> None:
    entries, language, schema = PdfVocabularyImporter.entries_from_text(
        "机器学习提高系统性能。机器学习改善搜索质量。"
    )

    assert language == "中文"
    assert schema.columns == ("Index", "English", "Chinese", "Frequency", "Forms")
    terms = {entry.word: entry.frequency for entry in entries}
    assert terms["机器"] == 2
    assert terms["学习"] == 2


def test_pdf_vocabulary_importer_writes_csv_compatible_with_csv_importer(tmp_path) -> None:
    csv_path = tmp_path / "word_frequency.csv"
    entries, _language, schema = PdfVocabularyImporter.entries_from_text(
        "Machine learning helps machine translation."
    )

    PdfVocabularyImporter.write_csv(entries, schema, csv_path)
    result = WordFrequencyCsvImporter(csv_path).load_with_metadata()

    assert result.language == "英语"
    assert result.entries
    assert result.entries[0].source_index == 1
    assert result.entries[0].frequency > 0


def test_pdf_vocabulary_importer_ignores_formulas_urls_and_code() -> None:
    text = """
    E = mc^2 + \\alpha / \\beta
    https://example.com/paper?id=42
    import numpy as np
    def train_model(x):
        return model.predict(x)
    Machine learning improves translation quality.
    Machine learning improves retrieval quality.
    """

    entries, _language, _schema = PdfVocabularyImporter.entries_from_text(text)

    terms = {entry.word for entry in entries}
    assert "machine learning" in terms
    assert "https" not in terms
    assert "numpy" not in terms
    assert "predict" not in terms


def test_pdf_vocabulary_importer_filters_identifiers_names_and_keeps_eponym_terms() -> None:
    entries, _language, _schema = PdfVocabularyImporter.entries_from_text(
        "Li Hua generated a file named wordApp which mentions Bayes' theorem. "
        "tempScore hasError JSONParserFactory VECTOR_SIZE should not become vocabulary. "
        "Bayes' theorem improves inference."
    )

    terms = {entry.word: entry for entry in entries}
    assert "bayes' theorem" in terms
    assert terms["bayes' theorem"].frequency == 2
    assert "li" not in terms
    assert "hua" not in terms
    assert "wordapp" not in terms
    assert "tempscore" not in terms
    assert "haserror" not in terms
    assert "jsonparserfactory" not in terms
    assert "vector_size" not in terms


def test_pdf_vocabulary_importer_does_not_emit_repeated_bigram_noise() -> None:
    entries, _language, _schema = PdfVocabularyImporter.entries_from_text(
        "parser parser parser parser extraction extraction extraction extraction"
    )

    terms = {entry.word for entry in entries}
    assert "parser" in terms
    assert "extraction" in terms
    assert "parser parser" not in terms
    assert "extraction extraction" not in terms


def test_pdf_vocabulary_importer_optional_llm_cleaner_filters_and_reindexes(tmp_path, monkeypatch) -> None:
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

    result = PdfVocabularyImporter(pdf_path, csv_path, vocabulary_cleaner=FakeCleaner()).build()

    assert "john" not in {entry.word for entry in result.entries}
    assert [entry.source_index for entry in result.entries] == list(range(1, len(result.entries) + 1))
    assert "john" not in csv_path.read_text(encoding="utf-8-sig").lower()


def test_run_pdf_import_isolated_uses_in_process_path_when_cleaner_is_present(tmp_path, monkeypatch) -> None:
    class FakeCleaner:
        def clean_pdf_vocabulary_entries(self, entries, language, progress_callback=None):
            return entries[:1]

    monkeypatch.setattr(
        PdfVocabularyImporter,
        "extract_text",
        staticmethod(lambda _path: "Machine learning improves systems."),
    )

    result = run_pdf_import_isolated(tmp_path / "source.pdf", tmp_path / "words.csv", FakeCleaner())

    assert result.language == "英语"
    rows = (tmp_path / "words.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(rows) == 2


def test_run_pdf_import_isolated_uses_child_when_cleaner_supports_pdf_payload(tmp_path, monkeypatch) -> None:
    progress: list[tuple[str, int]] = []
    created_processes = []

    class FakeCleaner:
        def isolated_pdf_import_payload(self, pdf_path, csv_path):
            return json.dumps(
                {
                    "pdf_path": str(pdf_path),
                    "csv_path": str(csv_path),
                    "use_llm_cleanup": True,
                    "model_dir": str(tmp_path / "model"),
                }
            )

        def isolated_pdf_import_environment(self):
            return {"WORDPYCKET_PDF_CHILD": "1"}

    class FakeStdin:
        def __init__(self) -> None:
            self.value = ""

        def write(self, value: str) -> None:
            self.value += value

        def close(self) -> None:
            return None

    class FakeProcess:
        def __init__(self, _command, **kwargs) -> None:
            self.stdin = FakeStdin()
            self.stdout = iter(
                [
                    json.dumps({"type": "progress", "message": "AI 检查词表：1/2", "percent": 40}) + "\n",
                    json.dumps({"type": "result", "language": "英语", "csv_path": str(tmp_path / "words.csv")}) + "\n",
                ]
            )
            self.kwargs = kwargs
            created_processes.append(self)

        def wait(self) -> int:
            return 0

    monkeypatch.setattr("wordpycket.infrastructure.pdf_vocabulary_importer.subprocess.Popen", FakeProcess)

    result = run_pdf_import_isolated(
        tmp_path / "source.pdf",
        tmp_path / "words.csv",
        FakeCleaner(),
        progress_callback=lambda message, percent: progress.append((message, percent)),
    )

    assert result.language == "英语"
    assert progress == [("AI 检查词表：1/2", 40)]
    process = created_processes[0]
    assert process.kwargs["env"]["WORDPYCKET_PDF_CHILD"] == "1"
    assert json.loads(process.stdin.value)["use_llm_cleanup"] is True


def test_engine_can_analyze_text_folder(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("Use tools. The system uses tools.", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_text(
        "Der Vektor ist wichtig. Die Vektoren sind wichtig.",
        encoding="utf-8",
    )

    entries, _language, schema = DeterministicMultilingualFrequencyEngine().analyze_folder(tmp_path)

    assert schema.columns == ("Index", "English", "Chinese", "Frequency", "Forms")
    assert {entry.word for entry in entries} >= {"use", "tool", "vektor"}
