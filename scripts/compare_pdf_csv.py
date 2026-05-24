from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED = PROJECT_ROOT / "pdf_parser_llm_cleanup_expected.csv"
DEFAULT_ACTUAL = PROJECT_ROOT / "input" / "pdf_parser_llm_cleanup_test.csv"


@dataclass(frozen=True)
class CsvEntry:
    word: str
    frequency: int
    forms: tuple[str, ...]


def load_entries(path: Path) -> dict[str, CsvEntry]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        word_column = detect_word_column(reader.fieldnames)
        frequency_column = detect_column(reader.fieldnames, ("Frequency", "Häufigkeit", "Fréquence", "Frecuencia"))
        forms_column = detect_column(reader.fieldnames, ("Forms", "Formen", "Formes"))

        entries: dict[str, CsvEntry] = {}
        for row_number, row in enumerate(reader, start=2):
            raw_word = row.get(word_column, "")
            word = normalize_word(raw_word)
            if not word:
                continue
            frequency = parse_int(row.get(frequency_column, ""), path, row_number, frequency_column)
            forms = normalize_forms(row.get(forms_column, ""))
            entries[word] = CsvEntry(word=word, frequency=frequency, forms=forms)
        return entries


def detect_word_column(fieldnames: list[str]) -> str:
    return detect_column(
        fieldnames,
        (
            "English",
            "Deutsch",
            "Français",
            "Español",
            "Italiano",
            "Português",
            "Nederlands",
            "日本語",
            "한국어",
        ),
    )


def detect_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    normalized = {name.strip(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(f"CSV header is missing one of {candidates}. Actual header: {fieldnames}")


def normalize_word(value: str) -> str:
    return " ".join(value.strip().lower().split())


def normalize_forms(value: str) -> tuple[str, ...]:
    forms = {
        " ".join(form.strip().lower().split())
        for form in value.split(";")
        if form.strip()
    }
    return tuple(sorted(forms))


def parse_int(value: str | None, path: Path, row_number: int, column: str) -> int:
    try:
        return int(value or "0")
    except ValueError as error:
        raise ValueError(f"Invalid integer in {path}:{row_number} column {column}: {value!r}") from error


def compare(expected_path: Path, actual_path: Path) -> int:
    expected = load_entries(expected_path)
    actual = load_entries(actual_path)

    expected_words = set(expected)
    actual_words = set(actual)
    missing = sorted(expected_words - actual_words)
    extra = sorted(actual_words - expected_words)
    common = sorted(expected_words & actual_words)
    frequency_mismatches = [
        word
        for word in common
        if expected[word].frequency != actual[word].frequency
    ]
    forms_mismatches = [
        word
        for word in common
        if expected[word].forms != actual[word].forms
    ]

    print("CSV comparison")
    print(f"expected: {expected_path}")
    print(f"actual:   {actual_path}")
    print()
    print("Summary")
    print(f"expected entries: {len(expected)}")
    print(f"actual entries:   {len(actual)}")
    print(f"matched words:    {len(common)}")
    print(f"missing words:    {len(missing)}")
    print(f"extra words:      {len(extra)}")
    print(f"frequency diffs:  {len(frequency_mismatches)}")
    print(f"forms diffs:      {len(forms_mismatches)}")

    print_words("Missing expected words", missing)
    print_words("Extra actual words", extra)
    print_frequency_diffs(frequency_mismatches, expected, actual)
    print_forms_diffs(forms_mismatches, expected, actual)

    return 1 if missing or extra or frequency_mismatches or forms_mismatches else 0


def print_words(title: str, words: list[str]) -> None:
    if not words:
        return
    print()
    print(title)
    for word in words:
        print(f"- {word}")


def print_frequency_diffs(words: list[str], expected: dict[str, CsvEntry], actual: dict[str, CsvEntry]) -> None:
    if not words:
        return
    print()
    print("Frequency differences")
    for word in words:
        print(f"- {word}: expected={expected[word].frequency} actual={actual[word].frequency}")


def print_forms_diffs(words: list[str], expected: dict[str, CsvEntry], actual: dict[str, CsvEntry]) -> None:
    if not words:
        return
    print()
    print("Forms differences")
    for word in words:
        expected_forms = ";".join(expected[word].forms)
        actual_forms = ";".join(actual[word].forms)
        print(f"- {word}: expected={expected_forms!r} actual={actual_forms!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare expected and actual WordPycket PDF-import CSV files by word.")
    parser.add_argument(
        "actual",
        type=Path,
        nargs="?",
        default=DEFAULT_ACTUAL,
        help="CSV generated by the application, for example input/pdf_parser_llm_cleanup_test.csv",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_EXPECTED,
        help="Expected baseline CSV. Defaults to pdf_parser_llm_cleanup_expected.csv in the current directory.",
    )
    args = parser.parse_args()
    raise SystemExit(compare(args.expected, args.actual))


if __name__ == "__main__":
    main()
