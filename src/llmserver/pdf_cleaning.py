from __future__ import annotations

import json
import os
import re
from typing import Any

from llmserver.contracts import WordEntry


def pdf_clean_batches(entries: list[WordEntry], max_rows: int, char_budget: int) -> list[list[WordEntry]]:
    batches: list[list[WordEntry]] = []
    current: list[WordEntry] = []
    current_size = 0
    for entry in entries:
        row_size = len(json.dumps(csv_review_row(entry), ensure_ascii=False, separators=(",", ":")))
        if current and (len(current) >= max_rows or current_size + row_size > char_budget):
            batches.append(current)
            current = []
            current_size = 0
        current.append(entry)
        current_size += row_size
    if current:
        batches.append(current)
    return batches


def pdf_clean_batch_size(auto_size: int) -> int:
    raw_value = os.getenv("WORDPYCKET_PDF_CLEAN_BATCH")
    if raw_value is None:
        return auto_size
    try:
        return max(10, min(100, int(raw_value)))
    except ValueError as error:
        raise RuntimeError("WORDPYCKET_PDF_CLEAN_BATCH 必须是整数。") from error


def pdf_clean_prompt_char_budget(auto_budget: int) -> int:
    raw_value = os.getenv("WORDPYCKET_PDF_CLEAN_CHARS")
    if raw_value is None:
        return auto_budget
    try:
        return max(1000, min(12000, int(raw_value)))
    except ValueError as error:
        raise RuntimeError("WORDPYCKET_PDF_CLEAN_CHARS 必须是整数。") from error


def csv_review_row(entry: WordEntry) -> list[int | str]:
    return [
        entry.source_index,
        truncate_for_prompt(entry.word, 80),
        int(entry.frequency),
        truncate_for_prompt(entry.forms, 80),
    ]


def truncate_for_prompt(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def build_pdf_vocabulary_cleaning_prompt(entries: list[WordEntry], language: str) -> str:
    rows = [csv_review_row(entry) for entry in entries]
    return (
        "You are reviewing a rough vocabulary CSV generated from PDF text.\n"
        f"Detected language: {language}\n"
        "Rows are arrays: [csv_index, term, frequency, forms].\n"
        "For each row, decide whether term is a real learnable word or meaningful terminology phrase.\n"
        "Remove rows that are not learnable vocabulary terms, including:\n"
        "- programming keywords or code/control-flow fragments such as if, else, endif, end if, return, for, while;\n"
        "- variable/function/class identifiers, camelCase/PascalCase/snake_case names, constants, library names, filenames, URLs, emails, or code fragments;\n"
        "- file-like or app-like identifiers such as wordApp, tempScore, hasError, request_id, VECTOR_SIZE, JSONParserFactory;\n"
        "- letter noise, OCR fragments, or non-words such as xy, abc, tmp, idx, foo, bar when they are not established terms;\n"
        "- person names, full person names, author-list names, organization names, venue names, or bibliography artifacts;\n"
        "- page/header/footer artifacts, citation markers, broken fragments, or table labels.\n"
        "Keep rows that are real words, academic terms, domain terms, abbreviations with established meaning, "
        "or meaningful fixed phrases, even if they are rare.\n"
        "Keep ordinary inflected vocabulary rows such as generated, named, mentions, uses, used, using, parsed, or parses "
        "when the normalized term is a real word.\n"
        "Keep technical software/domain vocabulary such as parser, extraction, token, corpus, embedding, vector, matrix, "
        "gradient, inference, retrieval, pipeline, normalization, architecture, repository, aggregate, adapter, service, "
        "domain, translation, alignment, frequency, vocabulary, and machine learning.\n"
        "Keep eponyms and name-derived technical terms when they are used as concepts, methods, or adjectives, "
        "such as Bayes' theorem, Fourier transform, Gaussian distribution, Bayesian, Newtonian, Eulerian, Markov, "
        "Laplace, Hamiltonian, or Turing.\n"
        "Delete a name-derived term only when it is clearly just an author/person entry or bibliography residue. "
        "For example, remove Li, Hua, and Li Hua as person-name noise, but keep Bayes' theorem as a technical term.\n"
        "Review only these CSV rows. Do not infer from the original PDF. "
        "You will receive at most 100 rows in this batch. "
        "Return only the csv_index values from the first column. Do not return batch row numbers. "
        "Do not include reasons or explanations.\n"
        "Return JSON only, exactly like: {\"remove_csv_indexes\": [101, 102]}\n"
        f"CSV rows:\n{json.dumps(rows, ensure_ascii=False, separators=(',', ':'))}"
    )


def parse_pdf_vocabulary_cleaning_response(content: str, batch: list[WordEntry]) -> set[int]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return parse_pdf_cleaning_numbers(content, batch)
    if isinstance(data, list):
        data = {"remove_csv_indexes": data}
    if not isinstance(data, dict):
        return parse_pdf_cleaning_numbers(content, batch)
    values = data.get(
        "remove_csv_indexes",
        data.get("remove_source_indexes", data.get("remove", [])),
    )
    valid_indexes = {entry.source_index for entry in batch}
    indexes: set[int] = set()
    if isinstance(values, list):
        for value in values:
            index = coerce_pdf_clean_index(value, batch, valid_indexes)
            if index is not None:
                indexes.add(index)
    raw_words = data.get("remove_words", [])
    if isinstance(raw_words, list):
        words = {str(value).strip().lower() for value in raw_words if str(value).strip()}
        indexes.update(
            entry.source_index
            for entry in batch
            if entry.word.lower() in words
        )
    return indexes


def parse_pdf_cleaning_numbers(content: str, batch: list[WordEntry]) -> set[int]:
    valid_indexes = {entry.source_index for entry in batch}
    indexes: set[int] = set()
    for value in re.findall(r"\b\d+\b", content):
        index = coerce_pdf_clean_index(value, batch, valid_indexes)
        if index is not None:
            indexes.add(index)
    return indexes


def coerce_pdf_clean_index(
    value: Any,
    batch: list[WordEntry],
    valid_indexes: set[int],
) -> int | None:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    if index in valid_indexes:
        return index
    if 1 <= index <= len(batch):
        return batch[index - 1].source_index
    return None
