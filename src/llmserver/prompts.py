from __future__ import annotations

import json
import re

from llmserver.contracts import GeneratedCorrection, GeneratedExample, GeneratedExplanation, WordEntry


def build_prompt(entry: WordEntry, scope: str = "") -> str:
    forms = f"\nWord forms: {entry.forms}" if entry.forms else ""
    scope_text = build_scope_text(scope)
    meaning_text = entry.meaning or "(empty)"
    needs_meaning = not _has_cjk(entry.meaning)
    meaning_requirement = (
        "- The Chinese meaning is empty; provide a concise Chinese vocabulary meaning in the meaning field.\n"
        if needs_meaning
        else "- Do not change, validate, or repeat the Chinese meaning field.\n"
    )
    json_shape = (
        '{"example_sentence": "...", "example_sentence_cn": "...", "meaning": "..."}'
        if needs_meaning
        else '{"example_sentence": "...", "example_sentence_cn": "..."}'
    )
    return (
        "Generate one natural, short English sentence for this vocabulary word, "
        "and provide a fluent Chinese translation.\n"
        f"{scope_text}"
        f"Word: {entry.word}\n"
        f"Chinese meaning: {meaning_text}"
        f"{forms}\n"
        "Requirements:\n"
        "- The English sentence must include the word or a common form of it.\n"
        "- Interpret the word according to the scope above when choosing meanings and translations.\n"
        f"{meaning_requirement}"
        "- Keep the sentence under 16 English words.\n"
        "- Do not explain anything.\n"
        f"Return JSON exactly like: {json_shape}"
    )


def build_batch_prompt(entries: list[WordEntry], scope: str = "") -> str:
    scope_text = build_scope_text(scope)
    rows = []
    for index, entry in enumerate(entries, start=1):
        rows.append(
            {
                "batch_index": index,
                "word": entry.word,
                "meaning": entry.meaning or "",
                "forms": entry.forms or "",
                "needs_meaning": not _has_cjk(entry.meaning),
            }
        )
    return (
        "Generate one natural, short English sentence for each vocabulary item, "
        "and provide a fluent Chinese translation for each sentence.\n"
        f"{scope_text}"
        "Vocabulary items JSON:\n"
        f"{json.dumps(rows, ensure_ascii=False)}\n"
        "Requirements:\n"
        "- Return exactly one result for every input item.\n"
        "- Preserve each batch_index exactly.\n"
        "- The English sentence must include the word or a common form of it.\n"
        "- Interpret each word according to the scope above when choosing meanings and translations.\n"
        "- If needs_meaning is true, provide only a concise dictionary-style Chinese meaning in the meaning field.\n"
        "- The meaning field must be a short Chinese word or phrase, not a definition, explanation, example, or sentence.\n"
        "- The Chinese translation must be fluent Simplified Chinese and must not leave English words untranslated.\n"
        "- If needs_meaning is false, omit meaning or return an empty meaning string.\n"
        "- Keep each sentence under 16 English words.\n"
        "- Do not explain anything.\n"
        'Return JSON exactly like: {"items": [{"batch_index": 1, "example_sentence": "...", '
        '"example_sentence_cn": "...", "meaning": "..."}]}'
    )


def build_correction_prompt(entry: WordEntry, scope: str = "") -> str:
    normalized_word = normalize_english_term(entry.word)
    scope_text = build_scope_text(scope)
    return (
        "Correct this vocabulary record.\n"
        f"{scope_text}"
        f"Original English: {entry.word}\n"
        f"Normalized English candidate: {normalized_word}\n"
        "Requirements:\n"
        "- Fix English formatting errors such as underscores used as spaces.\n"
        "- If the original English contains underscores, replace them with spaces, not hyphens.\n"
        "- Keep proper compounds/hyphenation only when standard English requires them.\n"
        "- Do not correct, translate, validate, or comment on the Chinese meaning.\n"
        "- Do not correct, validate, or comment on word forms.\n"
        "- Do not explain anything outside JSON.\n"
        'Return JSON exactly like: {"corrected_word": "...", "note": "..."}'
    )


def build_explanation_prompt(entry: WordEntry, scope: str = "", language: str = "") -> str:
    scope_text = build_scope_text(scope)
    language_text = language.strip() or "Infer from the word or phrase and the vocabulary table."
    domain_text = scope.strip() or "(empty)"
    return (
        "Explain this vocabulary item for a Chinese learner.\n"
        f"Target vocabulary language: {language_text}\n"
        f"{scope_text}"
        f"Word or phrase: {entry.word}\n"
        f"Chinese meaning in the app, for reference only: {entry.meaning or '(empty)'}\n"
        f"Word forms: {entry.forms or '(empty)'}\n"
        f"Example sentence: {entry.example_sentence or '(empty)'}\n"
        f"Example Chinese translation: {entry.example_sentence_cn or '(empty)'}\n"
        f"Required domain for domain-specific usage: {domain_text}\n"
        "Requirements:\n"
        "- Write in Simplified Chinese.\n"
        "- Keep it concise but useful.\n"
        "- Structure the explanation in exactly three labeled sections: 意思, 常规用法, 领域用法.\n"
        "- In 意思, give the meaning of the target word or phrase itself, using the app's Chinese meaning only as a reference when helpful.\n"
        "- In 常规用法, write the explanation in Chinese, but explicitly explain the English usage of the target word or phrase: grammar role, common English collocations, register, nuance, and at least one natural English phrase when useful.\n"
        "- In 领域用法, write the explanation in Chinese, but explicitly explain how the English target word or phrase is used in the required domain/scope, including common English technical collocations or term patterns when useful; if no domain is provided, say that no specific domain was set and give the most likely subject-area usage.\n"
        "- If an example sentence is provided, explain the target-language usage in that sentence.\n"
        "- Do not explain how the Chinese translation is used in Chinese.\n"
        "- Do not make 常规用法 or 领域用法 only a Chinese definition; both sections must mention target-language/English usage.\n"
        "- Do not treat the Chinese meaning as the target word; it is only a reference to disambiguate the vocabulary item.\n"
        "- Do not invent a different headword.\n"
        "- Keep the three section labels visible in the explanation text.\n"
        'Return JSON exactly like: {"explanation": "..."}'
    )


def build_scope_text(scope: str) -> str:
    cleaned = scope.strip()
    if not cleaned:
        return ""
    return (
        f"Scope/domain: {cleaned}\n"
        "Use this scope to resolve ambiguous English terms. "
        "For example, decide whether an English term should stay hyphenated, "
        "spaced, or joined based on this scope.\n"
    )


def normalize_english_term(text: str) -> str:
    normalized = re.sub(r"[_]+", " ", text.strip())
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def parse_response(content: str, require_meaning: bool = False) -> GeneratedExample:
    data = _json_object(content)
    example_sentence = str(data.get("example_sentence", "")).strip()
    example_sentence_cn = str(data.get("example_sentence_cn", "")).strip()
    meaning = str(data.get("meaning", "")).strip()
    _validate_generated_example(example_sentence, example_sentence_cn, content)
    if require_meaning and not _has_cjk(meaning):
        raise RuntimeError(f"模型返回缺少中文释义字段：{content}")
    return GeneratedExample(example_sentence, example_sentence_cn, meaning)


def parse_batch_response(content: str, entries: list[WordEntry]) -> list[tuple[WordEntry, GeneratedExample]]:
    data = _json_batch_items(content)
    by_index: dict[int, dict] = {}
    for item in data:
        if not isinstance(item, dict):
            raise RuntimeError(f"模型批量返回包含非对象条目：{content}")
        try:
            batch_index = int(item.get("batch_index", 0))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"模型批量返回缺少有效 batch_index：{content}") from error
        by_index[batch_index] = item

    results: list[tuple[WordEntry, GeneratedExample]] = []
    for index, entry in enumerate(entries, start=1):
        item = by_index.get(index)
        if item is None:
            raise RuntimeError(f"模型批量返回缺少第 {index} 条：{content}")
        example_sentence = str(item.get("example_sentence", "")).strip()
        example_sentence_cn = str(item.get("example_sentence_cn", "")).strip()
        meaning = str(item.get("meaning", "")).strip()
        _validate_generated_example(example_sentence, example_sentence_cn, content, index)
        if not _has_cjk(entry.meaning) and not _has_cjk(meaning):
            raise RuntimeError(f"模型批量返回第 {index} 条缺少中文释义字段：{content}")
        results.append((entry, GeneratedExample(example_sentence, example_sentence_cn, meaning)))
    return results


def _json_batch_items(content: str) -> list:
    try:
        data = _json_object(content)
    except RuntimeError:
        return _json_array(content)
    items = data.get("items")
    if isinstance(items, list):
        return items
    if "batch_index" in data:
        return [data]
    raise RuntimeError(f"模型批量返回缺少 items 数组：{content}")


def _validate_generated_example(
    example_sentence: str,
    example_sentence_cn: str,
    content: str,
    batch_index: int | None = None,
) -> None:
    label = f"第 {batch_index} 条" if batch_index is not None else ""
    if not example_sentence or not example_sentence_cn:
        raise RuntimeError(f"模型返回{label}缺少例句字段：{content}")
    if _has_cjk(example_sentence) or not re.search(r"[A-Za-z]", example_sentence):
        raise RuntimeError(f"模型返回{label}英文例句不是英文：{content}")
    if not _has_cjk(example_sentence_cn):
        raise RuntimeError(f"模型返回{label}中文例句缺少中文：{content}")


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def parse_correction_response(content: str, original: WordEntry) -> GeneratedCorrection:
    data = _json_object(content)
    corrected_word = str(data.get("corrected_word", "")).strip()
    note = str(data.get("note", "")).strip()
    normalized_word = normalize_english_term(original.word)
    if "_" in original.word:
        corrected_word = normalized_word
    if not corrected_word:
        corrected_word = normalized_word
    return GeneratedCorrection(corrected_word, note)


def parse_explanation_response(content: str) -> GeneratedExplanation:
    data = _json_object(content)
    explanation = _format_explanation(data.get("explanation", ""))
    if not explanation:
        raise RuntimeError(f"模型返回缺少解释字段：{content}")
    return GeneratedExplanation(explanation)


def _format_explanation(value) -> str:
    if isinstance(value, dict):
        preferred_labels = ["意思", "常规用法", "领域用法"]
        lines = []
        used_labels = set()
        for label in preferred_labels:
            text = str(value.get(label, "")).strip()
            if text:
                lines.append(f"{label}：{text}")
                used_labels.add(label)
        for label, text_value in value.items():
            if label in used_labels:
                continue
            text = str(text_value).strip()
            if text:
                lines.append(f"{label}：{text}")
        return "\n".join(lines).strip()
    return str(value).strip()


def _json_object(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"模型返回内容不是有效 JSON：{content}") from error
    if not isinstance(data, dict):
        raise RuntimeError(f"模型返回内容不是 JSON 对象：{content}")
    return data


def _json_array(content: str) -> list:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"模型返回内容不是有效 JSON：{content}") from error
    if not isinstance(data, list):
        raise RuntimeError(f"模型返回内容不是 JSON 数组：{content}")
    return data
