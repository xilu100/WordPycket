from __future__ import annotations

import json
import re

from llmserver.contracts import GeneratedCorrection, GeneratedExample, GeneratedExplanation, WordEntry


def _target_language(language: str = "") -> str:
    cleaned = language.strip()
    return "德语" if cleaned == "德语" else "英语"


def needs_chinese_meaning(entry: WordEntry) -> bool:
    return not _has_cjk(entry.meaning)


def build_prompt(entry: WordEntry, scope: str = "", language: str = "") -> str:
    if _target_language(language) == "德语":
        return build_german_prompt(entry, scope)
    return build_english_prompt(entry, scope)


def build_english_prompt(entry: WordEntry, scope: str = "") -> str:
    forms = f"\nWord forms: {entry.forms}" if entry.forms else ""
    scope_text = build_english_scope_text(scope)
    meaning_text = entry.meaning or "(empty)"
    needs_meaning = needs_chinese_meaning(entry)
    meaning_requirement = (
        "- The Chinese meaning is missing or not Chinese; provide a concise Chinese vocabulary meaning in the meaning field.\n"
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
        "- Field example_sentence must be English only and must not contain Chinese characters.\n"
        "- Field example_sentence_cn must be Simplified Chinese.\n"
        "- The English sentence must include the word or a common form of it.\n"
        "- Interpret the word according to the scope above when choosing meanings and translations.\n"
        f"{meaning_requirement}"
        "- Keep the sentence under 16 English words.\n"
        "- Do not explain anything.\n"
        f"Return JSON exactly like: {json_shape}"
    )


def build_german_prompt(entry: WordEntry, scope: str = "") -> str:
    forms = f"\nGerman word forms: {entry.forms}" if entry.forms else ""
    scope_text = build_german_scope_text(scope)
    meaning_text = entry.meaning or "(empty)"
    needs_meaning = needs_chinese_meaning(entry)
    meaning_requirement = (
        "- The Chinese meaning is missing or not Chinese; provide a concise Chinese vocabulary meaning in the meaning field.\n"
        if needs_meaning
        else "- Do not change, validate, or repeat the Chinese meaning field.\n"
    )
    json_shape = (
        '{"example_sentence": "...", "example_sentence_cn": "...", "meaning": "..."}'
        if needs_meaning
        else '{"example_sentence": "...", "example_sentence_cn": "..."}'
    )
    return (
        "Generate one natural, short German sentence for this vocabulary word, "
        "and provide a fluent Chinese translation.\n"
        f"{scope_text}"
        f"German word: {entry.word}\n"
        f"Chinese meaning: {meaning_text}"
        f"{forms}\n"
        "Requirements:\n"
        "- Field example_sentence must be German only and must not contain Chinese characters.\n"
        "- Field example_sentence_cn must be Simplified Chinese.\n"
        "- The German sentence must include the word or a common German form of it.\n"
        "- Interpret the word according to the scope above when choosing meanings and translations.\n"
        f"{meaning_requirement}"
        "- Keep the sentence under 16 German words.\n"
        "- Do not explain anything.\n"
        f"Return JSON exactly like: {json_shape}"
    )


def build_batch_prompt(entries: list[WordEntry], scope: str = "", language: str = "") -> str:
    if _target_language(language) == "德语":
        return build_german_batch_prompt(entries, scope)
    return build_english_batch_prompt(entries, scope)


def _batch_rows(entries: list[WordEntry]) -> list[dict[str, object]]:
    rows = []
    for index, entry in enumerate(entries, start=1):
        rows.append(
            {
                "batch_index": index,
                "word": entry.word,
                "meaning": entry.meaning or "",
                "forms": entry.forms or "",
                "needs_meaning": needs_chinese_meaning(entry),
            }
        )
    return rows


def build_english_batch_prompt(entries: list[WordEntry], scope: str = "") -> str:
    scope_text = build_english_scope_text(scope)
    rows = _batch_rows(entries)
    return (
        "Generate one natural, short English sentence for each vocabulary item, "
        "and provide a fluent Chinese translation for each sentence.\n"
        f"{scope_text}"
        "Vocabulary items JSON:\n"
        f"{json.dumps(rows, ensure_ascii=False)}\n"
        "Requirements:\n"
        "- Return exactly one result for every input item.\n"
        "- Preserve each batch_index exactly.\n"
        "- Every example_sentence must be English only and must not contain Chinese characters.\n"
        "- Every example_sentence_cn must be Simplified Chinese.\n"
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


def build_german_batch_prompt(entries: list[WordEntry], scope: str = "") -> str:
    scope_text = build_german_scope_text(scope)
    rows = _batch_rows(entries)
    return (
        "Generate one natural, short German sentence for each vocabulary item, "
        "and provide a fluent Chinese translation for each sentence.\n"
        f"{scope_text}"
        "German vocabulary items JSON:\n"
        f"{json.dumps(rows, ensure_ascii=False)}\n"
        "Requirements:\n"
        "- Return exactly one result for every input item.\n"
        "- Preserve each batch_index exactly.\n"
        "- Every example_sentence must be German only and must not contain Chinese characters.\n"
        "- Every example_sentence_cn must be Simplified Chinese.\n"
        "- The German sentence must include the word or a common German form of it.\n"
        "- Interpret each word according to the scope above when choosing meanings and translations.\n"
        "- If needs_meaning is true, provide only a concise dictionary-style Chinese meaning in the meaning field.\n"
        "- The meaning field must be a short Chinese word or phrase, not a definition, explanation, example, or sentence.\n"
        "- The Chinese translation must be fluent Simplified Chinese and must not leave German words untranslated.\n"
        "- If needs_meaning is false, omit meaning or return an empty meaning string.\n"
        "- Keep each sentence under 16 German words.\n"
        "- Do not explain anything.\n"
        'Return JSON exactly like: {"items": [{"batch_index": 1, "example_sentence": "...", '
        '"example_sentence_cn": "...", "meaning": "..."}]}'
    )


def build_correction_prompt(entry: WordEntry, scope: str = "", language: str = "") -> str:
    if _target_language(language) == "德语":
        return build_german_correction_prompt(entry, scope)
    return build_english_correction_prompt(entry, scope)


def build_english_correction_prompt(entry: WordEntry, scope: str = "") -> str:
    normalized_word = normalize_target_term(entry.word)
    scope_text = build_english_scope_text(scope)
    return (
        "Check whether this vocabulary record needs a correction.\n"
        f"{scope_text}"
        f"Original English: {entry.word}\n"
        f"Chinese meaning: {entry.meaning or '(empty)'}\n"
        f"Word forms: {entry.forms or '(empty)'}\n"
        f"Normalized English candidate: {normalized_word}\n"
        "Requirements:\n"
        "- Only check two things: whether the English word/phrase matches the Chinese meaning, and whether the English spelling/spacing/hyphenation is clearly wrong.\n"
        "- If the English and Chinese meaning already match and the English spelling/format is acceptable, set should_update to false and return the original English unchanged.\n"
        "- Only set should_update to true when there is a clear error and the corrected English word/phrase is unambiguous.\n"
        "- Fix obvious English formatting errors such as underscores used as spaces.\n"
        "- If the original English contains underscores and they are the only issue, replace them with spaces, not hyphens.\n"
        "- Do not rewrite the word for style, preference, broader terminology, or a more common synonym.\n"
        "- Do not change, translate, validate, or comment on the Chinese meaning.\n"
        "- Do not change, validate, or comment on word forms except as context for detecting an obvious English headword error.\n"
        "- Do not explain anything outside JSON.\n"
        'Return JSON exactly like: {"should_update": false, "corrected_word": "...", "note": "..."}'
    )


def build_german_correction_prompt(entry: WordEntry, scope: str = "") -> str:
    normalized_word = normalize_target_term(entry.word)
    scope_text = build_german_scope_text(scope)
    return (
        "Check whether this German vocabulary record needs a correction.\n"
        f"{scope_text}"
        f"Original German: {entry.word}\n"
        f"Chinese meaning: {entry.meaning or '(empty)'}\n"
        f"German word forms: {entry.forms or '(empty)'}\n"
        f"Normalized German candidate: {normalized_word}\n"
        "Requirements:\n"
        "- Only check two things: whether the German word/phrase matches the Chinese meaning, and whether the German spelling/spacing/hyphenation/capitalization is clearly wrong.\n"
        "- If the German word and Chinese meaning already match and the German spelling/format is acceptable, set should_update to false and return the original German unchanged.\n"
        "- Only set should_update to true when there is a clear error and the corrected German word/phrase is unambiguous.\n"
        "- Fix obvious German formatting errors such as underscores used as spaces.\n"
        "- If the original German contains underscores and they are the only issue, replace them with spaces, not hyphens.\n"
        "- Do not rewrite the word for style, preference, broader terminology, or a more common synonym.\n"
        "- Do not change, translate, validate, or comment on the Chinese meaning.\n"
        "- Do not change, validate, or comment on word forms except as context for detecting an obvious German headword error.\n"
        "- Do not explain anything outside JSON.\n"
        'Return JSON exactly like: {"should_update": false, "corrected_word": "...", "note": "..."}'
    )


def build_explanation_prompt(entry: WordEntry, scope: str = "", language: str = "") -> str:
    if _target_language(language) == "德语":
        return build_german_explanation_prompt(entry, scope)
    return build_english_explanation_prompt(entry, scope)


def build_english_explanation_prompt(entry: WordEntry, scope: str = "") -> str:
    scope_text = build_english_scope_text(scope)
    domain_text = scope.strip() or "(empty)"
    return (
        "Explain this vocabulary item for a Chinese learner.\n"
        "Target vocabulary language: 英语\n"
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


def build_german_explanation_prompt(entry: WordEntry, scope: str = "") -> str:
    scope_text = build_german_scope_text(scope)
    domain_text = scope.strip() or "(empty)"
    return (
        "Explain this German vocabulary item for a Chinese learner.\n"
        "Target vocabulary language: 德语\n"
        f"{scope_text}"
        f"German word or phrase: {entry.word}\n"
        f"Chinese meaning in the app, for reference only: {entry.meaning or '(empty)'}\n"
        f"German word forms: {entry.forms or '(empty)'}\n"
        f"German example sentence: {entry.example_sentence or '(empty)'}\n"
        f"Example Chinese translation: {entry.example_sentence_cn or '(empty)'}\n"
        f"Required domain for domain-specific usage: {domain_text}\n"
        "Requirements:\n"
        "- Write in Simplified Chinese.\n"
        "- Keep it concise but useful.\n"
        "- Structure the explanation in exactly three labeled sections: 意思, 常规用法, 领域用法.\n"
        "- In 意思, give the meaning of the German word or phrase itself, using the app's Chinese meaning only as a reference when helpful.\n"
        "- In 常规用法, write the explanation in Chinese, but explicitly explain the German usage of the target word or phrase: grammar role, gender/plural or verb forms when relevant, common German collocations, register, nuance, and at least one natural German phrase when useful.\n"
        "- In 领域用法, write the explanation in Chinese, but explicitly explain how the German target word or phrase is used in the required domain/scope, including common German technical collocations or term patterns when useful; if no domain is provided, say that no specific domain was set and give the most likely subject-area usage.\n"
        "- If an example sentence is provided, explain the German usage in that sentence.\n"
        "- Do not explain how the Chinese translation is used in Chinese.\n"
        "- Do not make 常规用法 or 领域用法 only a Chinese definition; both sections must mention German usage.\n"
        "- Do not treat the Chinese meaning as the target word; it is only a reference to disambiguate the vocabulary item.\n"
        "- Do not invent a different headword.\n"
        "- Keep the three section labels visible in the explanation text.\n"
        'Return JSON exactly like: {"explanation": "..."}'
    )


def build_scope_text(scope: str) -> str:
    return build_english_scope_text(scope)


def build_english_scope_text(scope: str) -> str:
    cleaned = scope.strip()
    if not cleaned:
        return ""
    return (
        f"Scope/domain: {cleaned}\n"
        "Use this scope to resolve ambiguous English terms. "
        "For example, decide whether an English term should stay hyphenated, "
        "spaced, or joined based on this scope.\n"
    )


def build_german_scope_text(scope: str) -> str:
    cleaned = scope.strip()
    if not cleaned:
        return ""
    return (
        f"Scope/domain: {cleaned}\n"
        "Use this scope to resolve ambiguous German terms. "
        "For example, decide the intended German meaning, form, or compound usage based on this scope.\n"
    )


def normalize_english_term(text: str) -> str:
    return normalize_target_term(text)


def normalize_target_term(text: str) -> str:
    normalized = re.sub(r"[_]+", " ", text.strip())
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def parse_response(
    content: str,
    require_meaning: bool = False,
    language: str = "",
    entry: WordEntry | None = None,
) -> GeneratedExample:
    data = _json_object(content)
    example_sentence = str(data.get("example_sentence", "")).strip()
    example_sentence_cn = str(data.get("example_sentence_cn", "")).strip()
    meaning = str(data.get("meaning", "")).strip()
    _validate_generated_example(example_sentence, example_sentence_cn, content, language=language, entry=entry)
    if require_meaning and not _has_cjk(meaning):
        raise RuntimeError(f"模型返回缺少中文释义字段：{content}")
    return GeneratedExample(example_sentence, example_sentence_cn, meaning)


def parse_batch_response(
    content: str,
    entries: list[WordEntry],
    language: str = "",
) -> list[tuple[WordEntry, GeneratedExample]]:
    data = _json_batch_items(content)
    by_index: dict[int, dict] = {}
    valid_indexes = set(range(1, len(entries) + 1))
    for item in data:
        if not isinstance(item, dict):
            raise RuntimeError(f"模型批量返回包含非对象条目：{content}")
        try:
            batch_index = int(item.get("batch_index", 0))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"模型批量返回缺少有效 batch_index：{content}") from error
        if batch_index not in valid_indexes:
            raise RuntimeError(f"模型批量返回 batch_index 越界：{content}")
        if batch_index in by_index:
            raise RuntimeError(f"模型批量返回重复 batch_index：{content}")
        by_index[batch_index] = item

    results: list[tuple[WordEntry, GeneratedExample]] = []
    for index, entry in enumerate(entries, start=1):
        item = by_index.get(index)
        if item is None:
            raise RuntimeError(f"模型批量返回缺少第 {index} 条：{content}")
        example_sentence = str(item.get("example_sentence", "")).strip()
        example_sentence_cn = str(item.get("example_sentence_cn", "")).strip()
        meaning = str(item.get("meaning", "")).strip()
        _validate_generated_example(example_sentence, example_sentence_cn, content, index, language, entry)
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
    language: str = "",
    entry: WordEntry | None = None,
) -> None:
    label = f"第 {batch_index} 条" if batch_index is not None else ""
    target_label = "德语" if _target_language(language) == "德语" else "英文"
    if not example_sentence or not example_sentence_cn:
        raise RuntimeError(f"模型返回{label}缺少例句字段：{content}")
    if _has_cjk(example_sentence) or not re.search(r"[A-Za-z]", example_sentence):
        raise RuntimeError(f"模型返回{label}{target_label}例句不是{target_label}：{content}")
    if not _has_cjk(example_sentence_cn):
        raise RuntimeError(f"模型返回{label}中文例句缺少中文：{content}")
    if entry is not None and not _example_mentions_entry(example_sentence, entry):
        raise RuntimeError(f"模型返回{label}例句未包含目标词 {entry.word}：{content}")


def _example_mentions_entry(example_sentence: str, entry: WordEntry) -> bool:
    candidates = [entry.word]
    candidates.extend(part for part in re.split(r"[;,，；]", entry.forms or "") if part.strip())
    candidates.extend(_simple_english_variants(candidate) for candidate in list(candidates))
    return any(_text_contains_term(example_sentence, candidate) for candidate in candidates)


def _simple_english_variants(term: str) -> str:
    normalized = normalize_target_term(term).casefold()
    if not re.fullmatch(r"[a-z]+", normalized):
        return ""
    if normalized.endswith("y") and len(normalized) > 1 and normalized[-2] not in "aeiou":
        return f"{normalized[:-1]}ies"
    if normalized.endswith(("s", "x", "z", "ch", "sh")):
        return f"{normalized}es"
    return f"{normalized}s"


def _text_contains_term(text: str, term: str) -> bool:
    normalized_term = normalize_target_term(term).casefold()
    if not normalized_term:
        return False
    normalized_text = normalize_target_term(text).casefold()
    escaped = re.escape(normalized_term)
    if re.fullmatch(r"[a-z0-9]+", normalized_term):
        return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", normalized_text) is not None
    return normalized_term in normalized_text


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def parse_correction_response(content: str, original: WordEntry) -> GeneratedCorrection:
    data = _json_object(content)
    corrected_word = str(data.get("corrected_word", "")).strip()
    note = str(data.get("note", "")).strip()
    should_update = _bool_value(data.get("should_update", False))
    normalized_word = normalize_english_term(original.word)
    if "_" in original.word:
        corrected_word = normalized_word
        should_update = corrected_word != original.word.strip()
    if not corrected_word:
        corrected_word = normalized_word
    if corrected_word == original.word.strip():
        should_update = False
    return GeneratedCorrection(corrected_word, note, should_update)


def _bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "是", "需要"}
    return bool(value)


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
