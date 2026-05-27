from __future__ import annotations

import csv
import json
import logging
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from PySide6.QtPdf import QPdfDocument

from wordpycket.domain.entities import WordEntry
from wordpycket.infrastructure.csv_importer import CsvColumnSchema

try:
    import ftfy
except ImportError:  # pragma: no cover - optional dependency in portable builds
    ftfy = None

try:
    from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phraser, Phrases
except ImportError:  # pragma: no cover - phrase extraction is optional
    ENGLISH_CONNECTOR_WORDS = frozenset()
    Phraser = None
    Phrases = None


LOGGER = logging.getLogger(__name__)


def run_pdf_import_isolated(
    pdf_path: Path,
    csv_path: Path,
    vocabulary_cleaner: VocabularyCleaner | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
) -> "PdfVocabularyImportResult":
    if vocabulary_cleaner is not None:
        if hasattr(vocabulary_cleaner, "isolated_pdf_import_environment") and hasattr(
            vocabulary_cleaner,
            "isolated_pdf_import_payload",
        ):
            payload = vocabulary_cleaner.isolated_pdf_import_payload(pdf_path, csv_path)
            env = vocabulary_cleaner.isolated_pdf_import_environment()
            return _run_pdf_import_child(payload, env, csv_path, progress_callback)
        return PdfVocabularyImporter(
            pdf_path,
            csv_path,
            vocabulary_cleaner=vocabulary_cleaner,
            progress_callback=progress_callback,
        ).build()

    payload = json.dumps(
        {
            "pdf_path": str(pdf_path),
            "csv_path": str(csv_path),
        },
        ensure_ascii=False,
    )
    env = os.environ.copy()
    env["WORDPYCKET_PDF_CHILD"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    python_paths = [path for path in sys.path if path]
    existing_python_path = env.get("PYTHONPATH")
    if existing_python_path:
        python_paths.append(existing_python_path)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    return _run_pdf_import_child(payload, env, csv_path, progress_callback)


def _run_pdf_import_child(
    payload: str,
    env: Mapping[str, str],
    csv_path: Path,
    progress_callback: Callable[[str, int], None] | None = None,
) -> "PdfVocabularyImportResult":
    command = [sys.executable, "-m", "wordpycket.infrastructure.pdf_vocabulary_importer"]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(payload)
    process.stdin.close()

    result: dict[str, Any] | None = None
    output_lines: list[str] = []
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        output_lines.append(line)
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict):
            continue
        if message.get("type") == "progress" and progress_callback is not None:
            progress_callback(str(message.get("message", "")), int(message.get("percent", 0)))
        elif message.get("type") == "result":
            result = message

    returncode = process.wait()
    if returncode != 0:
        detail = "\n".join(output_lines[-20:]).strip()
        if detail:
            raise RuntimeError(f"PDF 解析子进程退出代码 {returncode}：{detail}")
        raise RuntimeError(f"PDF 解析子进程退出代码 {returncode}。")
    if result is None:
        detail = "\n".join(output_lines[-20:]).strip()
        raise RuntimeError(f"PDF 解析子进程没有返回结果。{detail}")
    return PdfVocabularyImportResult(
        entries=[],
        language=str(result.get("language", "")),
        csv_path=Path(str(result.get("csv_path", csv_path))),
    )


def _main() -> None:
    payload = json.loads(sys.stdin.read())

    def progress(message: str, percent: int) -> None:
        print(
            json.dumps({"type": "progress", "message": message, "percent": percent}, ensure_ascii=False),
            flush=True,
        )

    vocabulary_cleaner = None
    if payload.get("use_llm_cleanup"):
        from wordpycket.infrastructure.example_generator import LocalLlmExampleGenerator

        vocabulary_cleaner = LocalLlmExampleGenerator(Path(payload["model_dir"]))

    result = PdfVocabularyImporter(
        Path(payload["pdf_path"]),
        Path(payload["csv_path"]),
        vocabulary_cleaner=vocabulary_cleaner,
        progress_callback=progress,
    ).build()
    print(
        json.dumps(
            {
                "type": "result",
                "language": result.language,
                "csv_path": str(result.csv_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


@dataclass(frozen=True)
class PdfVocabularyImportResult:
    entries: list[WordEntry]
    language: str
    csv_path: Path


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    label: str
    spacy_models: tuple[str, ...]
    markers: frozenset[str]
    stopwords: frozenset[str]
    allowed_pos: frozenset[str] = frozenset({"NOUN", "VERB", "ADJ", "PROPN"})
    phrase_extraction: bool = True


@dataclass(frozen=True)
class TextToken:
    surface: str
    lemma: str
    pos: str
    language: str
    document_id: str


@dataclass
class FrequencyBucket:
    frequency: int = 0
    document_ids: set[str] = field(default_factory=set)
    forms: Counter[str] = field(default_factory=Counter)
    language_counts: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class CorpusTerm:
    canonical: str
    frequency: int
    document_frequency: int
    forms: tuple[str, ...]
    language: str
    chinese: str = ""


class VocabularyCleaner(Protocol):
    def clean_pdf_vocabulary_entries(
        self,
        entries: list[WordEntry],
        language: str,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> list[WordEntry]: ...


class DeterministicMultilingualFrequencyEngine:
    """Deterministic corpus frequency analyzer. No LLMs, APIs, or transformers."""

    ENGLISH_SCHEMA = CsvColumnSchema("英语", ("Index", "English", "Chinese", "Frequency", "Forms"))
    GERMAN_SCHEMA = CsvColumnSchema("德语", ("Index", "Deutsch", "Chinesisch", "Häufigkeit", "Formen"))
    STRICT_SCHEMA = ENGLISH_SCHEMA
    SCHEMAS_BY_CODE: Mapping[str, CsvColumnSchema] = {
        "en": ENGLISH_SCHEMA,
        "de": GERMAN_SCHEMA,
    }
    _LATIN_TOKEN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*")
    _CJK_RUN = re.compile(r"[\u3400-\u9fff]+")
    _OCR_GLUE = re.compile(r"(?<=[a-z])-\s*\n\s*(?=[a-z])", re.IGNORECASE)
    _WHITESPACE = re.compile(r"[ \t\r\f\v]+")
    _REFERENCE_ONLY = re.compile(r"^\[?\d+(?:[-,]\s*\d+)*\]?$")
    _CITATION = re.compile(r"\[[0-9,\s-]+\]|\([0-9,\s-]+\)")
    _URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    _EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
    _NOISE_LINE = re.compile(r"^[\W\d_]+$")
    _SPACY_PIPELINES: dict[str, Any | None] = {}
    _KNOWN_PHRASES = frozenset(
        {
            "artificial intelligence",
            "bayes theorem",
            "bayes' theorem",
            "deep learning",
            "fourier transform",
            "gaussian distribution",
            "laplace law",
            "laplace transform",
            "machine learning",
            "natural language",
            "neural network",
            "random forest",
            "reinforcement learning",
            "support vector",
        }
    )
    _TECHNICAL_HEAD_NOUNS = frozenset(
        {
            "algorithm",
            "analysis",
            "architecture",
            "condition",
            "density",
            "distribution",
            "equation",
            "feature",
            "field",
            "function",
            "gradient",
            "kernel",
            "law",
            "layer",
            "matrix",
            "method",
            "model",
            "network",
            "operator",
            "parameter",
            "pressure",
            "probability",
            "signal",
            "state",
            "structure",
            "temperature",
            "tension",
            "tensor",
            "theorem",
            "transform",
            "vector",
            "wave",
            "weight",
        }
    )
    _TECHNICAL_MODIFIERS = frozenset(
        {
            "adaptive",
            "boundary",
            "central",
            "conditional",
            "continuous",
            "convex",
            "deep",
            "differential",
            "discrete",
            "dynamic",
            "finite",
            "linear",
            "local",
            "neural",
            "normal",
            "optimal",
            "partial",
            "probability",
            "random",
            "spectral",
            "statistical",
            "stochastic",
            "surface",
            "thermal",
        }
    )
    _EPONYM_MODIFIERS = frozenset(
        {
            "bayes",
            "bayesian",
            "euler",
            "eulerian",
            "fourier",
            "gauss",
            "gaussian",
            "hamilton",
            "hamiltonian",
            "laplace",
            "markov",
            "newton",
            "newtonian",
            "turing",
        }
    )
    _CANONICAL_PHRASES = {
        "bayes theorem": "bayes' theorem",
    }
    _TECHNICAL_SINGLE_TERMS = frozenset(
        {
            "adapter",
            "aggregate",
            "alignment",
            "architecture",
            "corpus",
            "domain",
            "embedding",
            "extraction",
            "frequency",
            "gradient",
            "inference",
            "matrix",
            "normalization",
            "parser",
            "pipeline",
            "repository",
            "retrieval",
            "service",
            "token",
            "translation",
            "vector",
            "vocabulary",
        }
    )
    _DOCUMENT_ARTIFACT_TERMS = frozenset(
        {
            "above",
            "batch",
            "below",
            "candidate",
            "cleanup",
            "document",
            "figure",
            "footer",
            "header",
            "ideal",
            "include",
            "label",
            "note",
            "page",
            "pass",
            "purpose",
            "row",
            "section",
            "table",
            "target",
            "test",
            "validation",
        }
    )
    _COMMON_PERSON_NAME_PARTS = frozenset(
        {
            "alice",
            "bob",
            "brown",
            "david",
            "davis",
            "emily",
            "hua",
            "john",
            "johnson",
            "laura",
            "li",
            "mary",
            "michael",
            "miller",
            "moore",
            "sarah",
            "smith",
            "taylor",
            "william",
            "williams",
            "wilson",
        }
    )

    _EN_STOPWORDS = frozenset(
        "a an and are as at be by for from has have had he her his i in is it its of on or our she that the their "
        "them they this to was were with without we you your not no yes if then than into over under between within "
        "above before below but can could during may might must should would which while"
        .split()
    )
    _DE_STOPWORDS = frozenset(
        "aber als am an auf aus bei bin bis bist da dadurch daher darum das dass dein deine dem den der des dessen "
        "die dies diese diesem diesen dieser dieses doch dort du durch ein eine einem einen einer eines er es euer "
        "für hatte hatten hattest hattet hier hinter ich ihr ihre im in ist ja jede jedem jeden jeder jedes jener "
        "jenes jetzt kann kannst können könnt machen mein meine mit muss musst müssen müsst nach nicht nichts noch "
        "nun oder seid sein seine sich sie sind soll sollen sollst sonst soweit sowie und unser unsere unter vom von "
        "vor wann warum was weiter weitere wenn wer werde werden werdet weshalb wie wieder wieso wir wird wirst wo "
        "woher wohin zu zum zur über"
        .split()
    )
    PROFILES: Mapping[str, LanguageProfile] = {
        "en": LanguageProfile(
            code="en",
            label="英语",
            spacy_models=("en_core_web_sm",),
            markers=frozenset(
                "the and of to in that is for was with this which should not become generated named mentions improves"
                .split()
            ),
            stopwords=_EN_STOPWORDS,
        ),
        "de": LanguageProfile(
            code="de",
            label="德语",
            spacy_models=("de_core_news_sm",),
            markers=frozenset("der die das und ist nicht mit ein eine von zu".split()),
            stopwords=_DE_STOPWORDS,
        ),
    }

    _GERMAN_LEMMA_OVERRIDES = {
        "bin": "sein",
        "bist": "sein",
        "ist": "sein",
        "sind": "sein",
        "seid": "sein",
        "war": "sein",
        "waren": "sein",
        "gewesen": "sein",
        "hat": "haben",
        "hast": "haben",
        "habt": "haben",
        "hatte": "haben",
        "hatten": "haben",
    }
    _ENGLISH_LEMMA_OVERRIDES = {
        "am": "be",
        "are": "be",
        "bayes": "bayes",
        "is": "be",
        "was": "be",
        "were": "be",
        "been": "be",
        "being": "be",
        "has": "have",
        "had": "have",
        "having": "have",
        "does": "do",
        "did": "do",
        "done": "do",
        "doing": "do",
        "ran": "run",
        "corpus": "corpus",
        "embedding": "embedding",
        "normalizing": "normalize",
    }

    def __init__(
        self,
        language_by_file: Mapping[str, str] | None = None,
        dictionary_path: Path | None = None,
        enable_phrases: bool = True,
    ) -> None:
        self._language_by_file = dict(language_by_file or {})
        self._dictionary = self._load_dictionary(dictionary_path)
        self._enable_phrases = enable_phrases

    @classmethod
    def from_config(cls, config_path: Path | None) -> "DeterministicMultilingualFrequencyEngine":
        if config_path is None or not config_path.exists():
            return cls()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        dictionary = Path(data["dictionary"]) if data.get("dictionary") else None
        return cls(
            language_by_file=data.get("language_by_file", {}),
            dictionary_path=dictionary,
            enable_phrases=bool(data.get("enable_phrases", True)),
        )

    def analyze_folder(self, folder: Path, max_entries: int = 10000) -> tuple[list[WordEntry], str, CsvColumnSchema]:
        documents = list(self._iter_text_files(folder))
        if not documents:
            raise RuntimeError(f"没有找到可分析的 .txt 文件：{folder}")
        terms = self._analyze_documents(documents, max_entries)
        language = self._dominant_language(terms)
        return self._entries_from_terms(terms), language, self._schema_for_language_label(language)

    def analyze_text(self, text: str, max_entries: int = 10000, document_id: str = "pdf") -> tuple[list[WordEntry], str, CsvColumnSchema]:
        terms = self._analyze_documents(((document_id, text),), max_entries)
        language = self._dominant_language(terms)
        return self._entries_from_terms(terms), language, self._schema_for_language_label(language)

    def _iter_text_files(self, folder: Path) -> Iterator[tuple[str, str]]:
        for path in sorted(folder.rglob("*.txt"), key=lambda item: str(item).lower()):
            try:
                yield str(path), path.read_text(encoding="utf-8", errors="replace")
            except OSError as error:
                LOGGER.warning("跳过无法读取的文本文件 %s: %s", path, error)

    def _analyze_documents(self, documents: Iterable[tuple[str, str]], max_entries: int) -> list[CorpusTerm]:
        buckets: dict[str, FrequencyBucket] = defaultdict(FrequencyBucket)
        phrase_training: dict[str, list[list[str]]] = defaultdict(list)
        tokenized_docs: list[list[TextToken]] = []

        for document_id, raw_text in documents:
            cleaned = self.clean_text(raw_text)
            language = self._configured_language(document_id) or self.detect_language(cleaned)
            tokens = list(self.tokenize(cleaned, language, document_id))
            tokenized_docs.append(tokens)
            phrase_training[language].append([token.lemma for token in tokens])
            for token in tokens:
                self._add_token(buckets, token.lemma, token.surface, token.language, token.document_id)

        if self._enable_phrases:
            for tokens in tokenized_docs:
                self._add_phrases(buckets, tokens, phrase_training.get(tokens[0].language, []) if tokens else [])
            self._remove_known_phrase_constituents(buckets)

        terms = [self._term_from_bucket(canonical, bucket) for canonical, bucket in buckets.items()]
        terms.sort(key=lambda item: (-item.frequency, item.canonical.count(" "), item.canonical))
        return terms[:max_entries]

    def clean_text(self, text: str) -> str:
        if ftfy is not None:
            text = ftfy.fix_text(text)
        text = self._OCR_GLUE.sub("", text)
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = self._WHITESPACE.sub(" ", raw_line).strip()
            if not line:
                continue
            if self._REFERENCE_ONLY.fullmatch(line) or self._NOISE_LINE.fullmatch(line):
                continue
            if self._looks_like_code(line) or self._looks_like_formula_or_noise(line):
                continue
            line = self._URL.sub(" ", line)
            line = self._EMAIL.sub(" ", line)
            line = self._CITATION.sub(" ", line)
            line = self._WHITESPACE.sub(" ", line).strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    def detect_language(self, text: str) -> str:
        latin_words = [match.group(0).lower() for match in self._LATIN_TOKEN.finditer(text)]
        if not latin_words:
            return "en"

        counts = Counter(latin_words)
        scores = {}
        for code, profile in self.PROFILES.items():
            marker_hits = sum(counts[word] for word in profile.markers)
            lexical_hits = sum(
                count for word, count in counts.items()
                if word not in profile.stopwords and self._is_latin_candidate(word)
            )
            scores[code] = marker_hits * 8 + lexical_hits
        priority = {"en": 2, "de": 1}
        return max(scores, key=lambda code: (scores[code], priority.get(code, 0)))

    def tokenize(self, text: str, language: str, document_id: str) -> Iterator[TextToken]:
        profile = self.PROFILES.get(language, self.PROFILES["en"])
        nlp = self._spacy_pipeline(profile)
        if nlp is not None:
            yield from self._spacy_tokens(nlp, text, profile, document_id)
            return
        yield from self._fallback_latin_tokens(text, profile, document_id)

    def _spacy_tokens(self, nlp: Any, text: str, profile: LanguageProfile, document_id: str) -> Iterator[TextToken]:
        for doc in nlp.pipe(self._text_batches(text), batch_size=32):
            for token in doc:
                surface = token.text.strip()
                if self._looks_like_identifier_surface(surface):
                    continue
                lemma = (token.lemma_ or surface).strip().lower()
                surface_lower = surface.lower()
                if profile.code in {"en", "de"} and surface_lower != lemma:
                    lemma = self._fallback_lemma(surface_lower, profile.code)
                pos = token.pos_ or ""
                if self._keep_token(surface, lemma, pos, profile):
                    yield TextToken(surface=surface_lower, lemma=lemma, pos=pos, language=profile.code, document_id=document_id)

    def _fallback_latin_tokens(self, text: str, profile: LanguageProfile, document_id: str) -> Iterator[TextToken]:
        for match in self._LATIN_TOKEN.finditer(text):
            raw_surface = match.group(0)
            if self._looks_like_identifier_surface(raw_surface):
                continue
            surface = raw_surface.lower()
            if profile.code == "en" and surface in self._COMMON_PERSON_NAME_PARTS:
                continue
            lemma = self._fallback_lemma(surface, profile.code)
            if self._keep_token(surface, lemma, "", profile):
                yield TextToken(surface=surface, lemma=lemma, pos="", language=profile.code, document_id=document_id)

    @staticmethod
    def _text_batches(text: str, max_chars: int = 50000) -> Iterator[str]:
        current: list[str] = []
        current_size = 0
        for line in text.splitlines():
            if current and current_size + len(line) > max_chars:
                yield "\n".join(current)
                current = []
                current_size = 0
            current.append(line)
            current_size += len(line)
        if current:
            yield "\n".join(current)

    def _add_phrases(
        self,
        buckets: dict[str, FrequencyBucket],
        tokens: list[TextToken],
        training_sentences: list[list[str]],
    ) -> None:
        if not tokens:
            return
        profile = self.PROFILES[tokens[0].language]
        if not profile.phrase_extraction or len(tokens) < 2:
            return
        rule_phrases = self._add_rule_phrases(buckets, tokens, profile)
        if Phrases is None or Phraser is None or len(training_sentences) < 1:
            return
        try:
            phrases = Phrases(
                training_sentences,
                min_count=2,
                threshold=10.0,
                connector_words=ENGLISH_CONNECTOR_WORDS if profile.code == "en" else frozenset(),
            )
            phraser = Phraser(phrases)
        except Exception as error:  # pragma: no cover - gensim may be unavailable in some builds
            LOGGER.warning("短语抽取失败，已跳过：%s", error)
            return
        lemmas = [token.lemma for token in tokens]
        surfaces = [token.surface for token in tokens]
        transformed = list(phraser[lemmas])
        cursor = 0
        for item in transformed:
            width = item.count("_") + 1
            if width > 1:
                canonical = item.replace("_", " ")
                parts = canonical.split()
                if len(set(parts)) != len(parts):
                    cursor += width
                    continue
                if canonical in rule_phrases:
                    cursor += width
                    continue
                surface = " ".join(surfaces[cursor: cursor + width])
                self._add_token(buckets, canonical, surface, profile.code, tokens[cursor].document_id)
            cursor += width

    def _add_rule_phrases(
        self,
        buckets: dict[str, FrequencyBucket],
        tokens: list[TextToken],
        profile: LanguageProfile,
    ) -> set[str]:
        added: set[str] = set()
        counts = Counter(
            (tokens[index].lemma, tokens[index + 1].lemma)
            for index in range(0, len(tokens) - 1)
        )
        for index in range(0, len(tokens) - 1):
            first = tokens[index].lemma
            second = tokens[index + 1].lemma
            phrase = f"{first} {second}"
            surface_phrase = f"{tokens[index].surface} {tokens[index + 1].surface}"
            count = counts[(first, second)]
            if not self._is_rule_phrase(first, second, phrase, surface_phrase, count):
                continue
            canonical = self._CANONICAL_PHRASES.get(phrase, surface_phrase if surface_phrase in self._KNOWN_PHRASES else phrase)
            surface = "Bayes' theorem" if canonical == "bayes' theorem" else surface_phrase
            self._add_token(buckets, canonical, surface, profile.code, tokens[index].document_id)
            added.add(canonical)
        return added

    def _is_rule_phrase(self, first: str, second: str, phrase: str, surface_phrase: str, count: int) -> bool:
        if first == second:
            return False
        if phrase in self._KNOWN_PHRASES or surface_phrase in self._KNOWN_PHRASES:
            return True
        if first in self._EPONYM_MODIFIERS and second in self._TECHNICAL_HEAD_NOUNS:
            return True
        if first in self._TECHNICAL_MODIFIERS and second in self._TECHNICAL_HEAD_NOUNS:
            return True
        return count >= 2 and second in self._TECHNICAL_HEAD_NOUNS

    @staticmethod
    def _add_token(
        buckets: dict[str, FrequencyBucket],
        canonical: str,
        surface: str,
        language: str,
        document_id: str,
    ) -> None:
        bucket = buckets[canonical]
        bucket.frequency += 1
        bucket.document_ids.add(document_id)
        bucket.forms[surface] += 1
        bucket.language_counts[language] += 1

    def _term_from_bucket(self, canonical: str, bucket: FrequencyBucket) -> CorpusTerm:
        language = bucket.language_counts.most_common(1)[0][0] if bucket.language_counts else "en"
        forms = tuple(form for form, _count in sorted(bucket.forms.items(), key=lambda item: (-item[1], item[0])))
        mapped = self._dictionary.get(canonical, {})
        english = str(mapped.get("en") or canonical).strip()
        chinese = str(mapped.get("zh") or "").strip()
        return CorpusTerm(
            canonical=english,
            chinese=chinese,
            frequency=bucket.frequency,
            document_frequency=len(bucket.document_ids),
            forms=forms,
            language=language,
        )

    @staticmethod
    def _remove_known_phrase_constituents(buckets: dict[str, FrequencyBucket]) -> None:
        phrase_parts = {
            "bayes' theorem": ("bayes", "theorem"),
            "machine learning": ("machine", "learn"),
        }
        for phrase, parts in phrase_parts.items():
            if phrase not in buckets:
                continue
            for part in parts:
                buckets.pop(part, None)

    @staticmethod
    def _entries_from_terms(terms: list[CorpusTerm]) -> list[WordEntry]:
        entries = [
            WordEntry(
                source_index=index,
                word=term.canonical,
                meaning=term.chinese,
                frequency=term.frequency,
                forms=";".join(term.forms),
            )
            for index, term in enumerate(terms, start=1)
            if term.canonical
        ]
        if not entries:
            raise RuntimeError("语料中没有识别到可统计的词或词组。")
        return entries

    def _dominant_language(self, terms: list[CorpusTerm]) -> str:
        counts: Counter[str] = Counter()
        for term in terms:
            counts[term.language] += term.frequency
        code = counts.most_common(1)[0][0] if counts else "en"
        return self.PROFILES.get(code, self.PROFILES["en"]).label

    def _schema_for_language_label(self, language: str) -> CsvColumnSchema:
        for schema in self.SCHEMAS_BY_CODE.values():
            if schema.language == language:
                return schema
        return self.ENGLISH_SCHEMA

    @classmethod
    def _spacy_pipeline(cls, profile: LanguageProfile):
        if profile.code in cls._SPACY_PIPELINES:
            return cls._SPACY_PIPELINES[profile.code]
        try:
            import spacy
        except ImportError:
            cls._SPACY_PIPELINES[profile.code] = None
            return None
        for model_name in profile.spacy_models:
            try:
                cls._SPACY_PIPELINES[profile.code] = spacy.load(
                    model_name,
                    disable=["parser", "ner", "textcat"],
                    exclude=["transformer"],
                )
                return cls._SPACY_PIPELINES[profile.code]
            except (OSError, ValueError):
                continue
        cls._SPACY_PIPELINES[profile.code] = None
        return None

    def _keep_token(self, surface: str, lemma: str, pos: str, profile: LanguageProfile) -> bool:
        candidate = lemma.strip().lower()
        if not candidate or candidate in profile.stopwords:
            return False
        if surface in profile.stopwords:
            return False
        if profile.code == "en" and candidate in self._COMMON_PERSON_NAME_PARTS:
            return False
        if profile.code == "en" and candidate in self._DOCUMENT_ARTIFACT_TERMS:
            return False
        if surface.isnumeric() or candidate.isnumeric():
            return False
        if pos and pos not in profile.allowed_pos:
            return False
        return self._is_latin_candidate(candidate)

    @staticmethod
    def _is_latin_candidate(word: str) -> bool:
        if len(word) <= 1 or len(word) > 40:
            return False
        if any(char.isdigit() for char in word) or "_" in word:
            return False
        if word.count("-") > 2 or word.count("'") > 1:
            return False
        letters = [char for char in word if char.isalpha()]
        if not letters:
            return False
        vowels = set("aeiouyäöüàâæçéèêëîïôœùûüáéíóúñãõ")
        return len(word) <= 3 or any(char in vowels for char in word)

    @staticmethod
    def _looks_like_identifier_surface(surface: str) -> bool:
        if "_" in surface or any(char.isdigit() for char in surface):
            return True
        if re.search(r"[a-z][A-Z]", surface):
            return True
        if len(surface) > 3 and surface.isupper():
            return True
        return False

    @classmethod
    def _fallback_lemma(cls, word: str, language: str) -> str:
        if language == "en":
            if word in cls._ENGLISH_LEMMA_OVERRIDES:
                return cls._ENGLISH_LEMMA_OVERRIDES[word]
            return cls._english_lemma(word)
        if language == "de":
            if word in cls._GERMAN_LEMMA_OVERRIDES:
                return cls._GERMAN_LEMMA_OVERRIDES[word]
            return cls._german_lemma(word)
        return word

    @staticmethod
    def _english_lemma(word: str) -> str:
        if word in DeterministicMultilingualFrequencyEngine._TECHNICAL_SINGLE_TERMS:
            return word
        if len(word) > 5 and word.endswith("ies"):
            return f"{word[:-3]}y"
        if len(word) > 5 and word.endswith("ves") and not word.endswith("oves"):
            return f"{word[:-3]}f"
        if len(word) >= 5 and word.endswith("ing"):
            base = word[:-3]
            if len(base) > 2 and base[-1] == base[-2]:
                base = base[:-1]
            return f"{base}e" if f"{base}e" in {"use", "make", "take", "write", "normalize"} else base
        if len(word) >= 4 and word.endswith("ed"):
            base = word[:-2]
            if len(base) > 2 and base[-1] == base[-2]:
                return base[:-1]
            return f"{base}e" if not base.endswith("e") else base
        if len(word) > 4 and word.endswith("es"):
            return word[:-2] if word.endswith(("ches", "shes", "sses", "xes", "zes", "oes")) else word[:-1]
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    @staticmethod
    def _german_lemma(word: str) -> str:
        for suffix in ("innen", "ungen", "heiten", "keiten", "lichkeit"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                return word[: -len(suffix)]
        for suffix in ("ern", "er", "en", "es", "e", "n", "s"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                return word[: -len(suffix)]
        return word

    def _configured_language(self, document_id: str) -> str | None:
        value = self._language_by_file.get(document_id) or self._language_by_file.get(Path(document_id).name)
        return value if value in self.PROFILES else None

    @staticmethod
    def _load_dictionary(dictionary_path: Path | None) -> dict[str, dict[str, str]]:
        if dictionary_path is None or not dictionary_path.exists():
            return {}
        data = json.loads(dictionary_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("dictionary.json 必须是对象映射。")
        dictionary: dict[str, dict[str, str]] = {}
        for key, value in data.items():
            if isinstance(value, str):
                dictionary[str(key).strip().lower()] = {"zh": value}
            elif isinstance(value, dict):
                dictionary[str(key).strip().lower()] = {
                    "en": str(value.get("en", "")).strip(),
                    "zh": str(value.get("zh", "")).strip(),
                }
        return dictionary

    @staticmethod
    def _looks_like_formula_or_noise(line: str) -> bool:
        if len(line) <= 2:
            return True
        letters = sum(1 for char in line if char.isalpha())
        digits = sum(1 for char in line if char.isdigit())
        operators = sum(1 for char in line if char in "=+*/∑∫√≈≤≥≠±∞∂∆→←↔^_|")
        punctuation = sum(1 for char in line if not char.isalnum() and not char.isspace())
        length = max(1, len(line))
        if letters == 0:
            return True
        if operators >= 2 and (operators + digits) / length > 0.18:
            return True
        if digits > letters and punctuation / length > 0.16:
            return True
        return punctuation / length > 0.45

    @staticmethod
    def _looks_like_code(line: str) -> bool:
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith(
            (
                "#include",
                "import ",
                "from ",
                "def ",
                "class ",
                "return ",
                "for ",
                "while ",
                "if ",
                "else:",
                "elif ",
                "try:",
                "except ",
                "public ",
                "private ",
                "protected ",
                "function ",
                "const ",
                "let ",
                "var ",
            )
        ):
            return True
        if re.search(r"\b[a-zA-Z_]\w*\s*=\s*[^=]", stripped):
            return True
        if re.search(r"\b[a-zA-Z_]\w*\s*\([^)]*\)\s*[{:]?$", stripped):
            return True
        code_chars = sum(1 for char in stripped if char in "{}[]();<>")
        operators = sum(1 for char in stripped if char in "=+*/%&|!^~")
        length = max(1, len(stripped))
        return code_chars >= 3 and (code_chars + operators) / length > 0.12


class PdfVocabularyImporter:
    def __init__(
        self,
        pdf_path: Path,
        csv_path: Path,
        max_entries: int = 10000,
        vocabulary_cleaner: VocabularyCleaner | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> None:
        self._pdf_path = pdf_path
        self._csv_path = csv_path
        self._max_entries = max_entries
        self._vocabulary_cleaner = vocabulary_cleaner
        self._progress_callback = progress_callback
        self._engine = DeterministicMultilingualFrequencyEngine()

    def build(self) -> PdfVocabularyImportResult:
        self._report_progress("读取 PDF 文本", 5)
        text = self.extract_text(self._pdf_path)
        self._report_progress("清洗文本并统计词频", 20)
        entries, language, schema = self.entries_from_text(text, self._max_entries)
        self._report_progress("生成 PDF 词表", 35)
        self.write_csv(entries, schema, self._csv_path)
        if self._vocabulary_cleaner is not None:
            self._report_progress("AI 检查 PDF 词表", 40)
            entries = self._vocabulary_cleaner.clean_pdf_vocabulary_entries(
                entries,
                language,
                progress_callback=self._progress_callback,
            )
            entries = self.reindex_entries(entries)
            if not entries:
                raise RuntimeError("AI 检查后没有保留可导入的词条。")
            self._report_progress("保存 AI 检查结果", 85)
            self.write_csv(entries, schema, self._csv_path)
        self._report_progress("PDF 词表生成完成", 90)
        return PdfVocabularyImportResult(entries=entries, language=language, csv_path=self._csv_path)

    def _report_progress(self, message: str, percent: int) -> None:
        if self._progress_callback is not None:
            self._progress_callback(message, percent)

    @staticmethod
    def extract_text(pdf_path: Path) -> str:
        document = QPdfDocument(None)
        error = document.load(str(pdf_path))
        if error != QPdfDocument.Error.None_:
            raise RuntimeError(f"PDF 无法读取：{pdf_path}。错误：{PdfVocabularyImporter._pdf_error_text(error)}")
        pages = [document.getAllText(page).text() for page in range(document.pageCount())]
        text = "\n".join(pages).strip()
        if not text:
            raise RuntimeError("PDF 没有可提取文本，可能是扫描图片 PDF。")
        return text

    @staticmethod
    def _pdf_error_text(error: QPdfDocument.Error) -> str:
        labels = {
            QPdfDocument.Error.Unknown: "未知错误",
            QPdfDocument.Error.DataNotYetAvailable: "数据尚不可用",
            QPdfDocument.Error.FileNotFound: "文件不存在",
            QPdfDocument.Error.InvalidFileFormat: "PDF 格式无效或文件损坏",
            QPdfDocument.Error.IncorrectPassword: "PDF 需要密码或密码错误",
            QPdfDocument.Error.UnsupportedSecurityScheme: "PDF 使用了不支持的安全/加密方案",
        }
        return labels.get(error, str(error))

    @classmethod
    def entries_from_text(cls, text: str, max_entries: int = 10000) -> tuple[list[WordEntry], str, CsvColumnSchema]:
        return DeterministicMultilingualFrequencyEngine().analyze_text(text, max_entries=max_entries)

    @classmethod
    def detect_language(cls, text: str) -> tuple[str, CsvColumnSchema]:
        engine = DeterministicMultilingualFrequencyEngine()
        code = engine.detect_language(engine.clean_text(text))
        language = engine.PROFILES.get(code, engine.PROFILES["en"]).label
        return language, engine._schema_for_language_label(language)

    @staticmethod
    def write_csv(entries: list[WordEntry], schema: CsvColumnSchema, csv_path: Path) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(schema.columns))
            writer.writeheader()
            for entry in entries:
                writer.writerow(
                    {
                        schema.source_index: entry.source_index,
                        schema.word: entry.word,
                        schema.meaning: entry.meaning,
                        schema.frequency: entry.frequency,
                        schema.forms: entry.forms,
                    }
                )

    @staticmethod
    def reindex_entries(entries: list[WordEntry]) -> list[WordEntry]:
        return [replace(entry, source_index=index) for index, entry in enumerate(entries, start=1)]


if __name__ == "__main__":
    _main()
