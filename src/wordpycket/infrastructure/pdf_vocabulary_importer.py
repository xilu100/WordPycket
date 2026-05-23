import csv
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from PySide6.QtPdf import QPdfDocument

from wordpycket.domain.entities import WordEntry
from wordpycket.infrastructure.csv_importer import CsvColumnSchema, WordFrequencyCsvImporter


@dataclass(frozen=True)
class PdfVocabularyImportResult:
    entries: list[WordEntry]
    language: str
    csv_path: Path


@dataclass(frozen=True)
class LanguageProfile:
    language: str
    schema: CsvColumnSchema
    stopwords: frozenset[str]
    markers: frozenset[str]


class VocabularyCleaner(Protocol):
    def clean_pdf_vocabulary_entries(
        self,
        entries: list[WordEntry],
        language: str,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> list[WordEntry]: ...


class PdfVocabularyImporter:
    _LATIN_TOKEN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*")
    _CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
    _KNOWN_COMPOUNDS = {
        "artificial intelligence",
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
    _GENERIC_PHRASE_EXTENSIONS = {
        "algorithm",
        "algorithms",
        "approach",
        "approaches",
        "data",
        "framework",
        "frameworks",
        "method",
        "methods",
        "model",
        "models",
        "paper",
        "papers",
        "result",
        "results",
        "study",
        "studies",
        "system",
        "systems",
        "technique",
        "techniques",
    }
    _GENERIC_PHRASE_STARTERS = {
        "article",
        "chapter",
        "experiment",
        "paper",
        "result",
        "results",
        "section",
        "study",
        "table",
    }
    _GENERIC_PHRASE_HEADS = {
        "approach",
        "approaches",
        "data",
        "example",
        "examples",
        "information",
        "paper",
        "papers",
        "result",
        "results",
        "study",
        "studies",
        "system",
        "systems",
        "thing",
        "things",
        "work",
    }
    _EPONYM_TERMS = {
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
    _EPONYM_HEAD_NOUNS = {
        "algorithm",
        "algorithms",
        "analysis",
        "approximation",
        "distribution",
        "distributions",
        "equation",
        "equations",
        "filter",
        "filters",
        "kernel",
        "kernels",
        "law",
        "laws",
        "matrix",
        "matrices",
        "method",
        "methods",
        "model",
        "models",
        "operator",
        "operators",
        "process",
        "processes",
        "series",
        "space",
        "theorem",
        "theorems",
        "transform",
        "transforms",
    }
    _TECHNICAL_HEAD_NOUNS = _EPONYM_HEAD_NOUNS | {
        "architecture",
        "architectures",
        "boundary",
        "boundaries",
        "coefficient",
        "coefficients",
        "condition",
        "conditions",
        "constraint",
        "constraints",
        "density",
        "derivative",
        "derivatives",
        "energy",
        "feature",
        "features",
        "field",
        "fields",
        "flow",
        "flows",
        "force",
        "forces",
        "function",
        "functions",
        "gradient",
        "gradients",
        "layer",
        "layers",
        "loss",
        "network",
        "networks",
        "parameter",
        "parameters",
        "pressure",
        "probability",
        "representation",
        "representations",
        "signal",
        "signals",
        "state",
        "states",
        "structure",
        "structures",
        "stress",
        "stresses",
        "temperature",
        "temperatures",
        "tension",
        "tensions",
        "tensor",
        "tensors",
        "variable",
        "variables",
        "vector",
        "vectors",
        "wave",
        "waves",
        "weight",
        "weights",
    }
    _TECHNICAL_MODIFIERS = {
        "absolute",
        "active",
        "adaptive",
        "angular",
        "average",
        "binary",
        "boundary",
        "canonical",
        "central",
        "conditional",
        "continuous",
        "convex",
        "critical",
        "deep",
        "differential",
        "discrete",
        "dynamic",
        "elastic",
        "electric",
        "empirical",
        "finite",
        "global",
        "hidden",
        "linear",
        "local",
        "logical",
        "magnetic",
        "maximum",
        "minimum",
        "molecular",
        "neural",
        "normal",
        "optimal",
        "partial",
        "physical",
        "potential",
        "probability",
        "random",
        "relative",
        "residual",
        "spectral",
        "statistical",
        "stochastic",
        "surface",
        "thermal",
        "total",
        "transition",
        "virtual",
    }
    _SPACY_MODELS = {
        "英语": ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"),
        "德语": ("de_core_news_sm", "de_core_news_md", "de_core_news_lg"),
        "法语": ("fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"),
        "西班牙语": ("es_core_news_sm", "es_core_news_md", "es_core_news_lg"),
        "意大利语": ("it_core_news_sm", "it_core_news_md", "it_core_news_lg"),
        "葡萄牙语": ("pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"),
        "荷兰语": ("nl_core_news_sm", "nl_core_news_md", "nl_core_news_lg"),
    }
    _SPACY_PIPELINES: dict[str, Any | None] = {}
    _ENGLISH_LEMMA_OVERRIDES = {
        "am": "be",
        "are": "be",
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
    }
    _LEMMA_OVERRIDES = {
        "英语": _ENGLISH_LEMMA_OVERRIDES,
        "德语": {
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
        },
        "法语": {
            "suis": "être",
            "es": "être",
            "est": "être",
            "sommes": "être",
            "êtes": "être",
            "sont": "être",
            "été": "être",
            "ai": "avoir",
            "as": "avoir",
            "a": "avoir",
            "avons": "avoir",
            "avez": "avoir",
            "ont": "avoir",
            "eu": "avoir",
        },
        "西班牙语": {
            "soy": "ser",
            "eres": "ser",
            "es": "ser",
            "somos": "ser",
            "son": "ser",
            "fue": "ser",
            "fueron": "ser",
            "estoy": "estar",
            "estás": "estar",
            "está": "estar",
            "estamos": "estar",
            "están": "estar",
        },
        "意大利语": {
            "sono": "essere",
            "sei": "essere",
            "è": "essere",
            "siamo": "essere",
            "siete": "essere",
            "era": "essere",
            "erano": "essere",
            "ho": "avere",
            "hai": "avere",
            "ha": "avere",
            "abbiamo": "avere",
            "hanno": "avere",
        },
        "葡萄牙语": {
            "sou": "ser",
            "és": "ser",
            "é": "ser",
            "somos": "ser",
            "são": "ser",
            "foi": "ser",
            "foram": "ser",
            "estou": "estar",
            "está": "estar",
            "estão": "estar",
        },
        "荷兰语": {
            "ben": "zijn",
            "bent": "zijn",
            "is": "zijn",
            "zijn": "zijn",
            "was": "zijn",
            "waren": "zijn",
            "geweest": "zijn",
            "heb": "hebben",
            "hebt": "hebben",
            "heeft": "hebben",
            "had": "hebben",
            "hadden": "hebben",
        },
    }

    _PROFILES = (
        LanguageProfile(
            "英语",
            WordFrequencyCsvImporter.schema_for_language("英语"),
            frozenset("a an and are as at be by for from has have in is it of on or that the to was were with".split()),
            frozenset("the and of to in that is for was with".split()),
        ),
        LanguageProfile(
            "德语",
            WordFrequencyCsvImporter.schema_for_language("德语"),
            frozenset("aber als am an auf aus bei das dem den der des die ein eine einem einen einer es für im in ist mit nicht oder und von zu".split()),
            frozenset("der die das und ist nicht mit ein eine von zu".split()),
        ),
        LanguageProfile(
            "法语",
            WordFrequencyCsvImporter.schema_for_language("法语"),
            frozenset("au aux avec ce ces dans de des du elle en est et la le les ne pas pour que qui un une".split()),
            frozenset("de la le les des et est pour une dans".split()),
        ),
        LanguageProfile(
            "西班牙语",
            WordFrequencyCsvImporter.schema_for_language("西班牙语"),
            frozenset("al con de del el en es la las los no para por que se un una y".split()),
            frozenset("el la los las de que en para una con".split()),
        ),
        LanguageProfile(
            "意大利语",
            WordFrequencyCsvImporter.schema_for_language("意大利语"),
            frozenset("a che con da del della di e gli il in la le lo non per un una".split()),
            frozenset("di che il la e per con una del non".split()),
        ),
        LanguageProfile(
            "葡萄牙语",
            WordFrequencyCsvImporter.schema_for_language("葡萄牙语"),
            frozenset("a as com da das de do dos e em o os para por que um uma".split()),
            frozenset("de que o a em para uma com dos das".split()),
        ),
        LanguageProfile(
            "荷兰语",
            WordFrequencyCsvImporter.schema_for_language("荷兰语"),
            frozenset("aan als de een en er het in is met niet of op te van voor was ze".split()),
            frozenset("de het een en van is in niet voor met".split()),
        ),
    )

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

    def build(self) -> PdfVocabularyImportResult:
        self._report_progress("读取 PDF 文本", 5)
        text = self.extract_text(self._pdf_path)
        self._report_progress("统计词频并生成粗制词表", 20)
        entries, language, schema = self.entries_from_text(text, self._max_entries)
        self._report_progress("写入粗制 CSV", 35)
        self.write_csv(entries, schema, self._csv_path)
        if self._vocabulary_cleaner is not None:
            self._report_progress("AI 审阅粗制 CSV", 40)
            entries = self._vocabulary_cleaner.clean_pdf_vocabulary_entries(
                entries,
                language,
                progress_callback=self._progress_callback,
            )
            entries = self.reindex_entries(entries)
            if not entries:
                raise RuntimeError("LLM 清理后没有保留可导入的词条。")
            self._report_progress("写回 AI 清理后的 CSV", 85)
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
        pages = []
        for page in range(document.pageCount()):
            pages.append(document.getAllText(page).text())
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
        cleaned_text = cls._clean_text(text)
        language, schema = cls.detect_language(cleaned_text)
        terms = cls._cjk_terms(cleaned_text) if language in {"日语", "韩语"} else cls._latin_terms(cleaned_text, language)
        entries = [
            WordEntry(
                source_index=index,
                word=term,
                meaning="",
                frequency=frequency,
                forms=forms,
            )
            for index, (term, frequency, forms) in enumerate(terms[:max_entries], start=1)
        ]
        if not entries:
            raise RuntimeError("PDF 中没有识别到可统计的词或词组。")
        return entries, language, schema

    @classmethod
    def detect_language(cls, text: str) -> tuple[str, CsvColumnSchema]:
        words = [match.group(0).lower() for match in cls._LATIN_TOKEN.finditer(text)]
        scores: dict[str, int] = {
            "日语": len(re.findall(r"[\u3040-\u30ff]+", text)),
            "韩语": len(re.findall(r"[\uac00-\ud7af]+", text)),
        }
        if not words:
            language = max(scores, key=scores.get)
            if scores[language] == 0:
                language = "英语"
            return language, WordFrequencyCsvImporter.schema_for_language(language)

        counts = Counter(words)
        for profile in cls._PROFILES:
            marker_hits = sum(counts[word] for word in profile.markers)
            stopword_hits = sum(counts[word] for word in profile.stopwords)
            candidate_words = sum(
                count
                for word, count in counts.items()
                if word not in profile.stopwords and cls._is_latin_word_candidate(word)
            )
            scores[profile.language] = marker_hits * 4 + stopword_hits + candidate_words
        language = max(scores, key=scores.get)
        if scores[language] == 0:
            language = "英语"
        return language, WordFrequencyCsvImporter.schema_for_language(language)

    @classmethod
    def _clean_text(cls, text: str) -> str:
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lowered = line.lower()
            if lowered.startswith(("http://", "https://", "www.")):
                continue
            if re.fullmatch(r"\[?\d+(?:[-,]\s*\d+)*\]?", line):
                continue
            if cls._looks_like_code(line):
                continue
            if cls._looks_like_formula_or_noise(line):
                continue
            line = re.sub(r"https?://\S+|www\.\S+", " ", line)
            line = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " ", line)
            line = re.sub(r"\[[0-9,\s-]+\]|\([0-9,\s-]+\)", " ", line)
            lines.append(line)
        return "\n".join(lines)

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
        if punctuation / length > 0.45:
            return True
        return False

    @staticmethod
    def _looks_like_code(line: str) -> bool:
        stripped = line.strip()
        lowered = stripped.lower()
        code_prefixes = (
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
        if lowered.startswith(code_prefixes):
            return True
        if re.search(r"\b[a-zA-Z_]\w*\s*=\s*[^=]", stripped):
            return True
        if re.search(r"\b[a-zA-Z_]\w*\s*\([^)]*\)\s*[{:]?$", stripped):
            return True
        code_chars = sum(1 for char in stripped if char in "{}[]();<>")
        operators = sum(1 for char in stripped if char in "=+*/%&|!^~")
        length = max(1, len(stripped))
        if code_chars >= 3 and (code_chars + operators) / length > 0.12:
            return True
        if stripped.endswith(";") and operators > 0:
            return True
        if "->" in stripped or "::" in stripped or "=>" in stripped:
            return True
        if re.search(r"\b(true|false|null|none|undefined)\b", lowered) and operators > 0:
            return True
        return False

    @staticmethod
    def _is_latin_word_candidate(word: str) -> bool:
        if len(word) <= 1 or len(word) > 40:
            return False
        if any(char.isdigit() for char in word):
            return False
        if "_" in word:
            return False
        if word.count("-") > 2 or word.count("'") > 1:
            return False
        letters = [char for char in word if char.isalpha()]
        if not letters:
            return False
        vowels = set("aeiouyäöüàâæçéèêëîïôœùûüáéíóúñãõ")
        if len(word) > 3 and not any(char in vowels for char in word.lower()):
            return False
        return True

    @classmethod
    def _latin_terms(cls, text: str, language: str) -> list[tuple[str, int, str]]:
        words = [match.group(0).lower() for match in cls._LATIN_TOKEN.finditer(text)]
        profile = next((profile for profile in cls._PROFILES if profile.language == language), cls._PROFILES[0])
        override_words = cls._LEMMA_OVERRIDES.get(language, {})
        content_words = [
            word
            for word in words
            if word not in profile.stopwords and cls._is_latin_word_candidate(word)
        ]

        phrase_terms, covered_content_positions = cls._latin_phrase_terms(content_words)
        covered_word_counts = Counter(
            word
            for index, word in enumerate(content_words)
            if index not in covered_content_positions
        )
        for word in words:
            if word in override_words and word in profile.stopwords and cls._is_latin_word_candidate(word):
                covered_word_counts[word] += 1

        word_counts: Counter[str] = covered_word_counts
        terms: dict[str, tuple[int, str, bool]] = {}
        if language in cls._LEMMA_OVERRIDES or language in {"德语", "法语", "西班牙语", "意大利语", "葡萄牙语", "荷兰语"}:
            for lemma, frequency, _forms in cls._lemma_counts(language, word_counts):
                terms[lemma] = (frequency, _forms, False)
        else:
            for word, frequency in word_counts.items():
                terms[word] = (frequency, "", False)
        for phrase, frequency in phrase_terms.items():
            terms[phrase] = (frequency, "", True)

        return sorted(
            ((term, frequency, forms) for term, (frequency, forms, _is_phrase) in terms.items()),
            key=lambda item: (-item[1], " " not in item[0], item[0]),
        )

    @classmethod
    def _latin_phrase_terms(cls, content_words: list[str]) -> tuple[Counter[str], set[int]]:
        bigram_counts: Counter[tuple[str, str]] = Counter(
            tuple(content_words[index : index + 2])
            for index in range(0, max(0, len(content_words) - 1))
        )
        kept_bigrams = {
            bigram
            for bigram, count in bigram_counts.items()
            if cls._is_fixed_bigram(bigram, count)
        }

        terms: Counter[str] = Counter()
        covered_positions: set[int] = set()
        for index in range(0, max(0, len(content_words) - 1)):
            bigram = tuple(content_words[index : index + 2])
            if bigram in kept_bigrams:
                terms[" ".join(bigram)] += 1
                covered_positions.update({index, index + 1})

        trigram_counts: Counter[tuple[str, str, str]] = Counter(
            tuple(content_words[index : index + 3])
            for index in range(0, max(0, len(content_words) - 2))
        )
        for trigram, count in trigram_counts.items():
            if not cls._is_fixed_trigram(trigram, count, kept_bigrams):
                continue
            terms[" ".join(trigram)] += count
            for index in range(0, max(0, len(content_words) - 2)):
                if tuple(content_words[index : index + 3]) == trigram:
                    covered_positions.update({index, index + 1, index + 2})
        return terms, covered_positions

    @classmethod
    def _is_fixed_bigram(cls, bigram: tuple[str, str], count: int) -> bool:
        if len(set(bigram)) == 1:
            return False
        phrase = " ".join(bigram)
        if phrase in cls._KNOWN_COMPOUNDS:
            return True
        if cls._is_eponym_bigram(bigram):
            return True
        if cls._is_technical_bigram(bigram, count):
            return True
        return False

    @classmethod
    def _is_eponym_bigram(cls, bigram: tuple[str, str]) -> bool:
        first, second = bigram
        return first in cls._EPONYM_TERMS and second in cls._EPONYM_HEAD_NOUNS

    @classmethod
    def _is_technical_bigram(cls, bigram: tuple[str, str], count: int) -> bool:
        first, second = bigram
        if first in cls._GENERIC_PHRASE_STARTERS or second in cls._GENERIC_PHRASE_HEADS:
            return False
        if first in cls._TECHNICAL_MODIFIERS and second in cls._TECHNICAL_HEAD_NOUNS:
            return True
        if count >= 2 and second in cls._TECHNICAL_HEAD_NOUNS:
            return True
        return False

    @classmethod
    def _is_fixed_trigram(
        cls,
        trigram: tuple[str, str, str],
        count: int,
        kept_bigrams: set[tuple[str, str]],
    ) -> bool:
        if count < 2 or len(set(trigram)) == 1:
            return False
        if trigram[-1] in cls._GENERIC_PHRASE_EXTENSIONS:
            return False
        if tuple(trigram[:2]) in kept_bigrams or tuple(trigram[1:]) in kept_bigrams:
            return False
        return True

    @classmethod
    def _lemma_counts(cls, language: str, word_counts: Counter[str]) -> list[tuple[str, int, str]]:
        library_counts = cls._library_lemma_counts(language, word_counts)
        if library_counts is None:
            raise RuntimeError(cls._missing_language_library_message(language))
        return library_counts

    @classmethod
    def _library_lemma_counts(cls, language: str, word_counts: Counter[str]) -> list[tuple[str, int, str]] | None:
        if language == "英语":
            nltk_counts = cls._nltk_lemma_counts(word_counts)
            if nltk_counts is not None:
                return nltk_counts

        spacy_counts = cls._spacy_lemma_counts(language, word_counts)
        if spacy_counts is not None:
            return spacy_counts
        return None

    @classmethod
    def _nltk_lemma_counts(cls, word_counts: Counter[str]) -> list[tuple[str, int, str]] | None:
        try:
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import wordnet
        except ImportError:
            return None

        try:
            wordnet.ensure_loaded()
        except LookupError:
            return None

        lemmatizer = WordNetLemmatizer()
        lemma_forms: dict[str, Counter[str]] = {}
        for word, count in word_counts.items():
            lemma = lemmatizer.lemmatize(word, "v")
            lemma = lemmatizer.lemmatize(lemma, "n")
            lemma_forms.setdefault(lemma, Counter())[word] += count
        return [
            (lemma, sum(forms.values()), cls._format_forms(lemma, forms))
            for lemma, forms in lemma_forms.items()
        ]

    @classmethod
    def _spacy_lemma_counts(cls, language: str, word_counts: Counter[str]) -> list[tuple[str, int, str]] | None:
        nlp = cls._spacy_pipeline(language)
        if nlp is None:
            return None
        lemma_forms: dict[str, Counter[str]] = {}
        for word, count in word_counts.items():
            doc = nlp(word)
            if not doc:
                continue
            lemma = doc[0].lemma_.strip().lower() or word
            lemma_forms.setdefault(lemma, Counter())[word] += count
        if not lemma_forms:
            return None
        return [
            (lemma, sum(forms.values()), cls._format_forms(lemma, forms))
            for lemma, forms in lemma_forms.items()
        ]

    @classmethod
    def _spacy_pipeline(cls, language: str):
        if language in cls._SPACY_PIPELINES:
            return cls._SPACY_PIPELINES[language]
        model_names = cls._SPACY_MODELS.get(language, ())
        if not model_names:
            cls._SPACY_PIPELINES[language] = None
            return None
        try:
            import spacy
        except ImportError:
            cls._SPACY_PIPELINES[language] = None
            return None
        for model_name in model_names:
            try:
                cls._SPACY_PIPELINES[language] = spacy.load(model_name, disable=["parser", "ner", "textcat"])
                return cls._SPACY_PIPELINES[language]
            except OSError:
                continue
        cls._SPACY_PIPELINES[language] = None
        return None

    @classmethod
    def _missing_language_library_message(cls, language: str) -> str:
        if language == "英语":
            return (
                "PDF 解析需要英文词形还原库：请安装 nltk 并下载 wordnet 数据。"
                "命令：pip install nltk；python -m nltk.downloader wordnet omw-1.4"
            )
        model_names = cls._SPACY_MODELS.get(language, ())
        if model_names:
            return (
                f"PDF 解析需要 {language} 的 spaCy 语言模型。"
                f"请先安装 spaCy 并安装模型，例如：pip install spacy；python -m spacy download {model_names[0]}"
            )
        return f"PDF 解析暂未配置 {language} 的强制语言库。"

    @classmethod
    def _lemma(cls, language: str, word: str) -> str:
        overrides = cls._LEMMA_OVERRIDES.get(language, {})
        if word in overrides:
            return overrides[word]
        if language == "英语":
            return cls._english_lemma(word)
        if language == "德语":
            return cls._german_lemma(word)
        if language in {"法语", "西班牙语", "意大利语", "葡萄牙语"}:
            return cls._romance_lemma(word)
        if language == "荷兰语":
            return cls._dutch_lemma(word)
        return word

    @classmethod
    def _english_lemma(cls, word: str) -> str:
        if len(word) > 5 and word.endswith("ies"):
            return f"{word[:-3]}y"
        if len(word) > 5 and word.endswith("ves"):
            return f"{word[:-3]}f"
        if len(word) >= 5 and word.endswith("ing"):
            base = word[:-3]
            if len(base) > 2 and base[-1] == base[-2]:
                base = base[:-1]
            if not base.endswith("e"):
                candidate = f"{base}e"
                if candidate in {"use", "make", "take", "write"}:
                    return candidate
            return base
        if len(word) >= 4 and word.endswith("ed"):
            base = word[:-2]
            if len(base) > 2 and base[-1] == base[-2]:
                return base[:-1]
            if not base.endswith("e"):
                return f"{base}e"
            return base
        if len(word) > 4 and word.endswith("es"):
            base = word[:-2]
            if word.endswith(("ches", "shes", "sses", "xes", "zes", "oes")):
                return base
            return word[:-1]
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    @staticmethod
    def _german_lemma(word: str) -> str:
        if len(word) > 6 and word.startswith("ge") and word.endswith("t"):
            return word[2:-1]
        for suffix in ("innen", "ungen", "heiten", "keiten"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                return word[: -len(suffix)]
        for suffix in ("ern", "er", "en", "es", "e", "n", "s"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                return word[: -len(suffix)]
        return word

    @staticmethod
    def _romance_lemma(word: str) -> str:
        for suffix in ("aciones", "uciones", "azione", "azioni", "ements", "ments"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                return word[: -len(suffix)]
        for suffix in ("ando", "iendo", "endo", "ant", "ent"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                return word[: -len(suffix)]
        if len(word) > 5 and word.endswith("ées"):
            return word[:-2]
        if len(word) > 4 and word.endswith(("os", "as", "es")):
            return word[:-1]
        if len(word) > 4 and word.endswith("s"):
            return word[:-1]
        return word

    @staticmethod
    def _dutch_lemma(word: str) -> str:
        for suffix in ("heden", "ingen"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                return word[: -len(suffix)]
        for suffix in ("en", "er", "ers", "s"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                return word[: -len(suffix)]
        return word

    @staticmethod
    def _format_forms(lemma: str, forms: Counter[str]) -> str:
        variants = [
            (word, count)
            for word, count in forms.items()
            if word != lemma
        ]
        variants.sort(key=lambda item: (-item[1], item[0]))
        return ", ".join(f"{word}({count})" for word, count in variants)

    @classmethod
    def _cjk_terms(cls, text: str) -> list[tuple[str, int, str]]:
        counts: Counter[str] = Counter()
        for run in cls._CJK_RUN.findall(text):
            cleaned = re.sub(r"\s+", "", run)
            if len(cleaned) < 2:
                continue
            for size in (2, 3, 4):
                for index in range(0, max(0, len(cleaned) - size + 1)):
                    counts[cleaned[index : index + size]] += 1
        return [
            (term, frequency, "")
            for term, frequency in sorted(counts.items(), key=lambda item: (-item[1], len(item[0]), item[0]))
        ]

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
        return [
            replace(entry, source_index=index)
            for index, entry in enumerate(entries, start=1)
        ]
