"""Shared paths, language settings and helpers for the Phase 6 pipeline.

Every script imports this module, so path layout and language rules live in one place.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import regex as uregex
import yaml

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parent
# PHASE6_SANDBOX=<dir> redirects all data, sources, review and results paths into
# <dir> (configs and prompts still come from the repo). Used for smoke tests.
_SANDBOX = os.environ.get("PHASE6_SANDBOX")
_WORK = Path(_SANDBOX).resolve() if _SANDBOX else PHASE_DIR
DATA_DIR = Path(_SANDBOX).resolve() / "data" if _SANDBOX else REPO_ROOT / "data" / "native_indic_qa"

RAW_DOCS_DIR = DATA_DIR / "raw_documents"
CONTEXTS_DIR = DATA_DIR / "processed_contexts"
EXTRACTED_DIR = CONTEXTS_DIR / "extracted"
CLEANED_DIR = CONTEXTS_DIR / "cleaned"
GENERATED_DIR = DATA_DIR / "generated_qa"
VALIDATED_DIR = DATA_DIR / "validated_qa"
# final/ holds the publishable release: questions, answers and offsets, but no passage
# text (Vikaspedia redistribution needs permission). full/ holds the same records with
# contexts; it is git-ignored and rebuilt by build_dataset.py or hydrate_dataset.py.
FINAL_DIR = DATA_DIR / "final"
FULL_DIR = DATA_DIR / "full"

SOURCES_DIR = _WORK / "sources"
REVIEW_DIR = _WORK / "human_review"            # public sheets (no context text)
REVIEW_PRIVATE_DIR = REVIEW_DIR / "private"     # reviewer sheets with context (git-ignored)
RESULTS_DIR = _WORK / "results"
CONFIG_PATH = PHASE_DIR / "configs" / "generation_config.yaml"
PROMPT_PATH = PHASE_DIR / "prompts" / "qa_generation_prompt.txt"

MANIFEST_FIELDS = [
    "document_id", "document_title", "language", "organization", "source_url",
    "publication_year", "region", "domain", "file_path", "license",
    "document_hash", "language_origin", "download_date",
]

DOMAINS = [
    "irrigation", "crop_production", "soil_management", "fertilizers_nutrition",
    "plant_diseases", "pest_management", "harvesting_storage", "agricultural_technology",
]
QUESTION_TYPES = [
    "factual", "numerical", "temporal", "procedural", "causal",
    "conditional", "identification", "benefit",
]
DIFFICULTIES = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------
# Script ranges are Unicode blocks. Hindi and Marathi share Devanagari, so they
# are told apart with high-frequency function words (see detect_devanagari_language).
LANGUAGES: dict[str, dict] = {
    "hi": {"name": "Hindi", "native": "हिंदी", "folder": "hindi", "prefix": "NATIVE_HI",
           "tesseract": "hin", "script": "devanagari"},
    "mr": {"name": "Marathi", "native": "मराठी", "folder": "marathi", "prefix": "NATIVE_MR",
           "tesseract": "mar", "script": "devanagari"},
    "bn": {"name": "Bengali", "native": "বাংলা", "folder": "bengali", "prefix": "NATIVE_BN",
           "tesseract": "ben", "script": "bengali"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "folder": "tamil", "prefix": "NATIVE_TA",
           "tesseract": "tam", "script": "tamil"},
}
FOLDER_TO_LANG = {cfg["folder"]: code for code, cfg in LANGUAGES.items()}

SCRIPT_RANGES = {
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "bengali": [(0x0980, 0x09FF)],
    "tamil": [(0x0B80, 0x0BFF)],
}

# Function words that are frequent in one language and rare in the other. Words
# shared by both (या, ही, होती, होता, जाती, जाते, कर) are deliberately left out:
# real Vikaspedia text showed them causing false rejections.
HINDI_MARKERS = {"है", "हैं", "और", "के", "की", "में", "को", "से", "लिए", "करें", "चाहिए", "यह",
                 "इस", "किया", "पर", "तक", "होता", "करना", "जाता"}
MARATHI_MARKERS = {"आहे", "आहेत", "आणि", "करावी", "करावे", "करावा", "मध्ये", "साठी", "नाही",
                   "असते", "असतो", "जातो", "हे", "व", "पाहिजे", "द्यावे", "द्यावी", "पिकाला",
                   "पिकाची", "किंवा", "त्यामुळे"}
# Short markers that double as abbreviations: हे. (hectare), को. (Co. variety), से. (cm), व.
ABBREVIATION_PRONE = {"हे", "को", "से", "व"}


def in_ranges(ch: str, ranges: list[tuple[int, int]]) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ranges)


def script_ratio(text: str, lang: str) -> float:
    """Share of letters (incl. combining vowel signs) that belong to the language's script.

    Digits, punctuation and whitespace are ignored. Latin technical terms and units
    lower the ratio, which is why thresholds are well below 1.0.
    """
    ranges = SCRIPT_RANGES[LANGUAGES[lang]["script"]]
    letters = [c for c in text if unicodedata.category(c)[0] in ("L", "M")]
    if not letters:
        return 0.0
    return sum(in_ranges(c, ranges) for c in letters) / len(letters)


def garbled_ratio(text: str, lang: str) -> float:
    """Share of letters that are neither the target script nor plain ASCII.

    Real Indic extension text mixes in English words (ASCII) but almost never
    Latin Extended / IPA / Greek letters. PDFs with broken glyph-to-Unicode maps
    (very common for Indic fonts) extract as Devanagari interleaved with exactly
    such letters, e.g. 'ठɘबक ɧ͆सचाई', so this ratio flags them for OCR.
    """
    ranges = SCRIPT_RANGES[LANGUAGES[lang]["script"]]
    letters = [c for c in text if unicodedata.category(c)[0] in ("L", "M")]
    if not letters:
        return 0.0
    odd = sum(not in_ranges(c, ranges) and ord(c) >= 0x80 for c in letters)
    # A second failure mode stays inside the script but maps glyphs to the wrong
    # code points (seen in an MPKV PDF: 'साधनसंपत्ती' -> 'िाधनिंपत्ती'). Valid
    # text never starts a word with a dependent vowel sign or virama.
    words = text.split()
    bad_starts = sum(1 for w in words if unicodedata.category(w[0]) in ("Mn", "Mc")
                     and w[0] not in "‌‍")
    # Third failure mode (ICAR InDesign PDFs): vowel signs extracted twice
    # ('कीी', 'दोोनोों') while half-letters are dropped ('उत्ादन'). A doubled
    # dependent vowel sign never occurs in valid text.
    doubled = len(DOUBLED_SIGN_RE.findall(text))
    n = len(words) or 1
    return max(odd / len(letters), bad_starts / n, doubled / n)


# Dependent vowel signs of Devanagari, Bengali and Tamil, repeated back to back.
DOUBLED_SIGN_RE = uregex.compile(r"([ा-ौা-ৌா-ௌ])\1")


MAX_GARBLED_RATIO = 0.01


def tokens(text: str) -> list[str]:
    return re.findall(r"[^\s।॥.,;:!?\"'()\[\]{}<>/\\|-]+", text)


def detect_devanagari_language(text: str) -> str | None:
    """Return 'hi', 'mr' or None (undecided) using function-word votes.

    A language wins only with at least 2 votes and more than twice the other's,
    so a single shared or abbreviated word cannot flip a paragraph."""
    words = []
    for raw in text.split():
        word = raw.strip("\"'“”‘’()[]{},;:!?।॥")
        if word.endswith(".") and word.rstrip(".") in ABBREVIATION_PRONE:
            continue  # हे. / को. / से. are abbreviations, not function words
        words.append(word.rstrip("."))
    hi = sum(w in HINDI_MARKERS for w in words)
    mr = sum(w in MARATHI_MARKERS for w in words)
    if hi >= 2 and hi > 2 * mr:
        return "hi"
    if mr >= 2 and mr > 2 * hi:
        return "mr"
    return None


def language_ok(text: str, lang: str, min_ratio: float) -> tuple[bool, str]:
    """Check that text is written in the expected language. Returns (ok, reason)."""
    ratio = script_ratio(text, lang)
    if ratio < min_ratio:
        return False, f"script_ratio_{ratio:.2f}"
    if garbled_ratio(text, lang) > MAX_GARBLED_RATIO:
        return False, "garbled_text"
    if lang in ("hi", "mr"):
        guess = detect_devanagari_language(text)
        if guess is not None and guess != lang:
            return False, f"looks_like_{guess}"
    return True, ""


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def nfc(text: str) -> str:
    """NFC-normalise. ZWJ/ZWNJ are kept on purpose: Bengali and Marathi use them
    to select conjunct and eyelash-ra forms, so removing them changes spelling."""
    return unicodedata.normalize("NFC", text or "")


def norm_for_match(text: str) -> str:
    """Aggressive key for duplicate detection only (never for stored text)."""
    text = nfc(text).lower()
    text = re.sub(r"[‌‍]", "", text)
    # Python's re \w does not match Indic vowel signs (category M), so use the
    # `regex` module's Unicode properties to keep letters, marks and digits.
    text = uregex.sub(r"[^\p{L}\p{M}\p{N}\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_url(url: str) -> str:
    """Canonical form for cross-collection URL matching.

    The same Vikaspedia article is reachable under several URLs: language
    subdomains (ta./mr./hi./bn.) or agriculture.vikaspedia.in, /viewcontent/ or
    not, '?lgn=xx', '//' in the path, and percent-encoded vs raw Unicode slugs.
    All of these collapse to one key here.
    """
    from urllib.parse import unquote, urlparse
    if not url:
        return ""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    if host.endswith("vikaspedia.in"):
        host = "vikaspedia.in"
    path = nfc(unquote(parsed.path))
    path = re.sub(r"/{2,}", "/", path).removeprefix("/viewcontent").rstrip("/")
    if host == "vikaspedia.in":
        path = "/".join("-".join(_decode_hex_word(w) for w in seg.split("-")) for seg in path.split("/"))
    return f"{host}{path}".lower()


def _decode_hex_word(word: str) -> str:
    """Vikaspedia's legacy slugs spell each code point as 3 hex digits
    ('907924930' = U+0907 U+0924 U+0930 = 'इतर'). Decode only when every
    chunk is an assigned character in one Indic block (U+0900-U+0DFF)."""
    if len(word) < 3 or len(word) % 3 or not re.fullmatch(r"[0-9a-f]+", word, re.IGNORECASE):
        return word
    chars = [chr(int(word[i:i + 3], 16)) for i in range(0, len(word), 3)]
    if not all(0x900 <= ord(c) <= 0xDFF and unicodedata.name(c, "") for c in chars):
        return word
    if len({ord(c) >> 7 for c in chars}) != 1:  # all from the same 128-code-point script block
        return word
    return nfc("".join(chars))


def word_count(text: str) -> int:
    return len(text.split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # utf-8-sig: reviewers often save from Excel, which prepends a BOM.
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [row for row in csv.DictReader(stream) if any((v or "").strip() for v in row.values())]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel shows Indic text correctly when reviewers open the sheet.
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def selected_languages(arg: str | None) -> list[str]:
    if not arg or arg == "all":
        return list(LANGUAGES)
    langs = [part.strip() for part in arg.split(",") if part.strip()]
    unknown = [lang for lang in langs if lang not in LANGUAGES]
    if unknown:
        raise SystemExit(f"Unknown language code(s): {unknown}. Use hi, mr, bn, ta or all.")
    return langs


def load_manifest() -> list[dict]:
    return read_csv(SOURCES_DIR / "source_manifest.csv")


def rel(path: Path) -> str:
    """Repo-relative POSIX path, so manifests are portable across machines."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()
