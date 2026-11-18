"""Step 4: clean extracted text into paragraphs.

- NFC normalisation (ZWJ/ZWNJ and all vowel signs are preserved)
- removes repeated headers/footers and page-number lines
- rebuilds paragraphs from PDF line breaks; short title-like lines become headings
- drops short, wrong-script, wrong-language (Hindi vs Marathi) or corrupted paragraphs
- removes exact and near-duplicate paragraphs across all documents of a language

Usage:
    python phase_06_native_indic_qa/scripts/clean_text.py --languages hi
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter

from rapidfuzz import fuzz, process

from common import (CLEANED_DIR, EXTRACTED_DIR, LANGUAGES, RESULTS_DIR, detect_devanagari_language,
                    language_ok, load_config, nfc, norm_for_match, read_json, selected_languages,
                    word_count, write_json)

TERMINATORS = ("।", "॥", ".", "?", "!", ":", "؟")
INDIC_DIGITS = "०१२३४५६७८९০১২৩৪৫৬৭৮৯௦௧௨௩௪௫௬௭௮௯"
DIGIT_CLASS = f"0-9{INDIC_DIGITS}"
PAGE_NUMBER_RE = re.compile(
    rf"^\s*(?:[-–—(\[]\s*)?(?:page|pg\.?|पृष्ठ|पान|पृ\.|পৃষ্ঠা|பக்கம்)?\s*[{DIGIT_CLASS}]{{1,4}}"
    rf"(?:\s*(?:/|of|का|का\s+)\s*[{DIGIT_CLASS}]{{1,4}})?\s*(?:[-–—)\]]\s*)?$",
    re.IGNORECASE,
)
BULLET_RE = re.compile(rf"^\s*(?:[•●▪■◆►✓\-*–]|\(?[{DIGIT_CLASS}a-zA-Zअ-ह]{{1,3}}[.)]\s)")
URL_RE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.IGNORECASE)
# Article credit lines ("स्त्रोत: कृषि विभाग, बिहार सरकार", "ஆதாரம்: ...") are provenance,
# not content; they are recorded in the source CSVs and removed from passages.
CREDIT_RE = re.compile(
    r"^\s*(?:स्त्रो|स्‍त्रो|स्त्राे|स्रोत|माहिती\s*(?:संदर्भ|स्रोत|दाता)|लेखक\s*:|ஆதாரம்|ஆதாரங்கள்|"
    r"সূত্র|সুত্র|সু্ত্র|তথ্য\s*সংকলন|তথ্যসূত্র|source\s*:)", re.IGNORECASE)
MIN_HEADING_SCRIPT_RATIO = 0.3  # headings often carry an English gloss: "மின்கடத்து திறன் (Electrical Conductivity)"


def line_key(line: str) -> str:
    """Header/footer key: digits removed so 'Page 3' and 'Page 4' match."""
    return re.sub(rf"[{DIGIT_CLASS}]+", "#", norm_for_match(line))


def find_repeated_lines(pages: list[dict], cfg: dict) -> set[str]:
    if len(pages) < 3:
        return set()
    counts = Counter()
    for page in pages:
        keys = {line_key(l) for l in page["text"].splitlines()
                if l.strip() and word_count(l) <= cfg["repeated_line_max_words"]}
        counts.update(keys)
    limit = max(2, cfg["repeated_line_page_fraction"] * len(pages))
    return {key for key, n in counts.items() if n >= limit and key}


def page_paragraphs(text: str, page_number, repeated: set[str], line_per_paragraph: bool = False) -> list[dict]:
    lines = [re.sub(r"[\t  ]+", " ", l).strip() for l in text.splitlines()]
    kept = []
    for line in lines:
        if not line:
            kept.append("")
        elif PAGE_NUMBER_RE.match(line) or URL_RE.match(line) or line_key(line) in repeated:
            continue
        else:
            kept.append(line)
    lengths = [len(l) for l in kept if l]
    typical = statistics.median(lengths) if lengths else 0

    paragraphs, current = [], []

    def flush():
        if current:
            paragraphs.append({"text": " ".join(current).strip(), "page_number": page_number,
                               "is_heading": False})
            current.clear()

    for i, line in enumerate(kept):
        if not line:
            flush()
            continue
        ends_sentence = line.endswith(TERMINATORS)
        heading_like = word_count(line) <= 8 and not ends_sentence and len(line) < 0.6 * max(typical, 1)
        has_following_text = any(kept[i + 1:])
        if heading_like and not current and has_following_text:
            paragraphs.append({"text": line.rstrip(":"), "page_number": page_number, "is_heading": True})
            continue
        if BULLET_RE.match(line):
            flush()
        current.append(line)
        # HTML extractors emit one paragraph per line; PDFs wrap paragraphs over many
        # lines, where a sentence-final line noticeably shorter than a full line ends one.
        if line_per_paragraph or (ends_sentence and len(line) < 0.75 * typical):
            flush()
    flush()
    return paragraphs


def is_corrupted(text: str) -> bool:
    if text.count("�") > max(1, len(text) // 100):
        return True
    # Legacy-font residue: long runs of Latin punctuation/symbol soup.
    return bool(re.search(r"[\\@#$%^&*{}~`|<>]{4,}", text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", default="all")
    args = parser.parse_args()
    cfg = load_config()["cleaning"]

    report = read_json(RESULTS_DIR / "cleaning_report.json", {}) or {}
    for lang in selected_languages(args.languages):
        folder = LANGUAGES[lang]["folder"]
        docs = sorted((EXTRACTED_DIR / folder).glob("*.json"))
        seen_exact: dict[str, str] = {}
        kept_keys: list[str] = []
        kept_owner: list[str] = []
        stats = Counter()
        doc_reports = []

        for doc_path in docs:
            doc = read_json(doc_path)
            pages = [p for p in doc["pages"] if not p.get("needs_ocr") and p["text"].strip()]
            repeated = find_repeated_lines(pages, cfg)
            out, votes = [], Counter()
            for page in pages:
                html = page.get("method") in ("trafilatura", "beautifulsoup", "embedded_article_html")
                for para in page_paragraphs(nfc(page["text"]), page["page_number"], repeated, html):
                    text = para["text"]
                    stats["paragraphs_seen"] += 1
                    if CREDIT_RE.match(text):
                        stats["dropped_credit_line"] += 1
                        continue
                    threshold = MIN_HEADING_SCRIPT_RATIO if para["is_heading"] else cfg["min_paragraph_script_ratio"]
                    ok, reason = language_ok(text, lang, threshold)
                    if lang in ("hi", "mr"):
                        votes[detect_devanagari_language(text)] += 1
                    if not ok:
                        stats[f"dropped_{reason.split('_0')[0]}"] += 1
                        continue
                    if para["is_heading"]:
                        out.append(para)
                        continue
                    if word_count(text) < cfg["min_paragraph_words"]:
                        # A short but complete sentence (common in list items, and in
                        # Tamil, where words are long) joins the preceding paragraph.
                        if text.rstrip().endswith(TERMINATORS) and out and not out[-1]["is_heading"] \
                                and word_count(text) >= 3:
                            out[-1] = {**out[-1], "text": out[-1]["text"] + " " + text}
                            stats["merged_short_sentence"] += 1
                        else:
                            stats["dropped_too_short"] += 1
                        continue
                    if is_corrupted(text):
                        stats["dropped_corrupted"] += 1
                        continue
                    key = norm_for_match(text)
                    if key in seen_exact:
                        stats["dropped_exact_duplicate"] += 1
                        continue
                    if kept_keys:
                        match = process.extractOne(key, kept_keys, scorer=fuzz.ratio,
                                                   score_cutoff=cfg["near_duplicate_threshold"])
                        if match:
                            stats["dropped_near_duplicate"] += 1
                            if kept_owner[match[2]] != doc["document_id"]:
                                stats["cross_document_duplicates"] += 1
                            continue
                    seen_exact[key] = doc["document_id"]
                    kept_keys.append(key)
                    kept_owner.append(doc["document_id"])
                    out.append(para)
                    stats["paragraphs_kept"] += 1

            # Drop headings that are not followed by any body paragraph.
            out = [p for i, p in enumerate(out)
                   if not p["is_heading"] or (i + 1 < len(out) and not out[i + 1]["is_heading"])]
            write_json(CLEANED_DIR / folder / doc_path.name,
                       {"document_id": doc["document_id"], "language": lang, "paragraphs": out})
            other = "mr" if lang == "hi" else "hi"
            mislabeled = lang in ("hi", "mr") and votes[other] > max(votes[lang], 3)
            doc_reports.append({"document_id": doc["document_id"],
                                "paragraphs": sum(not p["is_heading"] for p in out),
                                "words": sum(word_count(p["text"]) for p in out if not p["is_heading"]),
                                "possible_language_mislabel": mislabeled})
            if mislabeled:
                print(f"! {doc['document_id']}: most paragraphs look like {other}, not {lang} - check the manifest")

        report[lang] = {"documents": len(docs), **dict(stats), "per_document": doc_reports}
        print(f"[{lang}] {len(docs)} docs | kept {stats['paragraphs_kept']} of {stats['paragraphs_seen']} "
              f"paragraphs | near-dups {stats['dropped_near_duplicate']} "
              f"(cross-document {stats['cross_document_duplicates']})")
    write_json(RESULTS_DIR / "cleaning_report.json", report)


if __name__ == "__main__":
    main()
