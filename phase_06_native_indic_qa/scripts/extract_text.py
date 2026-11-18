"""Step 3: extract Unicode text from every document in sources/source_manifest.csv.

- Digital PDFs: PyMuPDF, page by page (page numbers kept).
- HTML: trafilatura main-content extraction, BeautifulSoup fallback.
- Scanned PDFs / images: Tesseract OCR with --ocr (needs the Tesseract binary and
  the hin/mar/ben/tam traineddata packs). OCR text must be spot-checked by hand.

Many Indian government PDFs are typeset in legacy (non-Unicode) fonts such as
Kruti Dev or Shivaji, or embed Indic fonts with broken glyph-to-Unicode maps.
Their extracted "text" is Latin gibberish or Devanagari mixed with stray
Latin-Extended/IPA letters. Such pages are detected (low script ratio or high
garbled ratio) and marked needs_ocr; with --ocr they are OCR'd.

Usage:
    python phase_06_native_indic_qa/scripts/extract_text.py
    python phase_06_native_indic_qa/scripts/extract_text.py --languages ta --ocr
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import (EXTRACTED_DIR, LANGUAGES, MAX_GARBLED_RATIO, REPO_ROOT, RESULTS_DIR, garbled_ratio,
                    load_config, load_manifest, nfc, read_json, rel, script_ratio, selected_languages,
                    write_json)


def ocr_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_image(image, lang: str) -> str:
    import pytesseract
    return pytesseract.image_to_string(image, lang=LANGUAGES[lang]["tesseract"])


def extract_pdf(path: Path, lang: str, cfg: dict, use_ocr: bool) -> list[dict]:
    import pymupdf
    pages = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc):
            text = nfc(page.get_text("text", sort=True))
            ratio = script_ratio(text, lang)
            garbled = garbled_ratio(text, lang)
            method = "pymupdf"
            needs_ocr = (len(text.strip()) < 40 or ratio < cfg["min_page_script_ratio"]
                         or garbled > MAX_GARBLED_RATIO)
            if needs_ocr and use_ocr:
                from PIL import Image
                pix = page.get_pixmap(dpi=cfg["ocr_dpi"])
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = nfc(ocr_image(image, lang))
                ratio = script_ratio(text, lang)
                method, needs_ocr = "tesseract", False
            pages.append({"page_number": index + 1, "text": text, "method": method,
                          "script_ratio": round(ratio, 3), "garbled_ratio": round(garbled_ratio(text, lang), 3),
                          "needs_ocr": needs_ocr})
    return pages


BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "div", "blockquote", "dt", "dd", "section"]


def html_fragment_to_text(fragment: str) -> str:
    """Article HTML -> text with one block (heading, paragraph, list item, table row)
    per line, so clean_text.py can rebuild paragraphs and section headings."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(fragment, "html.parser")
    for tag in soup(["script", "style", "img", "figure", "iframe"]):
        tag.decompose()
    # Line breaks in the HTML source are not structure (Vikaspedia puts them between
    # table cells and inside paragraphs); collapse them, then rebuild lines from tags.
    for node in soup.find_all(string=True):
        node.replace_with(re.sub(r"\s+", " ", str(node)))
    for row in soup.find_all("tr"):
        cells = [" ".join(c.get_text(" ").split()) for c in row.find_all(["td", "th"])]
        row.replace_with("\n" + " | ".join(c for c in cells if c) + "\n")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(BLOCK_TAGS):
        block.insert_before("\n")
        block.append("\n")
    lines = [" ".join(line.split()).strip(" |") for line in soup.get_text().splitlines()]
    return "\n".join(line for line in lines if line)


def embedded_article_html(html: str) -> str | None:
    """Some portals (e.g. Vikaspedia, a Next.js site) ship an empty body and put the
    article HTML inside JSON: the __NEXT_DATA__ blob ('content') and JSON-LD
    ('articleBody'). trafilatura then reads a flattened copy that it truncates at
    60,000 characters, so prefer the longest embedded HTML string."""
    import json
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    found: list[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    if key in ("content", "articleBody") and "<" in value and len(value) > 200:
                        found.append(value)
                else:
                    walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    scripts = soup.find_all("script", id="__NEXT_DATA__") + soup.find_all("script", type="application/ld+json")
    for script in scripts:
        try:
            walk(json.loads(script.string or ""))
        except ValueError:
            continue
    return max(found, key=len) if found else None


def extract_html(path: Path, lang: str) -> list[dict]:
    raw = path.read_bytes()
    html = None
    for encoding in ("utf-8", "utf-16"):
        try:
            html = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    html = html if html is not None else raw.decode("utf-8", errors="replace")

    embedded = embedded_article_html(html)
    if embedded:
        text = nfc(html_fragment_to_text(embedded))
        if len(text.strip()) >= 100:
            return [{"page_number": None, "text": text, "method": "embedded_article_html",
                     "script_ratio": round(script_ratio(text, lang), 3),
                     "garbled_ratio": round(garbled_ratio(text, lang), 3), "needs_ocr": False}]

    text, method = "", "trafilatura"
    try:
        import trafilatura
        text = trafilatura.extract(html, include_tables=True, include_comments=False,
                                   favor_recall=True) or ""
    except Exception:
        text = ""
    if len(text.strip()) < 100:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
            tag.decompose()
        node = soup.find("article") or soup.find("main") or soup.body or soup
        text, method = node.get_text("\n"), "beautifulsoup"
    text = nfc(text)
    return [{"page_number": None, "text": text, "method": method,
             "script_ratio": round(script_ratio(text, lang), 3), "needs_ocr": False}]


def extract_image(path: Path, lang: str, use_ocr: bool) -> list[dict]:
    if not use_ocr:
        return [{"page_number": 1, "text": "", "method": "none", "script_ratio": 0.0, "needs_ocr": True}]
    from PIL import Image
    text = nfc(ocr_image(Image.open(path), lang))
    return [{"page_number": 1, "text": text, "method": "tesseract",
             "script_ratio": round(script_ratio(text, lang), 3), "needs_ocr": False}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", default="all")
    parser.add_argument("--ocr", action="store_true", help="OCR pages with no usable Unicode text")
    parser.add_argument("--force", action="store_true", help="Re-extract documents already extracted")
    args = parser.parse_args()
    cfg = load_config()["extraction"]
    langs = selected_languages(args.languages)

    if args.ocr and not ocr_available():
        raise SystemExit("--ocr needs pytesseract and the Tesseract binary with hin/mar/ben/tam packs "
                         "(uv add pytesseract; install Tesseract from UB Mannheim on Windows).")

    report = []
    for row in load_manifest():
        lang = row["language"]
        if lang not in langs:
            continue
        doc_id = row["document_id"]
        out_path = EXTRACTED_DIR / LANGUAGES[lang]["folder"] / f"{doc_id}.json"
        if out_path.exists() and not args.force:
            continue
        path = REPO_ROOT / row["file_path"]
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                pages = extract_pdf(path, lang, cfg, args.ocr)
            elif suffix in (".html", ".htm"):
                pages = extract_html(path, lang)
            elif suffix == ".txt":
                text = nfc(path.read_text(encoding="utf-8", errors="replace"))
                pages = [{"page_number": None, "text": text, "method": "plain_text",
                          "script_ratio": round(script_ratio(text, lang), 3), "needs_ocr": False}]
            elif suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                pages = extract_image(path, lang, args.ocr)
            else:
                raise ValueError(f"unsupported file type {suffix}")
        except Exception as error:
            report.append({"document_id": doc_id, "language": lang, "status": "error",
                           "error": f"{type(error).__name__}: {error}"})
            print(f"x {doc_id}: {type(error).__name__}: {error}")
            continue

        write_json(out_path, {"document_id": doc_id, "language": lang, "file_path": row["file_path"],
                              "pages": pages})
        needs_ocr = sum(p["needs_ocr"] for p in pages)
        chars = sum(len(p["text"]) for p in pages)
        status = "needs_ocr" if needs_ocr == len(pages) else "partial_ocr_needed" if needs_ocr else "ok"
        report.append({"document_id": doc_id, "language": lang, "status": status, "pages": len(pages),
                       "pages_needing_ocr": needs_ocr, "characters": chars,
                       "methods": sorted({p["method"] for p in pages})})
        print(f"{'ok' if status == 'ok' else '!'} {doc_id}: {len(pages)} page(s), {chars} chars, "
              f"{needs_ocr} page(s) need OCR -> {rel(out_path)}")

    if report:
        # Merge with earlier runs so the report always covers every extracted document.
        report_path = RESULTS_DIR / "extraction_report.json"
        merged = {r["document_id"]: r for r in (read_json(report_path, []) or [])}
        merged.update({r["document_id"]: r for r in report})
        write_json(report_path, sorted(merged.values(), key=lambda r: r["document_id"]))
    flagged = [r["document_id"] for r in report if r["status"] != "ok"]
    print(f"\nExtracted {len(report) - len(flagged)} cleanly; {len(flagged)} flagged: {', '.join(flagged[:15])}")
    if flagged and not args.ocr:
        print("Flagged pages are likely scanned or legacy-font PDFs. Rerun with --ocr, or drop them "
              "(digital Unicode sources are preferred given the deadline).")


if __name__ == "__main__":
    main()
