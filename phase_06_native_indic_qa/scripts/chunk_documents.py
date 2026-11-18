"""Step 6: pack cleaned paragraphs into 100-200 word context passages.

Chunks never cross a section heading, so a passage stays on one topic. Each
context keeps document id, section title, page range, a keyword-based domain
hint and (if `transformers` is installed) its XLM-R token count, because Indic
text tokenises into far more subwords than English.

Usage:
    python phase_06_native_indic_qa/scripts/chunk_documents.py --languages hi
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

from common import (CLEANED_DIR, CONTEXTS_DIR, LANGUAGES, RESULTS_DIR, language_ok, load_config,
                    load_manifest, read_json, selected_languages, word_count, write_json, write_jsonl)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[।॥?!.])\s+")

# Substring cues per domain. Used only as a hint for balancing topics; the final
# domain comes from the LLM output and human review.
DOMAIN_KEYWORDS = {
    "hi": {
        "irrigation": ["सिंचाई", "ठिबक", "टपक", "फव्वारा", "बौछारी", "स्प्रिंकलर", "ड्रिप", "जल निकास", "पानी"],
        "soil_management": ["मृदा", "मिट्टी", "भूमि की तैयारी", "जुताई", "लवणीय", "क्षारीय"],
        "fertilizers_nutrition": ["उर्वरक", "खाद", "नाइट्रोजन", "फास्फोरस", "फॉस्फोरस", "पोटाश", "यूरिया", "पोषक"],
        "plant_diseases": ["रोग", "फफूंद", "झुलसा", "विषाणु", "उकठा", "गलन", "धब्बा"],
        "pest_management": ["कीट", "इल्ली", "माहू", "सुंडी", "कीटनाशक", "दीमक", "छेदक"],
        "harvesting_storage": ["कटाई", "भंडारण", "मड़ाई", "सुखाना", "उपज"],
        "agricultural_technology": ["यंत्र", "मशीन", "तकनीक", "ड्रोन", "सेंसर", "मोबाइल"],
        "crop_production": ["बुवाई", "बीज", "किस्म", "रोपाई", "पौध", "खेती"],
    },
    "mr": {
        "irrigation": ["सिंचन", "ठिबक", "तुषार", "पाणी", "ओलित", "पाण्याचा", "पाण्याची"],
        "soil_management": ["माती", "मृदा", "जमीन", "जमिनी", "नांगरट", "क्षारपड", "चोपण"],
        "fertilizers_nutrition": ["खत", "नत्र", "स्फुरद", "पालाश", "युरिया", "अन्नद्रव्य", "सेंद्रिय"],
        "plant_diseases": ["रोग", "बुरशी", "करपा", "मर ", "विषाणू", "कुज"],
        "pest_management": ["कीड", "किडी", "अळी", "मावा", "कीटकनाशक", "तुडतुडे", "फुलकिडे"],
        "harvesting_storage": ["काढणी", "साठवण", "मळणी", "वाळवण", "उत्पादन"],
        "agricultural_technology": ["यंत्र", "तंत्रज्ञान", "ड्रोन", "सेन्सर", "मोबाईल"],
        "crop_production": ["पेरणी", "बियाणे", "वाण", "लागवड", "रोपे", "बीजप्रक्रिया"],
    },
    "bn": {
        "irrigation": ["সেচ", "ড্রিপ", "ফোঁটা", "স্প্রিংকলার", "জলনিকাশ", "জল", "পানি"],
        "soil_management": ["মাটি", "মৃত্তিকা", "চাষ দেওয়া", "লবণাক্ত", "অম্ল"],
        "fertilizers_nutrition": ["সার", "নাইট্রোজেন", "ফসফরাস", "পটাশ", "ইউরিয়া", "পুষ্টি"],
        "plant_diseases": ["রোগ", "ছত্রাক", "ধসা", "ঝলসা", "ভাইরাস", "পচা"],
        "pest_management": ["পোকা", "কীট", "মাজরা", "শোষক", "কীটনাশক"],
        "harvesting_storage": ["ফসল কাটা", "সংরক্ষণ", "মাড়াই", "শুকানো", "গুদাম"],
        "agricultural_technology": ["যন্ত্র", "প্রযুক্তি", "ড্রোন", "মেশিন"],
        "crop_production": ["বীজ", "জাত", "বপন", "রোপণ", "চারা", "চাষ"],
    },
    "ta": {
        "irrigation": ["பாசன", "நீர்", "சொட்டு", "தெளிப்பு", "தண்ணீர்"],
        "soil_management": ["மண்", "உழவு", "களர்", "உவர்"],
        "fertilizers_nutrition": ["உரம்", "உரங்", "தழைச்சத்து", "மணிச்சத்து", "சாம்பல்சத்து", "யூரியா", "ஊட்டச்சத்து"],
        "plant_diseases": ["நோய்", "பூஞ்சை", "கருகல்", "வாடல்", "அழுகல்", "தேமல்"],
        "pest_management": ["பூச்சி", "புழு", "அசுவினி", "தத்துப்பூச்சி", "பூச்சிக்கொல்லி"],
        "harvesting_storage": ["அறுவடை", "சேமிப்", "உலர்த்", "கதிர்"],
        "agricultural_technology": ["இயந்திர", "தொழில்நுட்ப", "ட்ரோன்", "கருவி"],
        "crop_production": ["விதை", "ரகம்", "நடவு", "சாகுபடி", "நாற்று", "விதைப்பு"],
    },
}


def domain_hint(text: str, lang: str, fallback: str) -> str:
    scores = Counter({domain: sum(text.count(k) for k in keys)
                      for domain, keys in DOMAIN_KEYWORDS[lang].items()})
    best = scores.most_common(2)
    if not best or best[0][1] == 0 or (len(best) > 1 and best[0][1] == best[1][1]):
        return fallback or "unknown"
    return best[0][0]


def split_long(text: str, max_words: int) -> list[str]:
    """Split an over-long paragraph at sentence ends into pieces of <= max_words."""
    pieces, current = [], []
    for sentence in SENTENCE_SPLIT_RE.split(text):
        if current and word_count(" ".join(current + [sentence])) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def load_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)
    except Exception:
        return None


def chunk_document(doc: dict, cfg: dict) -> list[dict]:
    """Return list of {text, title, page_start, page_end}."""
    chunks: list[dict] = []
    section_title = None
    buffer: list[tuple[str, object]] = []

    def emit(final: bool):
        if not buffer:
            return
        text = "\n".join(t for t, _ in buffer)
        pages = [p for _, p in buffer if p is not None]
        chunk = {"text": text, "title": section_title, "page_start": min(pages) if pages else None,
                 "page_end": max(pages) if pages else None}
        words = word_count(text)
        if words < cfg["min_words"]:
            # Short tail of a section: attach to previous chunk of the same section if it fits.
            prev = chunks[-1] if chunks else None
            if prev and prev["title"] == section_title and \
                    word_count(prev["text"]) + words <= cfg["hard_max_words"]:
                prev["text"] += "\n" + text
                prev["page_end"] = chunk["page_end"] or prev["page_end"]
                buffer.clear()
                return
            # A short but complete section (common in web articles) still makes a
            # usable context; only fragments below min_standalone_words are dropped.
            if words < cfg.get("min_standalone_words", 40):
                buffer.clear()
                return
        chunks.append(chunk)
        buffer.clear()
        # Optional overlap: carry the last sentence forward if it is short enough.
        if cfg["overlap_words"] and not final:
            last = SENTENCE_SPLIT_RE.split(text)[-1]
            if word_count(last) <= cfg["overlap_words"]:
                buffer.append((last, chunk["page_end"]))

    for para in doc["paragraphs"]:
        if para["is_heading"]:
            emit(final=True)
            section_title = para["text"]
            continue
        pieces = split_long(para["text"], cfg["target_max_words"]) \
            if word_count(para["text"]) > cfg["hard_max_words"] else [para["text"]]
        for piece in pieces:
            current_words = sum(word_count(t) for t, _ in buffer)
            if buffer and current_words + word_count(piece) > cfg["target_max_words"] \
                    and current_words >= cfg["target_min_words"]:
                emit(final=False)
            elif buffer and current_words + word_count(piece) > cfg["hard_max_words"]:
                emit(final=False)
            buffer.append((piece, para["page_number"]))
    emit(final=True)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", default="all")
    parser.add_argument("--no-tokenizer", action="store_true", help="Skip XLM-R token counting")
    args = parser.parse_args()
    cfg = load_config()["chunking"]
    manifest = {row["document_id"]: row for row in load_manifest()}
    tokenizer = None if args.no_tokenizer else load_tokenizer(cfg["tokenizer"])
    if tokenizer is None and not args.no_tokenizer:
        print(f"(transformers/{cfg['tokenizer']} not available - token counts skipped)")

    report = read_json(RESULTS_DIR / "chunking_report.json", {}) or {}
    for lang in selected_languages(args.languages):
        rows, dropped = [], Counter()
        for path in sorted((CLEANED_DIR / LANGUAGES[lang]["folder"]).glob("*.json")):
            doc = read_json(path)
            meta = manifest.get(doc["document_id"], {})
            for chunk in chunk_document(doc, cfg):
                ok, reason = language_ok(chunk["text"], lang, cfg["min_context_script_ratio"])
                if not ok:
                    dropped[reason.split("_0")[0]] += 1
                    continue
                n = sum(r["document_id"] == doc["document_id"] for r in rows) + 1
                token_count = len(tokenizer(chunk["text"])["input_ids"]) if tokenizer else None
                rows.append({
                    "context_id": f"{doc['document_id']}_C{n:03d}",
                    "document_id": doc["document_id"],
                    "language": lang,
                    "title": chunk["title"] or meta.get("document_title", ""),
                    "context": chunk["text"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "word_count": word_count(chunk["text"]),
                    "xlmr_tokens": token_count,
                    "exceeds_max_tokens": bool(token_count and token_count > cfg["max_tokens"]),
                    "domain_hint": domain_hint(chunk["text"], lang, meta.get("domain", "")),
                    "source_url": meta.get("source_url") or None,
                    "region": meta.get("region") or None,
                })
        out = CONTEXTS_DIR / f"{lang}_contexts.jsonl"
        write_jsonl(out, rows)
        words = [r["word_count"] for r in rows]
        report[lang] = {
            "contexts": len(rows),
            "documents": len({r["document_id"] for r in rows}),
            "mean_words": round(sum(words) / len(words), 1) if words else 0,
            "min_words": min(words, default=0), "max_words": max(words, default=0),
            "exceeds_max_tokens": sum(r["exceeds_max_tokens"] for r in rows),
            "domain_hints": dict(Counter(r["domain_hint"] for r in rows).most_common()),
            "dropped": dict(dropped),
        }
        print(f"[{lang}] {len(rows)} contexts from {report[lang]['documents']} docs "
              f"(mean {report[lang]['mean_words']} words) -> {out.name}; domains {report[lang]['domain_hints']}")
    write_json(RESULTS_DIR / "chunking_report.json", report)


if __name__ == "__main__":
    main()
