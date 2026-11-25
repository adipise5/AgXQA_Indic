"""Non-neural Phase 2 data handling. No model is loaded by importing this module."""
from collections import Counter, defaultdict
import csv
from functools import lru_cache
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

PHASE = Path(__file__).resolve().parent
REPO = PHASE.parent
REVISION = "9fb0cf2a6d03ccfccbd67d5966179b43b5b77e2c"
COUNTS = {"train": 1503, "validation": 353, "test": 330}
HASHES = {"train": "72da2aaa34d90ce32f28c6d461f258dc9ec712f0af8ff3b5041ceb1528817d85",
          "validation": "386181aa34e6b71faa3474868008cc7b8bed7143da0218f22e546cdfc4ebc583",
          "test": "505fa18e7d7e7b6f86f090ae8c1172d5fd29ba5f6845a82107eb49df59c32a8b"}
LANGUAGES = {"hi": ("hindi", "hin_Deva", "Aditya Saxena"), "mr": ("marathi", "mar_Deva", "Aditya Pise"),
             "bn": ("bengali", "ben_Beng", "Saunak"), "ta": ("tamil", "tam_Taml", "Dhiraj")}
RATING_FIELDS = ("semantic_equivalence", "fluency", "terminology", "answer_correctness")
REVIEW_FIELDS = ["question_id", "language", "split", "reviewer", *RATING_FIELDS, "status", "comments",
                 "reviewed_at", "raw_record_sha256", "corrected_context", "corrected_question", "corrected_answers_json",
                 "override_issues", "adjudicator", "independent_review", "error_categories", "candidate_sha256"]


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    result = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}; preserve the original before repairing") from exc
    return result


def write_jsonl(path, rows, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_config(path=None):
    path = Path(path or PHASE / "config.json")
    if path.suffix in (".yaml", ".yml"):
        import yaml
        config = yaml.safe_load(path.read_text())
    else:
        config = json.loads(path.read_text())
    if config["dataset_revision"] != REVISION or config["languages"] != ["en", "hi", "mr", "bn", "ta"]:
        raise ValueError("Use the pinned official data and all five configured languages")
    return config


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def reference_urls(reference):
    urls = sorted(set(re.findall(r"https?://[^\s<>]+", reference)))
    normalized = []
    for url in urls:
        parts = urlsplit(url.rstrip(".,);]"))
        normalized.append(urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, "")))
    return normalized


def document_identity(row):
    reference = row.get("references", "")
    normalized = reference_urls(reference)
    if normalized:
        return "reference-url:" + digest(normalized), "reference_url_or_url_set"
    if reference.strip() and reference.strip().lower() not in ("none", "n/a", "unknown"):
        return "reference-text:" + digest(reference.strip()), "reference_text_surrogate"
    return "context:" + digest(row["context"]), "context_hash_surrogate"


def span_valid(context, text, start):
    return isinstance(context,str) and isinstance(text, str) and bool(text.strip()) and type(start) is int and start >= 0 and context[start:start + len(text)] == text


def load_source(directory, download=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows, ids, invalid, hashes = [], set(), [], {}
    for split, expected in COUNTS.items():
        filename = f"agxqa-{split}-2024-06-11.jsonl"
        path = directory / filename
        if download and not path.exists():
            url = f"https://huggingface.co/datasets/msu-ceco/agxqa_v1/resolve/{REVISION}/{filename}"
            with urlopen(url, timeout=120) as response:
                data = response.read()
            if digest(data) != HASHES[split]:
                raise ValueError(f"Downloaded {split} hash mismatch")
            path.write_bytes(data)
        hashes[split] = file_digest(path)
        if hashes[split] != HASHES[split]:
            raise ValueError(f"{split} file differs from the pinned official bytes")
        split_rows = read_jsonl(path)
        if len(split_rows) != expected:
            raise ValueError(f"{split}: expected {expected} rows")
        for original in split_rows:
            row = dict(original)
            if not isinstance(row.get("id"), str) or row["id"] in ids:
                raise ValueError("Missing or duplicate original ID")
            ids.add(row["id"])
            texts, starts = row["answers"]["text"], row["answers"]["answer_start"]
            if not texts or len(texts) != len(starts) or not all(isinstance(t, str) for t in texts):
                raise ValueError(f"Invalid source answers for {row['id']}")
            row["original_id"], row["split"] = row["id"], split
            row["source_document_id"], row["source_document_id_kind"] = document_identity(row)
            row["source_invalid_answer_indices"] = [i for i, (t, s) in enumerate(zip(texts, starts)) if not span_valid(row["context"], t, s)]
            row["source_row_sha256"] = digest(original)
            invalid.extend({"id": row["id"], "answer_index": i} for i in row["source_invalid_answer_indices"])
            rows.append(row)
    report = {"dataset": "msu-ceco/agxqa_v1", "revision": REVISION, "counts": COUNTS, "total": len(rows),
              "file_sha256": hashes, "unique_contexts": len({r["context"] for r in rows}),
              "unique_questions": len({r["question"] for r in rows}),
              "unique_answer_strings": len({a for r in rows for a in r["answers"]["text"]}),
              "invalid_source_answer_offsets": invalid, "raw_source_modified": False,
              "document_id_methods": dict(Counter(r["source_document_id_kind"] for r in rows)),
              "leakage": leakage_report(rows)}
    return rows, report


def leakage_report(rows):
    by_id, documents, contexts = defaultdict(set), defaultdict(set), defaultdict(set)
    mismatches = []
    for row in rows:
        original_id = row.get("original_id", row["id"])
        by_id[original_id].add(row["split"])
        documents[row["source_document_id"]].add(row["split"])
        contexts[digest(row["context"])].add(row["split"])
    for key, splits in by_id.items():
        if len(splits) > 1:
            mismatches.append(key)
    return {"cross_split_original_ids": mismatches,
            "cross_split_reference_document_groups": sum(len(v) > 1 for v in documents.values()),
            "cross_split_exact_context_groups": sum(len(v) > 1 for v in contexts.values()),
            "document_disjoint": not any(len(v) > 1 for v in documents.values()),
            "note": "Official splits are preserved. Existing document/context overlap is reported, never hidden by split-prefixed document IDs."}


def canonical_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def occurrences(context, answer):
    answer = canonical_text(answer)
    positions, start = [], 0
    if not answer:
        return positions
    while True:
        index = context.find(answer, start)
        if index < 0:
            return positions
        positions.append(index)
        start = index + 1


def align_answers(context, translated_answers):
    answers, alignment, issues = [], [], []
    for index, translated in enumerate(translated_answers):
        text = canonical_text(translated)
        positions = occurrences(context, text)
        state = "unique_exact_match" if len(positions) == 1 else "not_found" if not positions else "ambiguous_multiple_matches"
        alignment.append({"source_answer_index": index, "translated_answer": translated,
                          "normalized_answer": text, "status": state, "candidate_starts": positions})
        if len(positions) == 1:
            answers.append({"text": context[positions[0]:positions[0]+len(text)], "answer_start": positions[0]})
        else:
            issues.append("answer_" + state)
    return answers, alignment, sorted(set(issues))


def numeric_values(text):
    converted = "".join(str(unicodedata.digit(c)) if c.isdecimal() else c for c in text)
    values = []
    for value in re.findall(r"(?<!\d)[+-]?\d+(?:[,.]\d+)*(?!\d)", converted):
        try:
            values.append(str(Decimal(value.replace(",", "")).normalize()))
        except InvalidOperation:
            values.append(value)
    return Counter(values)


UNIT_ALIASES = {
    "acre": ["acre", "acres", "একর", "एकर", "ஏக்கர்"],
    "inch": ["inch", "inches", "ইঞ্চি", "इंच", "அங்குலம்", "அங்குலங்கள்"],
    "millimetre": ["mm", "millimeter", "millimeters", "millimetre", "millimetres", "মিমি", "মিলিমিটার", "मिमी", "मिलीमीटर", "மிமீ", "மில்லிமீட்டர்"],
    "centimetre": ["cm", "centimeter", "centimeters", "centimetre", "centimetres", "সেমি", "সেন্টিমিটার", "सेमी", "सेंटीमीटर", "செமீ", "சென்டிமீட்டர்"],
    "litre": ["litre", "litres", "liter", "liters", "লিটার", "लीटर", "लिटर", "லிட்டர்"],
    "percent": ["%", "percent", "percentage", "শতাংশ", "प्रतिशत", "टक्के", "சதவீதம்"]}


def units(text):
    result = set()
    for name, aliases in UNIT_ALIASES.items():
        if any(re.search(r"(?<![^\W\d])" + re.escape(alias) + r"(?![^\W\d])", text, re.I) for alias in aliases):
            result.add(name)
    return result


def script_report(text, language):
    blocks = {"bn": [(0x0980, 0x09ff)], "hi": [(0x0900, 0x097f)], "mr": [(0x0900, 0x097f)], "ta": [(0x0b80, 0x0bff)]}
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    expected = sum(any(left <= ord(c) <= right for left, right in blocks[language]) for c in letters)
    ratio = expected / len(letters) if letters else 0.0
    issues = []
    if "\ufffd" in text or any(0xd800 <= ord(c) <= 0xdfff for c in text):
        issues.append("unicode_corruption")
    if any(unicodedata.category(c) == "Cc" and c not in "\n\r\t" for c in text):
        issues.append("unexpected_control_character")
    if ratio < .5:
        issues.append("low_expected_script_ratio")
    if re.search(r"(?:[A-Za-z]{2,}\s+){5}[A-Za-z]{2,}", text):
        issues.append("long_untranslated_latin_passage")
    return {"expected_script_letter_ratio": ratio, "issues": issues,
            "limitation": "Script checks cannot distinguish Hindi from Marathi and do not establish semantic fidelity."}


@lru_cache(maxsize=1)
def language_identifier():
    from langid.langid import LanguageIdentifier, model
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=False)
    identifier.set_languages(["en", "hi", "mr", "bn", "ta"])
    return identifier


def quality_issues(source, candidate, use_langid=False):
    text = candidate["context"] + " " + candidate["question"]
    script = script_report(text, candidate["language"])
    issues = list(script["issues"])
    for field in ("context", "question"):
        if numeric_values(source[field]) != numeric_values(candidate[field]):
            issues.append(field + "_numeric_values_changed")
        if units(source[field]) - units(candidate[field]):
            issues.append(field + "_unit_alias_missing")
    language_id = {"method": "not_run", "predicted_language": None, "score": None}
    if use_langid:
        prediction, score = language_identifier().classify(text)
        language_id = {"method": "langid_1.1.6", "predicted_language": prediction,
                       "score": float(score), "score_is_calibrated_probability": False}
        if prediction != candidate["language"]:
            issues.append("language_id_disagreement")
    else:
        issues.append("language_id_not_run")
    return sorted(set(issues)), {"script": script, "language_id": language_id,
                                 "numeric_and_unit_checks": "Conservative heuristics; equivalent conversions and number words require reviewer judgment."}


def latest_raw(rows):
    latest = {}
    for row in rows:
        key = (row["original_id"], row["language"])
        previous = latest.get(key)
        if previous and (row["attempt"] <= previous["attempt"] or previous["translation_status"] == "complete"):
            raise ValueError(f"Duplicate successful output or unordered retry for {key}")
        latest[key] = row
    return list(latest.values())


def clean_record(source, raw, use_langid=False):
    candidate = {"id": source["id"] + "_" + raw["language"], "original_id": source["id"],
                 "translation_status": raw["translation_status"],
                 "translation_provenance":{"model_id":raw["model_id"],"model_revision":raw["model_revision"],
                                           "backend":raw.get("backend"),"quantization":raw.get("quantization"),
                                           "base_model_id":raw.get("base_model_id"),
                                           "target_language_tag":raw["target_language_tag"]},
                 "language": raw["language"], "split": source["split"], "source_document_id": source["source_document_id"],
                 "source_document_id_kind": source["source_document_id_kind"],
                 "context": canonical_text(raw.get("translations", {}).get("context", "")),
                 "question": canonical_text(raw.get("translations", {}).get("question", "")),
                 "validation_status": "pending_review", "human_reviewed": False,
                 "raw_record_sha256": digest(raw), "source_row_sha256": source["source_row_sha256"],
                 "source_reference_count": len(source["answers"]["text"]), "unicode_normalization": "NFC"}
    translated = raw.get("translations", {}).get("answers", [])
    aligned, alignment, issues = align_answers(candidate["context"], translated)
    if raw["translation_status"] != "complete":
        issues.append("translation_incomplete")
    if len(translated) != candidate["source_reference_count"]:
        issues.append("missing_answer_references")
    if source["source_invalid_answer_indices"]:
        issues.append("invalid_original_gold_offset_requires_source_review")
    issues.extend(raw.get("generation_issues", []))
    additional, details = quality_issues(source, candidate, use_langid) if candidate["context"] else (["empty_context"], {})
    if not candidate["question"]:
        additional.append("empty_question")
    candidate["answers"] = {"text": [a["text"] for a in aligned], "answer_start": [a["answer_start"] for a in aligned]}
    candidate["answer_alignment"] = alignment
    candidate["automated_issues"] = sorted(set(issues + additional))
    candidate["quality_checks"] = details
    candidate["candidate_sha256"] = candidate_fingerprint(candidate)
    return candidate


def source_candidate(row):
    candidate = {"id": row["id"] + "_en", "original_id": row["id"], "language": "en", "split": row["split"],
            "source_document_id": row["source_document_id"], "source_document_id_kind": row["source_document_id_kind"],
            "context": row["context"], "question": row["question"], "answers": row["answers"],
            "source_reference_count": len(row["answers"]["text"]), "human_reviewed": False,
            "validation_status": "source_unchanged" if not row["source_invalid_answer_indices"] else "pending_source_review",
            "automated_issues": ["invalid_original_gold_offset_requires_source_review"] if row["source_invalid_answer_indices"] else [],
            "source_row_sha256": row["source_row_sha256"], "raw_record_sha256": row["source_row_sha256"]}
    candidate["candidate_sha256"] = candidate_fingerprint(candidate)
    return candidate


def candidate_fingerprint(row):
    fields = {field:row.get(field) for field in ("id","original_id","language","split","source_document_id",
                   "context","question","answers","source_reference_count","automated_issues",
                   "source_row_sha256","raw_record_sha256")}
    if "required_reference_count" in row:
        fields["required_reference_count"] = row["required_reference_count"]
    return digest(fields)


def schema_issues(row, require_answer=False):
    if not isinstance(row,dict):
        return ["schema_record"]
    issues = []
    for key in ("id", "original_id", "language", "split", "source_document_id", "context", "question"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            issues.append("schema_" + key)
    if row.get("language") not in ("en", *LANGUAGES) or row.get("split") not in COUNTS:
        issues.append("schema_language_or_split")
    if row.get("id") != str(row.get("original_id")) + "_" + str(row.get("language")):
        issues.append("schema_id_alignment")
    if type(row.get("human_reviewed")) is not bool or not isinstance(row.get("validation_status"),str):
        issues.append("schema_validation_metadata")
    answer = row.get("answers", {})
    if not isinstance(answer, dict):
        return issues + ["schema_answers"]
    texts, starts = answer.get("text"), answer.get("answer_start")
    if not isinstance(texts, list) or not isinstance(starts, list) or len(texts) != len(starts):
        issues.append("schema_answers")
    elif not all(isinstance(text,str) for text in texts) or not all(type(start) is int for start in starts):
        issues.append("schema_answer_types")
    elif (require_answer or row.get("validation_status") in ("accepted", "automated_accepted")) and not texts:
        issues.append("empty_accepted_answers")
    return issues


def answer_issues(row):
    if not isinstance(row,dict):
        return ["schema_record"]
    answer = row.get("answers", {})
    if not isinstance(answer, dict):
        return ["schema_answers"]
    texts, starts = answer.get("text", []), answer.get("answer_start", [])
    if not isinstance(texts,list) or not isinstance(starts,list) or not texts or len(texts) != len(starts):
        return ["missing_answer_spans"]
    return ["answer_span_mismatch"] if any(not span_valid(row.get("context", ""), t, s) for t, s in zip(texts, starts)) else []


def review_sample_ids(source_rows, config):
    selected = set()
    for language in LANGUAGES:
        for split in COUNTS:
            rows = sorted([r["id"] for r in source_rows if r["split"] == split])
            fraction = config["review_policy"][split + "_fraction"]
            if not 0 <= fraction <= 1:
                raise ValueError("Review fractions must be between zero and one")
            random.Random(str(config["seed"]) + language + split).shuffle(rows)
            selected.update((original_id, language) for original_id in rows[:math.ceil(len(rows)*fraction)])
    return selected


def read_reviews(directory):
    result = defaultdict(list)
    for path in Path(directory).glob("*_review.csv"):
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(REVIEW_FIELDS).issubset(reader.fieldnames or []):
                raise ValueError(f"Review worksheet has missing columns: {path}")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed CSV row in {path}; quote embedded commas/newlines")
                if not any(str(v or "").strip() for v in row.values()):
                    continue
                if not row.get("status", "").strip():
                    continue  # An unfilled worksheet row is not a human judgment.
                status = row["status"].strip().lower()
                if status not in ("accept", "correct", "reject") or not row.get("reviewer", "").strip() or not row.get("reviewed_at", "").strip():
                    raise ValueError(f"Incomplete reviewer identity, timestamp, or decision in {path}")
                for field in RATING_FIELDS:
                    if not str(row.get(field) or "").isdigit() or not 1 <= int(row[field]) <= 5:
                        raise ValueError(f"Human rating {field} must be 1–5 in {path}")
                timestamp = datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    raise ValueError("Review timestamps must include a timezone")
                row["status"] = status
                if row["language"] not in ("en",*LANGUAGES) or row["split"] not in COUNTS or not row["question_id"]:
                    raise ValueError(f"Invalid review question/language/split in {path}")
                if not re.fullmatch(r"[0-9a-f]{64}",row["raw_record_sha256"]):
                    raise ValueError(f"Review requires its candidate's original raw SHA256: {path}")
                if not re.fullmatch(r"[0-9a-f]{64}",row["candidate_sha256"]):
                    raise ValueError(f"Review requires its exact cleaned candidate SHA256: {path}")
                if row["independent_review"].lower() not in ("", "true", "false"):
                    raise ValueError("independent_review must be blank, true, or false")
                result[(row["question_id"], row["language"])].append(row)
    for entries in result.values():
        if len({r["reviewer"] for r in entries}) != len(entries):
            raise ValueError("Multiple active judgments by the same reviewer require explicit version/adjudication")
    return result


def apply_review(candidate, reviews, source=None):
    if not reviews:
        return candidate, ["human_review_required"]
    if any(r.get("raw_record_sha256") != candidate["raw_record_sha256"] or r["split"] != candidate["split"] for r in reviews):
        return candidate, ["review_raw_hash_or_split_mismatch"]
    fingerprint = candidate_fingerprint(candidate)
    if candidate.get("candidate_sha256") != fingerprint or any(r.get("candidate_sha256") != fingerprint for r in reviews):
        return candidate, ["review_candidate_hash_mismatch"]
    if len({r["reviewer"] for r in reviews}) != len(reviews):
        return candidate, ["duplicate_reviewer_requires_version_adjudication"]
    signatures = {(r["status"], r.get("corrected_context", ""), r.get("corrected_question", ""), r.get("corrected_answers_json", "")) for r in reviews}
    adjudicated = [r for r in reviews if r.get("adjudicator", "").strip()]
    if len(adjudicated) > 1 or (len(signatures) > 1 and len(adjudicated) != 1):
        return candidate, ["reviewer_disagreement_requires_adjudication"]
    review = adjudicated[0] if adjudicated else reviews[0]
    if review["status"] == "reject":
        return candidate, ["human_rejected"]
    corrected = dict(candidate)
    if review["status"] == "correct":
        for field in ("context", "question"):
            if review.get("corrected_" + field, "").strip():
                corrected[field] = review["corrected_" + field]
        if review.get("corrected_answers_json", "").strip():
            corrected["answers"] = json.loads(review["corrected_answers_json"])
        if not review.get("comments", "").strip():
            return candidate, ["correction_requires_rationale"]
        if all(corrected.get(key) == candidate.get(key) for key in ("context", "question", "answers")):
            return candidate, ["correct_decision_contains_no_changes"]
    else:
        if any(review.get(field, "").strip() for field in ("corrected_context", "corrected_question", "corrected_answers_json")):
            return candidate, ["accept_decision_contains_unapplied_correction"]
    fatal = schema_issues(corrected, require_answer=True) + answer_issues(corrected)
    if fatal:
        return candidate, sorted(set(fatal))
    combined = corrected["context"] + corrected["question"]
    if "\ufffd" in combined or any(0xd800 <= ord(c) <= 0xdfff for c in combined):
        fatal.append("unicode_corruption")
    required_references = candidate.get("required_reference_count", candidate["source_reference_count"])
    if len(corrected.get("answers", {}).get("text", [])) != required_references:
        fatal.append("reference_count_changed_requires_separate_annotation_protocol")
    if fatal:
        return candidate, sorted(set(fatal))
    new_issues = []
    if source and corrected["language"] != "en":
        new_issues, _ = quality_issues(source,corrected,use_langid=False)
        # A named native reviewer supplies language verification for this corrected text.
        new_issues = [issue for issue in new_issues if issue != "language_id_not_run"]
    if corrected["language"] != "en" and any(unicodedata.normalize("NFC",corrected[f]) != corrected[f] for f in ("context","question")):
        return candidate, ["corrected_text_requires_NFC_normalization"]
    overrides = {v.strip() for v in review.get("override_issues", "").split(";") if v.strip()}
    unresolved = (set(candidate["automated_issues"]) | set(new_issues)) - overrides
    if review["status"] == "correct":
        unresolved -= {"answer_not_found", "answer_ambiguous_multiple_matches", "missing_answer_references", "empty_context", "empty_question",
                       "invalid_original_gold_offset_requires_source_review", "answer_span_mismatch"}
    if unresolved or (overrides and not review.get("comments", "").strip()):
        return candidate, ["automated_flags_require_explicit_review_resolution:" + ";".join(sorted(unresolved))]
    corrected.update({"validation_status": "accepted", "human_reviewed": True, "reviewer": review["reviewer"],
                      "reviewed_at": review["reviewed_at"], "review_decision": review["status"],
                      "review_comments": review["comments"], "resolved_issue_codes": sorted(overrides),
                      "reviewed_candidate_sha256":fingerprint,
                      "post_review_automated_issue_codes":sorted(set(new_issues)),
                      "human_error_categories":[s.strip() for s in review.get("error_categories", "").split(";") if s.strip()],
                      "human_ratings": {f: int(review[f]) for f in RATING_FIELDS}})
    return corrected, []


def kappa(first, second):
    if len(first) != len(second) or not first:
        return None
    count = len(first)
    a, b = Counter(first), Counter(second)
    observed = sum(x == y for x,y in zip(first, second)) / count
    expected = sum(a[v] * b[v] for v in set(a) | set(b)) / count**2
    return None if expected == 1 else (observed - expected) / (1 - expected)


def as_shared(row):
    converted = dict(row)
    converted["id"] = row["original_id"]
    converted["answers"] = [{"text": t, "answer_start": s} for t,s in zip(row["answers"]["text"],row["answers"]["answer_start"])]
    return converted
