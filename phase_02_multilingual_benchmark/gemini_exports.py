"""Import the retained Gemini cohort and enforce its review gate."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unicodedata import normalize

from .operations import assemble, create_review_worksheets, write_dataset_splits
from .pipeline import (
    LANGUAGES, answer_issues, candidate_fingerprint, digest, file_digest,
    leakage_report, occurrences, quality_issues, read_jsonl, read_reviews,
    review_sample_ids, schema_issues, span_valid, write_json, write_jsonl,
)


EXPORTS = {
    "train": ("train_flat_multilingual_clean.jsonl", 1304),
    "validation": ("fixed_tval_flat.jsonl", 347),
    "test": ("test_flat_multilingual.jsonl", 327),
}
ALL_LANGUAGES = ("en", *LANGUAGES)
JOINERS = {"\u200c", "\u200d"}


def repair_span(original_context, context, answer, offset):
    normalized_answer = normalize("NFC", answer)
    if span_valid(original_context, answer, offset):
        mapped = len(normalize("NFC", original_context[:offset]))
        if span_valid(context, normalized_answer, mapped):
            method = "unchanged" if (normalized_answer, mapped) == (answer, offset) else "nfc_mapped_offset"
            return normalized_answer, mapped, method
    if span_valid(context, normalized_answer, offset):
        return normalized_answer, offset, "nfc_at_recorded_offset"
    matches = occurrences(context, normalized_answer)
    if len(matches) == 1:
        return normalized_answer, matches[0], "unique_exact_span"
    if not matches:
        # Compare without joiners, then copy the exact span with its joiners intact.
        indices = [i for i, char in enumerate(context) if char not in JOINERS]
        plain_context = "".join(context[i] for i in indices)
        plain_answer = "".join(char for char in normalized_answer if char not in JOINERS)
        matches = occurrences(plain_context, plain_answer)
        if len(matches) == 1:
            start = indices[matches[0]]
            end = indices[matches[0] + len(plain_answer) - 1] + 1
            if span_valid(context, context[start:end], start):
                return context[start:end], start, "unique_joiner_equivalent_span"
    return normalized_answer, offset, "unresolved"


def import_rows(directory, sources):
    rows_by_id = {}
    retained_ids = set()
    exclusions = []
    for split, (filename, expected_count) in EXPORTS.items():
        expected_languages = tuple(LANGUAGES) if split == "validation" else ALL_LANGUAGES
        identifiers = {language: set() for language in expected_languages}
        for row in read_jsonl(directory / filename):
            identifier, _, suffix = row["id"].rpartition("_")
            if row["language"] != suffix or suffix not in identifiers or row["id"] in rows_by_id:
                raise ValueError(f"Unexpected language or duplicate row ID in {filename}: {row['id']}")
            if identifier not in sources or sources[identifier]["split"] != split:
                raise ValueError(f"Unknown ID or changed official split in {filename}: {row['id']}")
            rows_by_id[row["id"]] = row
            identifiers[suffix].add(identifier)

        common = set.intersection(*identifiers.values())
        if len(common) != expected_count or (split != "validation" and any(ids != common for ids in identifiers.values())):
            raise ValueError(f"Unexpected shared ID coverage in {filename}")
        retained_ids.update(common)
        for language, ids in identifiers.items():
            exclusions.extend({"split": split, "id": f"{identifier}_{language}",
                               "reason": "not_present_in_all_translated_validation_languages"}
                              for identifier in sorted(ids - common))
    cohort = [source for identifier, source in sources.items() if identifier in retained_ids]
    return rows_by_id, cohort, exclusions


def make_candidate(source, raw, language):
    original_context = raw["context"]
    context = normalize("NFC", original_context) if language != "en" else original_context
    question = normalize("NFC", raw["question"]) if language != "en" else raw["question"]
    texts, starts = raw["answers"]["text"], raw["answers"]["answer_start"]
    identifier = f"{source['id']}_{language}"
    if (not isinstance(texts, list) or not isinstance(starts, list) or not texts
            or len(texts) != len(starts) or not all(isinstance(text, str) for text in texts)
            or not all(type(start) is int for start in starts)):
        raise ValueError(f"Missing or malformed answer references: {identifier}")
    answers = {"text": [], "answer_start": []}
    span_changes = []
    for index, (text, start) in enumerate(zip(texts, starts)):
        if language == "en":
            method = "unchanged" if span_valid(context, text, start) else "unresolved"
        else:
            text, start, method = repair_span(original_context, context, text, start)
        answers["text"].append(text)
        answers["answer_start"].append(start)
        if method != "unchanged":
            span_changes.append({"reference_index": index, "method": method,
                                 "before": {"text": texts[index], "answer_start": starts[index]},
                                 "after": {"text": text, "answer_start": start}})

    candidate = {
        "id": identifier, "original_id": source["id"], "split": source["split"], "language": language,
        "source_document_id": source["source_document_id"],
        "source_document_id_kind": source["source_document_id_kind"],
        "source_row_sha256": source["source_row_sha256"],
        "source_reference_count": len(source["answers"]["text"]), "required_reference_count": len(texts),
        "raw_record_sha256": digest(raw), "context": context, "question": question, "answers": answers,
        "category": raw.get("category"), "human_reviewed": False, "validation_status": "pending_review",
        "translation_status": "complete" if language != "en" else "source",
        "translation_provenance": {"model_id": "gemini-3.5-flash-lite", "origin": "existing_export"} if language != "en" else None,
    }
    structural = schema_issues(candidate, require_answer=True)
    if structural:
        raise ValueError(f"{identifier}: {', '.join(structural)}")
    flags = answer_issues(candidate)
    if language != "en":
        quality_flags, quality = quality_issues(source, candidate)
        flags.extend(flag for flag in quality_flags if flag != "language_id_not_run")
        candidate["quality_checks"] = quality
    candidate["automated_issues"] = sorted(set(flags))
    candidate["candidate_sha256"] = candidate_fingerprint(candidate)

    changed_text = context != original_context or question != raw["question"]
    if not changed_text and not span_changes:
        return candidate, None
    change = {"id": identifier, "split": source["split"], "raw_sha256": candidate["raw_record_sha256"],
              "candidate_sha256": candidate["candidate_sha256"], "context_nfc_changed": context != original_context,
              "question_nfc_changed": question != raw["question"], "answer_diagnostics": span_changes}
    return candidate, change


def prepare_bundle(export_dir, output, sources, audit, config):
    if output.exists():
        raise ValueError(f"Bundle already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".gemini-import-", dir=output.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        inputs = bundle / "inputs"
        inputs.mkdir(parents=True)
        for filename, _ in EXPORTS.values():
            shutil.copyfile(export_dir / filename, inputs / filename)
        raw, cohort, exclusions = import_rows(inputs, {source["id"]: source for source in sources})
        candidates, changes = [], []
        for source in cohort:
            for language in ALL_LANGUAGES:
                if language == "en" and source["split"] == "validation":
                    record = source
                else:
                    record = raw[f"{source['id']}_{language}"]
                candidate, change = make_candidate(source, record, language)
                candidates.append(candidate)
                if change:
                    changes.append(change)

        unresolved = [row for row in candidates if answer_issues(row)]
        blocked_ids = {row["original_id"] for row in unresolved}
        span_checked = [row for row in candidates if row["original_id"] not in blocked_ids]
        span_checked_cohort = [source for source in cohort if source["id"] not in blocked_ids]
        write_dataset_splits(bundle / "span_checked", span_checked_cohort, span_checked)
        write_jsonl(bundle / "candidates.jsonl", candidates, exclusive=True)
        write_jsonl(bundle / "corrections.jsonl", changes, exclusive=True)
        write_jsonl(bundle / "validation_exclusions.jsonl", exclusions, exclusive=True)

        create_review_worksheets(cohort, candidates, config, bundle / "human_review")
        selected = review_sample_ids(cohort, config)
        source_by_id = {source["id"]: source for source in cohort}
        review_material = defaultdict(list)
        for row in candidates:
            if row["automated_issues"] or (row["original_id"], row["language"]) in selected:
                source = source_by_id[row["original_id"]]
                review_material[row["language"]].append({"question_id": row["original_id"], "source": source, "candidate": row})
        for language in ALL_LANGUAGES:
            write_jsonl(bundle / "review_material" / f"{language}.jsonl", review_material[language], exclusive=True)

        manifest = {
            "status": "span_checked_subset_prepared_human_review_pending",
            "source_dataset": {"revision": audit["revision"], "file_sha256": audit["file_sha256"]},
            "input_sha256": {path.name: file_digest(path) for path in sorted(inputs.iterdir())},
            "candidate_sha256": file_digest(bundle / "candidates.jsonl"),
            "file_sha256": {str(path.relative_to(bundle)): file_digest(path) for path in sorted(bundle.rglob("*.jsonl"))},
            "review_policy": config["review_policy"], "review_seed": config["seed"],
            "candidate_rows": dict(Counter(row["split"] for row in candidates)),
            "span_checked_rows": dict(Counter(row["split"] for row in span_checked)),
            "span_checked_question_groups": dict(Counter(source["split"] for source in span_checked_cohort)),
            "unresolved_span_row_ids": [row["id"] for row in unresolved],
            "excluded_validation_export_rows": len(exclusions),
            "automatic_answer_diagnostics": dict(Counter(item["method"] for change in changes for item in change["answer_diagnostics"])),
            "required_review_rows": {lang: len(rows) for lang, rows in review_material.items()},
            "human_reviewed_records": 0,
            "inherited_source_overlap_all_official_ids": audit["leakage"],
            "retained_english_source_overlap": leakage_report(cohort),
            "span_checked_export_overlap": leakage_report(span_checked),
        }
        write_json(bundle / "manifest.json", manifest)
        bundle.rename(output)
    return manifest


def finalize_bundle(bundle, review_dir, sources, audit, config):
    manifest = json.loads((bundle / "manifest.json").read_text())
    path = bundle / "candidates.jsonl"
    if file_digest(path) != manifest["candidate_sha256"]:
        raise ValueError("Candidate file has changed since the worksheets were generated")
    if audit["file_sha256"] != manifest["source_dataset"]["file_sha256"]:
        raise ValueError("The source dataset does not match the import manifest")
    if config["review_policy"] != manifest["review_policy"] or config["seed"] != manifest["review_seed"]:
        raise ValueError("Use the review policy that generated the worksheets")
    candidates = read_jsonl(path)
    identifiers = {row["original_id"] for row in candidates}
    cohort = [source for source in sources if source["id"] in identifiers]
    reviews = read_reviews(review_dir)
    report = assemble(cohort, candidates, reviews, config, bundle / "final")
    report["submitted_review_judgments"] = sum(map(len, reviews.values()))
    write_json(bundle / "finalization_report.json", report)
    return report
