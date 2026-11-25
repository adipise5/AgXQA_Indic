"""Import the retained Gemini cohort and enforce its review gate."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from unicodedata import normalize

from .gemini_audit import (
    ALL_LANGUAGES, document_disjoint, excluded_groups, exclusion_distribution, extractability,
    train_test_overlap, validate_parallel,
)
from .operations import assemble, create_review_worksheets, write_dataset_splits
from .pipeline import (
    LANGUAGES, PHASE, answer_issues, candidate_fingerprint, digest, file_digest,
    leakage_report, occurrences, quality_issues, read_jsonl, read_reviews,
    review_sample_ids, schema_issues, span_valid, write_json, write_jsonl,
)


EXPORTS = {
    "train": ("train_flat_multilingual_clean.jsonl", 1304),
    "validation": ("fixed_tval_flat.jsonl", 347),
    "test": ("test_flat_multilingual.jsonl", 327),
}
JOINERS = {"\u200c", "\u200d"}
VERSION = "gemini-v2"
VALIDATION_EVIDENCE = (
    "validation_multilingual_raw.jsonl", "validation_multilingual_aligned.jsonl", "fixed_multilingual_aligned.jsonl",
)


def repair_span(original_context, context, answer, offset):
    normalized_answer = normalize("NFC", answer.strip())
    if not normalized_answer:
        return answer, offset, "unresolved"
    if span_valid(original_context, answer, offset):
        first_character = offset + len(answer) - len(answer.lstrip())
        mapped = len(normalize("NFC", original_context[:first_character]))
        if span_valid(context, normalized_answer, mapped):
            method = "unchanged" if (normalized_answer, mapped) == (answer, offset) else "nfc_mapped_offset"
            return normalized_answer, mapped, method
    if span_valid(context, normalized_answer, offset):
        return normalized_answer, offset, "nfc_at_recorded_offset"
    matches = occurrences(context, normalized_answer)
    if len(matches) == 1:
        return normalized_answer, matches[0], "unique_exact_span"
    if not matches:
        parts = normalized_answer.split()
        if len(parts) > 1:
            pattern = r"\s+".join(re.escape(part) for part in parts)
            whitespace_matches = list(re.finditer(f"(?=({pattern}))", context))
            if len(whitespace_matches) == 1:
                match = whitespace_matches[0]
                return match.group(1), match.start(), "unique_whitespace_equivalent_span"
            if whitespace_matches:
                return normalized_answer, offset, "unresolved"
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


def prepare_bundle(export_dir, output, sources, audit, config, validation_evidence_dir=None):
    if output.exists():
        raise ValueError(f"Bundle already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".gemini-import-", dir=output.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        inputs = bundle / "inputs"
        inputs.mkdir(parents=True)
        for filename, _ in EXPORTS.values():
            shutil.copyfile(export_dir / filename, inputs / filename)
        evidence = bundle / "evidence"
        evidence.mkdir()
        shutil.copyfile(PHASE / "configs" / "gemini_provenance.json", evidence / "provenance.json")
        validation_raw = {}
        if validation_evidence_dir is not None:
            for filename in VALIDATION_EVIDENCE:
                shutil.copyfile(validation_evidence_dir / filename, evidence / filename)
            responses = read_jsonl(evidence / VALIDATION_EVIDENCE[0])
            validation_raw = {row["id"]: row for row in responses}
            expected = {source["id"] for source in sources if source["split"] == "validation"}
            if len(responses) != len(validation_raw) or set(validation_raw) != expected:
                raise ValueError("Raw validation evidence does not match the official validation IDs")

        source_by_id = {source["id"]: source for source in sources}
        raw, cohort, exclusions = import_rows(inputs, source_by_id)
        official_exclusions = excluded_groups(sources, cohort, raw, validation_raw)
        raw_rows = [dict(row, original_id=row["id"].rsplit("_", 1)[0],
                         split=source_by_id[row["id"].rsplit("_", 1)[0]]["split"]) for row in raw.values()]
        candidates, changes = [], []
        for source in cohort:
            for language in ALL_LANGUAGES:
                if language == "en":
                    record = source
                    export = raw.get(f"{source['id']}_en")
                    if export is not None:
                        fields = [field for field in ("context", "question", "answers", "category") if source[field] != export.get(field)]
                        if fields:
                            changes.append({"id": export["id"], "split": source["split"],
                                            "method": "pinned_english_source_replaces_export_variation", "fields": fields,
                                            "export_row_sha256": digest(export), "english_source_row_sha256": source["source_row_sha256"]})
                else:
                    record = raw[f"{source['id']}_{language}"]
                candidate, change = make_candidate(source, record, language)
                candidates.append(candidate)
                if change:
                    changes.append(change)

        validate_parallel(candidates, cohort)
        unresolved = [row for row in candidates if answer_issues(row)]
        blocked_ids = {row["original_id"] for row in unresolved}
        span_checked = [row for row in candidates if row["original_id"] not in blocked_ids]
        span_checked_cohort = [source for source in cohort if source["id"] not in blocked_ids]
        validate_parallel(span_checked, span_checked_cohort)
        write_dataset_splits(bundle / "span_checked", span_checked_cohort, span_checked)
        write_jsonl(bundle / "candidates.jsonl", candidates, exclusive=True)
        write_jsonl(bundle / "corrections.jsonl", changes, exclusive=True)
        write_jsonl(bundle / "validation_exclusions.jsonl", exclusions, exclusive=True)
        write_jsonl(bundle / "excluded_question_groups.jsonl", official_exclusions, exclusive=True)
        write_jsonl(bundle / "retained_question_groups.jsonl",
                    [{"original_id": row["id"], "split": row["split"], "source_document_id": row["source_document_id"],
                      "span_checked": row["id"] not in blocked_ids} for row in cohort], exclusive=True)

        sensitivity, sensitivity_dropped, reassigned = document_disjoint(span_checked, span_checked_cohort, config["seed"])
        sensitivity_splits = {row["id"]: row["split"] for row in sensitivity}
        sensitivity_rows = [{**row, "split": sensitivity_splits[row["original_id"]], "official_split": row["split"]}
                            for row in span_checked if row["original_id"] in sensitivity_splits]
        validate_parallel(sensitivity_rows, sensitivity)
        if {source["split"] for source in sensitivity} == set(EXPORTS):
            write_dataset_splits(bundle / "document_disjoint", sensitivity, sensitivity_rows)
        write_jsonl(bundle / "document_disjoint_exclusions.jsonl", sensitivity_dropped, exclusive=True)
        write_json(bundle / "validation_report.json", {
            "raw_export_extractability": extractability(raw_rows, source_by_id),
            "candidate_extractability": extractability(candidates, source_by_id),
            "span_checked_extractability": extractability(span_checked, source_by_id),
            "reference_policy": "English uses the pinned source. Indic reference counts reflect the recovered exports. Counts do not measure semantic reference preservation.",
            "official_export_exclusions_by_split": dict(Counter(row["split"] for row in official_exclusions)),
            "exclusion_distribution": exclusion_distribution(sources, cohort),
            "val_rows_without_all_translation_languages": len(exclusions),
            "pending_english_test_groups": sorted(blocked_ids),
            "retained_candidate_parallel_coverage_valid": True,
            "span_checked_parallel_coverage_valid": True,
        })
        write_json(bundle / "overlap_report.json", {
            "official_english": train_test_overlap(sources),
            "retained_cohort_before_span_gate": train_test_overlap(candidates),
            "span_checked_release": train_test_overlap(span_checked),
            "document_disjoint": {
                "group_counts": dict(Counter(row["split"] for row in sensitivity)),
                "reassigned_from_official_train_to_validation": reassigned,
                "excluded_group_counts": dict(Counter(row["split"] for row in sensitivity_dropped)),
                "overlap": train_test_overlap(sensitivity_rows),
                "model_evaluation_performed": False,
            },
        })

        create_review_worksheets(cohort, candidates, config, bundle / "human_review")
        selected = review_sample_ids(cohort, config)
        review_material = defaultdict(list)
        for row in candidates:
            if row["automated_issues"] or (row["original_id"], row["language"]) in selected:
                source = source_by_id[row["original_id"]]
                review_material[row["language"]].append({"question_id": row["original_id"], "source": source, "candidate": row})
        for language in ALL_LANGUAGES:
            write_jsonl(bundle / "review_material" / f"{language}.jsonl", review_material[language], exclusive=True)

        manifest = {
            "dataset_version": VERSION,
            "status": "span_checked_subset_prepared_human_review_pending",
            "source_dataset": {"revision": audit["revision"], "file_sha256": audit["file_sha256"]},
            "input_sha256": {path.name: file_digest(path) for path in sorted(inputs.iterdir())},
            "candidate_sha256": file_digest(bundle / "candidates.jsonl"),
            "file_sha256": {str(path.relative_to(bundle)): file_digest(path) for path in sorted(bundle.rglob("*"))
                            if path.is_file() and path.parent.name != "human_review"},
            "review_policy": config["review_policy"], "review_seed": config["seed"],
            "candidate_rows": dict(Counter(row["split"] for row in candidates)),
            "span_checked_rows": dict(Counter(row["split"] for row in span_checked)),
            "span_checked_question_groups": dict(Counter(source["split"] for source in span_checked_cohort)),
            "unresolved_span_row_ids": [row["id"] for row in unresolved],
            "excluded_validation_export_rows": len(exclusions),
            "automatic_answer_diagnostics": dict(Counter(item["method"] for change in changes for item in change.get("answer_diagnostics", []))),
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
    if report["status"] == "ready":
        final_manifest = bundle / "final" / "build_manifest.json"
        record = json.loads(final_manifest.read_text(encoding="utf-8"))
        record.update(dataset_version=manifest.get("dataset_version"),
                      parent_bundle_manifest_sha256=file_digest(bundle / "manifest.json"))
        final_rows = [row for split in EXPORTS for row in read_jsonl(bundle / "final" / f"{split}.jsonl")]
        validate_parallel(final_rows, cohort)
        final_audit = bundle / "final" / "overlap_report.json"
        write_json(final_audit, train_test_overlap(final_rows))
        record["file_sha256"][final_audit.name] = file_digest(final_audit)
        write_json(final_manifest, record)
    report["submitted_review_judgments"] = sum(map(len, reviews.values()))
    write_json(bundle / "finalization_report.json", report)
    return report
