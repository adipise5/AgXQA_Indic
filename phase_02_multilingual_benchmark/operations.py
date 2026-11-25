"""Candidate validation, human-review gates, and transactional dataset assembly."""
from collections import Counter, defaultdict
import csv
import json
from itertools import combinations
from pathlib import Path

from .pipeline import (COUNTS, LANGUAGES, PHASE, RATING_FIELDS, REVIEW_FIELDS, answer_issues, apply_review,
                       as_shared, clean_record, digest, latest_raw, leakage_report, read_jsonl, read_reviews,
                       review_sample_ids, schema_issues, source_candidate, write_json, write_jsonl, file_digest, candidate_fingerprint)


def candidate_rows(directory):
    result = []
    for name in ("english", *(v[0] for v in LANGUAGES.values())):
        path = Path(directory)/(name+".jsonl")
        if path.exists():
            result.extend(read_jsonl(path))
    return result


def clean_translations(source_rows, raw_directory, output, use_langid=True):
    output, raw_directory = Path(output), Path(raw_directory)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Candidate directory is nonempty; create a new version instead of overwriting edits")
    output.mkdir(parents=True,exist_ok=True)
    lookup = {row["id"]:row for row in source_rows}
    run = json.loads((raw_directory/"run_manifest.json").read_text())
    if any(not (raw_directory/name).exists() or file_digest(raw_directory/name) != expected for name,expected in run.get("raw_files_sha256",{}).items()):
        raise ValueError("Raw-file checksums disagree with their recorded run; preserve and investigate before cleaning")
    write_jsonl(output/"english.jsonl",[source_candidate(r) for r in source_rows],exclusive=True)
    total = 0
    for language,(name,_,_) in LANGUAGES.items():
        path = raw_directory/(name+".jsonl")
        raw = latest_raw(read_jsonl(path)) if path.exists() else []
        cleaned = []
        for row in raw:
            source = lookup.get(row["original_id"])
            if not source or row["language"] != language or row["split"] != source["split"] or row["source_row_sha256"] != source["source_row_sha256"]:
                raise ValueError("Raw output has an unknown ID, language, changed source, or split mismatch")
            if row["model_id"] != run["model_id"] or row["model_revision"] != run["model_revision"] or row["target_language_tag"] != LANGUAGES[language][1]:
                raise ValueError("Raw row disagrees with recorded model/revision/target tag")
            cleaned.append(clean_record(source,row,use_langid))
        write_jsonl(output/(name+".jsonl"),cleaned,exclusive=True)
        total += len(cleaned)
    write_json(output/"candidate_manifest.json",{"status":"pending_human_review","translated_candidate_rows":total,
               "english_rows":len(source_rows),"raw_outputs_modified":False,"langid_requested":use_langid,
               "raw_run_manifest_sha256":file_digest(raw_directory/"run_manifest.json"),
               "raw_files_sha256":{path.name:file_digest(path) for path in raw_directory.glob("*.jsonl")},
               "cleaning_source_sha256":{path.name:file_digest(path) for path in (PHASE/"pipeline.py",PHASE/"operations.py")}})
    return total


def validate_candidates(rows, source_rows):
    source = {r["id"]:r for r in source_rows}
    seen, reports = set(), []
    for row in rows:
        if not isinstance(row,dict):
            reports.append({"id":None,"original_id":None,"language":None,"split":None,"schema_valid":False,
                            "answer_span_valid":False,"all_reference_spans_valid":False,"language_id":None,"issues":"schema_record"})
            continue
        key = tuple(row.get(field) if isinstance(row.get(field),str) else None for field in ("original_id","language"))
        issues = schema_issues(row) + answer_issues(row)
        if key in seen:
            issues.append("duplicate_original_id_language")
        seen.add(key)
        original = source.get(key[0])
        if not original or row.get("split") != original["split"]:
            issues.append("unknown_id_or_split_changed")
        if original and (row.get("source_row_sha256") != original["source_row_sha256"] or row.get("source_document_id") != original["source_document_id"]):
            issues.append("source_identity_mismatch")
        answer_map = row.get("answers",{}) if isinstance(row.get("answers"),dict) else {}
        answer_texts = answer_map.get("text",[]) if isinstance(answer_map.get("text",[]),list) else []
        if original and len(answer_texts) != len(original["answers"]["text"]):
            issues.append("missing_answer_references")
        stored_issues = row.get("automated_issues",[])
        if not isinstance(stored_issues,list) or not all(isinstance(v,str) for v in stored_issues):
            stored_issues = ["schema_automated_issues"]
        issues = sorted(set(issues + stored_issues))
        quality = row.get("quality_checks",{}) if isinstance(row.get("quality_checks"),dict) else {}
        language_id = quality.get("language_id",{}) if isinstance(quality.get("language_id"),dict) else {}
        reports.append({"id":row.get("id"),"original_id":row.get("original_id"),"language":row.get("language"),
                        "split":row.get("split"),"schema_valid":not schema_issues(row),"answer_span_valid":not answer_issues(row),
                        "all_reference_spans_valid":not answer_issues(row) and not any(v in issues for v in ("missing_answer_references","answer_not_found","answer_ambiguous_multiple_matches")),
                        "language_id":language_id.get("predicted_language"),
                        "issues":";".join(issues)})
    expected = {(r["id"],language) for r in source_rows for language in ("en",*LANGUAGES)}
    return reports, {"expected_records":len(expected),"observed_unique_records":len(seen),"missing_pair_count":len(expected-seen),
                     "missing_pair_examples":[{"original_id":a,"language":b} for a,b in sorted(expected-seen)[:100]],
                     "extra_pairs":[{"original_id":a,"language":b} for a,b in sorted(seen-expected,key=lambda pair:tuple(str(v) for v in pair))]}


def create_review_worksheets(source_rows, candidates, config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    selected = review_sample_ids(source_rows,config)
    for language,name in [("en","english_source"),*((key,value[0]) for key,value in LANGUAGES.items())]:
        path = directory/(name+"_review.csv")
        if path.exists() and len(path.read_text().splitlines()) > 1:
            raise ValueError(f"Existing review judgments/worksheet must not be overwritten: {path}")
        with path.open("w",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=REVIEW_FIELDS,lineterminator="\n")
            writer.writeheader()
            for row in candidates:
                key = (row["original_id"],row["language"])
                required = bool(row["automated_issues"]) or key in selected
                if row["language"] == language and required:
                    writer.writerow({"question_id":row["original_id"],"language":language,"split":row["split"],
                                     "candidate_sha256":row["candidate_sha256"],
                                     "raw_record_sha256":row["raw_record_sha256"]})


def write_dataset_splits(directory, source_rows, rows):
    from shared.validate import validate

    for split in COUNTS:
        split_rows = [row for row in rows if row["split"] == split]
        shared = [as_shared(row) for row in split_rows]
        validate(shared)  # The manifest reports inherited document overlap across splits.
        write_jsonl(directory / f"{split}.jsonl", split_rows, exclusive=True)
        write_jsonl(directory / "shared" / f"{split}.jsonl", shared, exclusive=True)

    groups = defaultdict(dict)
    for row in rows:
        groups[row["original_id"]][row["language"]] = row
    parallel = [{"original_id": source["id"], "split": source["split"],
                 "source_document_id": source["source_document_id"], "languages": groups[source["id"]]}
                for source in source_rows]
    write_jsonl(directory / "parallel_multilingual.jsonl", parallel, exclusive=True)


def assemble(source_rows, candidates, reviews, config, output, reviewed_output=None):
    lookup, duplicate = {}, []
    for row in candidates:
        key = (row["original_id"],row["language"])
        if key in lookup:
            duplicate.append(key)
        lookup[key] = row
    selected = review_sample_ids(source_rows,config)
    if set(reviews) - set(lookup):
        raise ValueError("Reviews refer to missing candidate records")
    accepted, blocked = [], []
    for source in source_rows:
        for language in ("en",*LANGUAGES):
            key = (source["id"],language)
            row = lookup.get(key)
            if not row:
                blocked.append({"original_id":key[0],"language":language,"issues":["missing_candidate"]})
                continue
            if row["split"] != source["split"] or row["source_row_sha256"] != source["source_row_sha256"] or row["source_document_id"] != source["source_document_id"]:
                blocked.append({"original_id":key[0],"language":language,"issues":["source_identity_or_split_mismatch"]})
                continue
            if row.get("candidate_sha256") != candidate_fingerprint(row):
                blocked.append({"original_id":key[0],"language":language,"issues":["candidate_changed_after_cleaning_requires_new_version"]})
                continue
            needs_review = bool(row["automated_issues"]) or key in selected
            if needs_review or reviews.get(key):
                row, issues = apply_review(row,reviews.get(key,[]),source)
            else:
                issues = schema_issues(row,require_answer=True) + answer_issues(row)
                row = {**row,"validation_status":"source_unchanged" if language == "en" else "automated_accepted","human_reviewed":False}
            if issues:
                blocked.append({"original_id":key[0],"language":language,"issues":issues})
            else:
                accepted.append(row)
    expected = {(r["id"],language) for r in source_rows for language in ("en",*LANGUAGES)}
    extra = set(lookup)-expected
    ready = not blocked and not duplicate and not extra and len(accepted) == len(expected)
    report = {"status":"ready" if ready else "blocked_pending_validation_or_review","expected_records":len(expected),
              "accepted_records":len(accepted),"human_reviewed_records":sum(r["human_reviewed"] for r in accepted),
              "blocked":blocked,"duplicate_pairs":duplicate,"extra_pairs":sorted(extra),
              "source_leakage_preserved":leakage_report(source_rows),"review_policy":config["review_policy"],
              "note":"Final files are produced only after complete ID coverage and required reviews; no partial benchmark is silently emitted."}
    if not ready:
        return report
    if reviewed_output and any((Path(reviewed_output)/(name+".jsonl")).exists() for name,_,_ in LANGUAGES.values()):
        raise ValueError("Human-reviewed files already exist; use a new version instead of overwriting them")
    output = Path(output)
    if output.exists() and any(p.name != ".gitkeep" for p in output.iterdir()):
        raise ValueError("Final dataset directory is nonempty; use a versioned new destination")
    staging = output.with_name(output.name+".staging")
    if staging.exists():
        raise ValueError("A prior staging directory exists; inspect it before building another version")
    staging.mkdir(parents=True)
    write_dataset_splits(staging, source_rows, accepted)
    write_json(staging/"build_manifest.json",{**report,"file_sha256":{str(p.relative_to(staging)):digest(p.read_bytes()) for p in staging.rglob("*.jsonl")}})
    if output.exists():
        # Keep an existing tracked .gitkeep; publish complete validated files without overwriting data.
        for path in staging.iterdir():
            path.rename(output/path.name)
        staging.rmdir()
    else:
        staging.rename(output)
    if reviewed_output:
        reviewed_output = Path(reviewed_output)
        for language,(name,_,_) in LANGUAGES.items():
            write_jsonl(reviewed_output/(name+".jsonl"),[r for r in accepted if r["language"] == language and r["human_reviewed"]],exclusive=True)
    return report


def statistics(source_rows, candidates, reviews, config):
    source_lookup = {row["id"]:row for row in source_rows}
    candidate_lookup = {(r["original_id"],r["language"]):r for r in candidates}
    for key, entries in reviews.items():
        if key[0] not in source_lookup or key not in candidate_lookup:
            raise ValueError("Review refers to an unknown ID or missing candidate")
        for review in entries:
            if review["split"] != source_lookup[key[0]]["split"] or review["raw_record_sha256"] != candidate_lookup[key]["raw_record_sha256"]:
                raise ValueError("Review is stale or has changed official split membership")
            if review.get("candidate_sha256") != candidate_fingerprint(candidate_lookup[key]):
                raise ValueError("Review was made against a different cleaned candidate version")
    summary = {"status":"machine_translation_not_run" if not any(r["language"] != "en" for r in candidates) else "validation_in_progress",
               "expected":{"per_language":COUNTS,"per_language_total":len(source_rows),"translated_total":len(source_rows)*4,
                           "all_language_total":len(source_rows)*5},
               "observed":{},"human_quality":{},"human_review_coverage":{},"cohens_kappa":None,
               "kappa_note":"Only independent reviewers rating the same language/examples can establish agreement; one reviewer per language does not.",
               "source_leakage":leakage_report(source_rows),"new_model_execution_by_reporting":False,
               "quality_rating_denominator":"All submitted human judgments; coverage counts unique original IDs.",
               "paired_span_validity":{}}
    for language in ("en",*LANGUAGES):
        rows = [r for r in candidates if r["language"] == language]
        for split in COUNTS:
            expected = sum(r["split"] == split for r in source_rows)
            relevant = [r for r in rows if r["split"] == split]
            scored_reviews = [review for key,values in reviews.items() if key[1] == language for review in values if review["split"] == split]
            reviewed_ids = {r["question_id"] for r in scored_reviews}
            key = language+"/"+split
            summary["observed"][key] = {"expected":expected,"candidate_rows":len(relevant),
                     "successful_translation_rows":sum(r["language"] == "en" or r.get("translation_status") == "complete" for r in relevant),
                     "translation_completion_rate":sum(r["language"] == "en" or r.get("translation_status") == "complete" for r in relevant)/expected if expected else None,
                     "valid_span_rows":sum(not answer_issues(r) and len(r["answers"]["text"]) == r["source_reference_count"] for r in relevant),
                     "span_validity_rate":sum(not answer_issues(r) and len(r["answers"]["text"]) == r["source_reference_count"] for r in relevant)/len(relevant) if relevant else None}
            summary["human_review_coverage"][key] = {"reviewed_unique_ids":len(reviewed_ids),"total_expected":expected,
                                                    "coverage":len(reviewed_ids)/expected if expected else None}
        values = [review for key,entries in reviews.items() if key[1] == language for review in entries]
        decisions = Counter(r["status"] for r in values)
        summary["human_quality"][language] = {"review_judgments":len(values),
             **{field:sum(int(r[field]) for r in values)/len(values) if values else None for field in RATING_FIELDS},
             "acceptance_without_correction_rate":decisions["accept"]/len(values) if values else None,
             "correction_rate":decisions["correct"]/len(values) if values else None,
             "rejection_rate":decisions["reject"]/len(values) if values else None}
    summary["error_counts"] = dict(Counter(issue for row in candidates for issue in row["automated_issues"]))
    summary["human_error_counts"] = dict(Counter(category.strip() for entries in reviews.values() for review in entries
                   for category in review.get("error_categories", "").split(";") if category.strip()))
    for language in LANGUAGES:
        pairs, unresolved = [], 0
        for key, entries in reviews.items():
            if key[1] != language:
                continue
            before = candidate_lookup[key]
            after, issues = apply_review(before,entries,source_lookup[key[0]])
            if issues:
                unresolved += 1
            else:
                pairs.append((before,after))
        summary["paired_span_validity"][language] = {
            "accepted_review_resolved_pairs":len(pairs),"reviewed_unresolved_records":unresolved,
            "before_valid_rows":sum(not answer_issues(a) and len(a["answers"]["text"]) == a["source_reference_count"] for a,b in pairs),
            "after_valid_rows":sum(not answer_issues(b) for a,b in pairs),
            "denominator_note":"Same review-resolved IDs before/after; excludes rejects and unresolved reviews. Not full-dataset coverage."}
    agreement = []
    from .pipeline import kappa
    for language in LANGUAGES:
        independent = defaultdict(dict)
        for key, entries in reviews.items():
            if key[1] != language:
                continue
            for review in entries:
                if review.get("independent_review", "").lower() == "true" and not review.get("adjudicator", ""):
                    independent[review["reviewer"]][key[0]] = review["status"]
        for first,second in combinations(sorted(independent),2):
            paired = sorted(set(independent[first]) & set(independent[second]))
            if len(paired) >= 2:
                value = kappa([independent[first][k] for k in paired],[independent[second][k] for k in paired])
                agreement.append({"language":language,"reviewers":[first,second],"paired_examples":len(paired),
                                  "cohens_kappa":value,"undefined_if_constant_expected_agreement":value is None})
    summary["cohens_kappa"] = agreement or None
    return summary


def write_reports(source_rows, candidates, reviews, config, directory, raw_manifest=None):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    report = statistics(source_rows,candidates,reviews,config)
    report["translation_configuration"] = config
    report["execution_metadata"] = None if raw_manifest is None else {key:raw_manifest.get(key) for key in (
        "status","model_id","model_revision","provider","backend","quantization","base_model_id","started_utc","updated_utc","runtime",
        "last_session_elapsed_seconds_monotonic","total_recorded_execution_seconds","raw_files_sha256",
        "successful_unique_rows","observed_unique_rows","api_token_consumption","api_cost")}
    automated, coverage = validate_candidates(candidates,source_rows)
    write_json(directory/"dataset_statistics.json",{**report,"coverage":coverage})
    write_json(directory/"translation_report.json",report)
    for filename,rows,fields in [
        ("automated_validation.csv",automated,["id","original_id","language","split","schema_valid","answer_span_valid","all_reference_spans_valid","language_id","issues"]),
        ("human_evaluation.csv",[{**r,"question_id":key[0],"language":key[1]} for key,values in reviews.items() for r in values],REVIEW_FIELDS),
        ("error_analysis.csv",[{"issue":k,"count":v,"origin":"automated_flag"} for k,v in report["error_counts"].items()]+
                              [{"issue":k,"count":v,"origin":"human_judgment"} for k,v in report["human_error_counts"].items()],
                              ["issue","count","origin"])]:
        with (directory/filename).open("w",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore",lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return report
