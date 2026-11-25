"""Phase 2 data preparation, validation, review and reporting commands."""
import argparse
import json
from pathlib import Path

from .pipeline import (PHASE, load_config, load_source, read_jsonl,
                       read_reviews, resolve_path, source_candidate, write_json, write_jsonl)
from .operations import (assemble, candidate_rows, clean_translations, create_review_worksheets,
                         validate_candidates, write_reports)


def run(command):
    parser = argparse.ArgumentParser(description=f"Phase 2: {command}")
    parser.add_argument("--config",type=Path,default=PHASE/"config.json")
    parser.add_argument("--source-dir",type=Path)
    parser.add_argument("--candidates-dir",type=Path)
    parser.add_argument("--raw-dir",type=Path)
    parser.add_argument("--review-dir",type=Path)
    parser.add_argument("--results-dir",type=Path,default=PHASE/"results")
    if command == "prepare_source":
        parser.add_argument("--download",action="store_true")
        parser.add_argument("--output-dir",type=Path,default=PHASE/"outputs/source")
    if command == "import_gemini_exports":
        parser.add_argument("--export-dir", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
    if command == "finalize_gemini_exports":
        parser.add_argument("--bundle-dir", type=Path, required=True)
    if command == "clean_translations":
        parser.add_argument("--output-dir",type=Path)
        parser.add_argument("--skip-language-id",action="store_true",help="Flags every candidate for human language verification")
    if command in ("validate_schema","validate_answer_spans","validate_language"):
        parser.add_argument("--input",type=Path)
    if command in ("merge_languages","build_dataset"):
        parser.add_argument("--output-dir",type=Path)
        parser.add_argument("--human-reviewed-output",type=Path)
    if command == "dataset_statistics":
        parser.add_argument("--plots",action="store_true")
        parser.add_argument("--dataset-statistics-output",type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    source_dir = args.source_dir or resolve_path(config["source_dir"])
    sources, audit = load_source(source_dir,download=command == "prepare_source" and args.download)
    if command == "import_gemini_exports":
        from .gemini_exports import prepare_bundle

        manifest = prepare_bundle(args.export_dir, args.output_dir, sources, audit, config)
        print(json.dumps({key: manifest[key] for key in ("status", "span_checked_rows", "unresolved_span_row_ids", "required_review_rows")}, indent=2))
        return
    if command == "finalize_gemini_exports":
        from .gemini_exports import finalize_bundle

        review_dir = args.review_dir or args.bundle_dir / "human_review"
        report = finalize_bundle(args.bundle_dir, review_dir, sources, audit, config)
        print(json.dumps({key: report[key] for key in ("status", "accepted_records", "human_reviewed_records", "submitted_review_judgments")}, indent=2))
        if report["status"] != "ready":
            parser.exit(2, "Final benchmark is blocked. See finalization_report.json in the bundle.\n")
        return
    raw_dir = args.raw_dir or resolve_path(config["raw_output_dir"])
    candidates_dir = args.candidates_dir or resolve_path(config["candidate_output_dir"])
    review_dir = args.review_dir or resolve_path(config["review_dir"])
    if command == "prepare_source":
        args.output_dir.mkdir(parents=True,exist_ok=True)
        write_jsonl(args.output_dir/"english_source.jsonl",sources)
        write_json(args.results_dir / "source_preflight.json", audit)
        print(json.dumps({"source_counts":audit["counts"],"unique_contexts":audit["unique_contexts"],
                          "invalid_source_offsets":len(audit["invalid_source_answer_offsets"]),"neural_execution":False}))
        return
    if command == "clean_translations":
        if not (raw_dir/"run_manifest.json").exists():
            raise ValueError("No recorded raw translation run found. Existing unverified legacy files must be imported with their actual provenance first.")
        count = clean_translations(sources,raw_dir,args.output_dir or candidates_dir,use_langid=not args.skip_language_id)
        print(json.dumps({"candidates":count,"human_reviewed":False,"raw_outputs_modified":False}))
        return
    candidates = candidate_rows(candidates_dir)
    if not candidates:
        candidates = [source_candidate(row) for row in sources]
    reviews = read_reviews(review_dir)
    raw_manifest = json.loads((raw_dir/"run_manifest.json").read_text()) if (raw_dir/"run_manifest.json").exists() else None
    if command in ("validate_schema","validate_answer_spans","validate_language"):
        if args.input:
            candidates = read_jsonl(args.input)
        reports,coverage = validate_candidates(candidates,sources)
        write_json(args.results_dir/(command+".json"),{"checks":reports,"coverage":coverage,
                   "language_check_note":"Stored language-ID and script assessments are heuristics; Hindi and Marathi share Devanagari."})
        print(json.dumps({"observed":len(candidates),"missing_pairs":coverage["missing_pair_count"],"valid_span_rows":sum(r["answer_span_valid"] for r in reports)}))
        if command == "validate_schema":
            failed = any(not r["schema_valid"] or any(code in r["issues"] for code in ("duplicate_original_id_language","source_identity_mismatch","unknown_id_or_split_changed")) for r in reports)
        elif command == "validate_answer_spans":
            failed = any(not r["all_reference_spans_valid"] for r in reports)
        else:
            failed = any(r["language"] != "en" and (r["language_id"] != r["language"] or any(code in r["issues"] for code in ("low_expected_script_ratio","unicode_corruption","long_untranslated_latin_passage"))) for r in reports)
        if failed or coverage["missing_pair_count"] or coverage["extra_pairs"]:
            parser.exit(2,"Validation incomplete/failed; inspect the saved per-record report and coverage.\n")
        return
    if command == "check_data_leakage":
        from .pipeline import leakage_report
        mismatches = [r["id"] for r in candidates if r["original_id"] not in {s["id"] for s in sources} or
                      next(s["split"] for s in sources if s["id"] == r["original_id"]) != r["split"]]
        report = {"official_source":leakage_report(sources),"candidate_id_split_mismatches":mismatches,
                  "official_splits_reshuffled":False,"note":"Existing official source-document overlap is retained and disclosed."}
        write_json(args.results_dir/"leakage_report.json",report)
        print(json.dumps(report,indent=2))
        if mismatches:
            raise ValueError("Candidate split membership differs from the official source")
        return
    if command == "create_review_worksheets":
        create_review_worksheets(sources,candidates,config,review_dir)
        print(json.dumps({"review_directory":str(review_dir),"filled_review_judgments":0,"note":"Worksheets are requests, not completed human reviews"}))
        return
    if command in ("merge_languages","build_dataset"):
        report = assemble(sources,candidates,reviews,config,args.output_dir or resolve_path(config["final_output_dir"]),
                          args.human_reviewed_output or resolve_path(config["human_reviewed_dir"]))
        write_json(args.results_dir/"build_report.json",report)
        print(json.dumps({key:value for key,value in report.items() if key not in ("blocked",)},indent=2))
        if report["status"] != "ready":
            parser.exit(2,"Build blocked: required translations, valid spans, or human judgments are missing. No final benchmark was emitted.\n")
        return
    if command == "dataset_statistics":
        report = write_reports(sources,candidates,reviews,config,args.results_dir,raw_manifest)
        if args.dataset_statistics_output:
            write_json(args.dataset_statistics_output,report)
        if args.plots:
            from .visualizations import generate_plots
            generate_plots(report,args.results_dir/"plots")
        print(json.dumps({"status":report["status"],"human_reviews":sum(v["review_judgments"] for v in report["human_quality"].values()),
                          "reported_ratings_are_measured":True,"unmeasured_values_are_null":True}))
        return
    raise ValueError("Unknown Phase 2 operation")
