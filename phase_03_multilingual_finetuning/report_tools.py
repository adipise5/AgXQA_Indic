"""Generate reproducible Phase 3 reports from frozen prediction artifacts."""

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
from statistics import mean, stdev

from shared.metrics import best_scores
from shared.qa_release import VARIANTS, sha256
from shared.validate import read_jsonl
from .common import LANGUAGES, load_data, read_json, read_settings, write_csv, write_json, write_jsonl
from .features import decode_logits
from .scoring import aggregate, bootstrap, compare_inputs, consistency, review_sample


ERROR_FIELDS = ("id", "original_id", "language", "model", "question", "gold_answer", "predicted_answer",
                "em", "f1", "error_category", "reviewer", "comments")
INTERVAL_FIELDS = ("language", "metric", "delta_pp", "lower_pp", "upper_pp", "unit", "units",
                   "question_groups", "iterations", "confidence", "seed")


def load_run(directory, expected=None):
    manifest = read_json(directory / "run_manifest.json")
    if manifest["protocol"] != "agxqa-indic-v1":
        raise ValueError("Expected the common Phase 3 scoring protocol")
    path = directory / "predictions.jsonl"
    if sha256(path) != manifest["output_file_sha256"]["predictions.jsonl"]:
        raise ValueError("Frozen predictions differ from the run manifest")
    rows = read_jsonl(path)
    if not rows or (expected and manifest["experiment"] != expected):
        raise ValueError("Run has no predictions or belongs to the wrong experiment")
    if len(rows) != manifest["dataset"]["coverage"]["test"]["records"]:
        raise ValueError("The prediction count differs from the frozen test split")
    for row in rows:
        if row["experiment"] != manifest["experiment"] or row["seed"] != manifest["seed"]:
            raise ValueError("Prediction record differs from its run metadata")
        score = best_scores(row["predicted_answer"], row["gold_answers"])
        if score["em"] != row["exact_match"] or score["f1"] != row["f1"]:
            raise ValueError(f"Saved scores disagree with the common scorer for {row['id']}")
    return manifest, rows


def match_runs(dir_a, dir_b):
    a, left = load_run(dir_a, "phase_3a")
    b, right = load_run(dir_b, "phase_3b")
    if a["dataset"] != b["dataset"] or a["protocol"] != b["protocol"] or a["tokenization"] != b["tokenization"]:
        raise ValueError("Comparative results require identical dataset hashes, tokenizer settings, and scoring protocol")
    return a, b, left, right, compare_inputs(left, right)


def result_tables(left, right):
    a, b = aggregate(left), aggregate(right)
    rows = [{"language": lang, "count": a[lang]["count"], "em_3a": a[lang]["exact_match"], "f1_3a": a[lang]["f1"],
             "em_3b": b[lang]["exact_match"], "f1_3b": b[lang]["f1"], "delta_f1_pp": b[lang]["f1"] - a[lang]["f1"]}
            for lang in (*LANGUAGES, "indic_macro", "overall_macro")]
    gaps = []
    for lang in LANGUAGES[1:]:
        before, after = a["en"]["f1"] - a[lang]["f1"], b["en"]["f1"] - b[lang]["f1"]
        gaps.append({"language": lang, "gap_3a_pp": before, "gap_3b_pp": after, "gap_reduction_pp": before - after})
    return rows, gaps


def length_bucket(length, boundaries):
    for cutoff in boundaries:
        if length <= cutoff:
            return f"at_most_{cutoff}"
    return f"over_{boundaries[-1]}"


def slice_results(by_a, by_b):
    slices = defaultdict(list)
    for identifier, group in by_a.items():
        for language, row in group.items():
            other = by_b[identifier][language]
            dimensions = {
                "context_chars": length_bucket(row["context_chars"], (500, 1000, 2000)),
                "context_tokens": length_bucket(row["context_tokens"], (128, 256, 384)),
                "first_answer_chars": length_bucket(row["answer_chars"], (16, 32, 64)),
                "category": row["category"] or "unlabeled",
                "human_reviewed": str(row["human_reviewed"]).lower(),
            }
            for name, value in dimensions.items():
                slices[(language, name, value)].append((row["f1"], other["f1"]))
    return [{"language": lang, "dimension": name, "value": value, "count": len(scores),
             "f1_3a": 100 * mean(a for a, _ in scores), "f1_3b": 100 * mean(b for _, b in scores)}
            for (lang, name, value), scores in sorted(slices.items())]


def review_categories(path, sample):
    expected = {(row["id"], row["model"]) for row in sample}
    seen = set()
    counted = Counter()
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if not row["error_category"].strip():
                continue
            key = row["id"], row["model"]
            if key not in expected or key in seen or not row["reviewer"].strip():
                raise ValueError("An error judgment is unmatched, duplicated, or missing a reviewer")
            seen.add(key)
            counted[row["error_category"].strip()] += 1
    return counted


def summarize_seeds(directories):
    runs = [load_run(directory) for directory in directories]
    first = runs[0][0]
    seeds = [manifest["seed"] for manifest, _ in runs]
    if len(seeds) != len(set(seeds)) or any(
        manifest[key] != first[key] for manifest, _ in runs
        for key in ("experiment", "dataset", "protocol", "tokenization")
    ):
        raise ValueError("Seed summaries require distinct seeds for the same experiment and dataset")
    scores = [aggregate(rows) for _, rows in runs]
    for _, rows in runs[1:]:
        compare_inputs(runs[0][1], rows)
    by_language = {}
    for language in scores[0]:
        by_language[language] = {}
        for metric in ("exact_match", "f1"):
            items = [score[language][metric] for score in scores]
            by_language[language][metric] = {"mean": mean(items), "sample_sd": stdev(items) if len(items) > 1 else None}
    return {"experiment": first["experiment"], "seeds": seeds, "by_language": by_language}


def postprocess(args):
    splits, identity = load_data("3b", args.bundle_dir, args.variant)
    manifest = read_json(args.run / "run_manifest.json")
    logits = args.run / "window_logits.jsonl"
    if identity != manifest["dataset"] or sha256(logits) != manifest["output_file_sha256"]["window_logits.jsonl"]:
        raise ValueError("Saved logits or the selected dataset differ from the run manifest")
    entries = read_jsonl(logits)
    decoded = decode_logits(splits["test"], entries, [row["start_logits"] for row in entries],
                            [row["end_logits"] for row in entries], manifest["tokenization"])
    write_jsonl(args.output, ({"id": row["id"], **decoded[row["id"]]} for row in splits["test"]))


def main(operation):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    if operation == "seeds":
        parser.add_argument("--runs", type=Path, nargs="+", required=True)
    elif operation in ("metrics", "consistency", "postprocess"):
        parser.add_argument("--run", type=Path, required=True)
    else:
        parser.add_argument("--a", type=Path, required=True)
        parser.add_argument("--b", type=Path, required=True)
        parser.add_argument("--bootstrap-unit", choices=("question", "context", "document"), default="question")
    if operation == "postprocess":
        parser.add_argument("--bundle-dir", type=Path)
        parser.add_argument("--variant", choices=VARIANTS, default="final")
    if operation == "reports":
        parser.add_argument("--training-dir", type=Path)
        parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    try:
        if operation == "postprocess":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            postprocess(args)
        elif operation in ("metrics", "consistency"):
            _, rows = load_run(args.run)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, aggregate(rows) if operation == "metrics" else consistency(rows))
        elif operation == "seeds":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, summarize_seeds(args.runs))
        else:
            a, b, left, right, groups = match_runs(args.a, args.b)
            settings = read_settings("evaluation_config.yaml")
            if operation != "reports":
                args.output.parent.mkdir(parents=True, exist_ok=True)
            if operation == "statistics":
                intervals = bootstrap(*groups, unit=args.bootstrap_unit, **settings["bootstrap"])
                write_csv(args.output, INTERVAL_FIELDS, intervals)
            elif operation == "errors":
                sample = review_sample(*groups, settings["error_sample_per_indic_language"], settings["bootstrap"]["seed"])
                write_csv(args.output, ERROR_FIELDS, sample)
            else:
                from .visualizations import generate_plots

                args.output.mkdir(parents=True, exist_ok=False)
                rows, gaps = result_tables(left, right)
                joint = {"phase_3a": consistency(left), "phase_3b": consistency(right)}
                intervals = bootstrap(*groups, unit=args.bootstrap_unit, **settings["bootstrap"])
                sample = review_sample(*groups, settings["error_sample_per_indic_language"], settings["bootstrap"]["seed"])
                categories = review_categories(args.annotations, sample) if args.annotations else {}
                write_csv(args.output / "comparative_results.csv", tuple(rows[0]), rows)
                write_csv(args.output / "performance_gap.csv", tuple(gaps[0]), gaps)
                write_csv(args.output / "confidence_intervals.csv", INTERVAL_FIELDS, intervals)
                write_csv(args.output / "error_analysis.csv", ERROR_FIELDS, sample)
                write_csv(args.output / "cross_lingual_consistency.csv",
                          ("experiment", "correct_languages", "count", "percent"),
                          [{"experiment": name, **entry} for name, stats in joint.items()
                           for entry in stats["correct_in_n_languages"]])
                write_json(args.output / "pairwise_agreement.json", joint)
                slices = slice_results(*groups)
                write_csv(args.output / "matched_slices.csv", tuple(slices[0]), slices)
                computation = {"phase_3a": read_json(args.a / "computational_statistics.json"),
                               "phase_3b": read_json(args.b / "computational_statistics.json")}
                write_json(args.output / "computational_statistics.json", computation)
                history = []
                if args.training_dir:
                    saved = read_json(args.training_dir / "best_checkpoint" / "phase3_checkpoint.json")
                    if saved != b["checkpoint"]:
                        raise ValueError("Training logs belong to a different checkpoint")
                    history = read_json(args.training_dir / "trainer_log_history.json")
                plots = generate_plots(args.output / "plots", rows, gaps, joint, history, categories)
                write_json(args.output / "comparison_manifest.json", {
                    "dataset": a["dataset"], "protocol": a["protocol"], "seeds": {"phase_3a": a["seed"], "phase_3b": b["seed"]},
                    "bootstrap_unit": args.bootstrap_unit, "plots": plots, "reviewed_error_categories": dict(categories),
                    "training_configuration": {"phase_3a": a["checkpoint"]["training_configuration"],
                                               "phase_3b": b["checkpoint"]["training_configuration"]},
                })
        print(f"Wrote {args.output}")
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f"Phase 3 {operation}: {error}\n")
