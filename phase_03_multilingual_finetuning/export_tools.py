"""Copy completed Phase 3 run artifacts into the committed results/ layout. No neural execution."""

import argparse
from pathlib import Path
import shutil

from shared.qa_release import sha256
from .common import read_json, write_json
from .report_tools import match_runs

PHASE = Path(__file__).resolve().parent
RUN_FILES = ("metrics.json", "language_wise_results.csv", "predictions.jsonl",
             "computational_statistics.json", "run_manifest.json")
TRAINING_FILES = ("training_logs.csv", "validation_metrics.csv", "trainer_log_history.json")
COMPARISON_FILES = ("comparative_results.csv", "performance_gap.csv", "cross_lingual_consistency.csv",
                    "pairwise_agreement.json", "confidence_intervals.csv", "error_analysis.csv",
                    "matched_slices.csv", "computational_statistics.json", "comparison_manifest.json")


def export(args):
    a, b, _, _, _ = match_runs(args.a, args.b)
    comparison = read_json(args.comparison / "comparison_manifest.json")
    if comparison["dataset"] != a["dataset"] or comparison["seeds"] != {"phase_3a": a["seed"], "phase_3b": b["seed"]}:
        raise ValueError("The comparison was not produced from these two runs")
    for directory, run in ((args.training_dir, b), (args.english_training_dir, a)):
        if directory and read_json(directory / "best_checkpoint" / "phase3_checkpoint.json") != run["checkpoint"]:
            raise ValueError(f"{directory} did not train the evaluated checkpoint")

    copies = []
    for name in RUN_FILES:
        copies += [(args.a / name, f"phase_3a/{name}"), (args.b / name, f"phase_3b/{name}")]
    for name in TRAINING_FILES:
        copies.append((args.training_dir / name, f"phase_3b/{name}"))
        if args.english_training_dir:
            copies.append((args.english_training_dir / name, f"phase_3a/{name}"))
    if args.english_training_dir and (args.english_training_dir / "english_source_validation_metrics.json").exists():
        copies.append((args.english_training_dir / "english_source_validation_metrics.json",
                       "phase_3a/english_source_validation_metrics.json"))
    copies += [(args.comparison / name, name) for name in COMPARISON_FILES]
    copies += [(path, f"plots/{path.name}") for path in sorted((args.comparison / "plots").glob("*.png"))]
    if args.context_intervals:
        copies.append((args.context_intervals, "confidence_intervals_context.csv"))

    missing = [str(source) for source, _ in copies if not source.is_file()]
    if missing:
        raise ValueError("Missing run artifacts: " + ", ".join(missing))
    files = {}
    for source, target in copies:
        destination = args.output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[target] = {"source": source.as_posix(), "sha256": sha256(destination)}
    (args.output / "export_manifest.json").unlink(missing_ok=True)
    write_json(args.output / "export_manifest.json", {
        "dataset": a["dataset"], "protocol": a["protocol"], "seeds": comparison["seeds"],
        "provisional": not a["dataset"].get("reviewed_release", False),
        "devices": {"phase_3a": a["checkpoint"]["computational_statistics"].get("device_name"),
                    "phase_3b": b["checkpoint"]["computational_statistics"].get("device_name")},
        "files": files,
    })
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="Phase 3A evaluation directory")
    parser.add_argument("--b", type=Path, required=True, help="Phase 3B evaluation directory")
    parser.add_argument("--comparison", type=Path, required=True, help="generate_plots.py output directory")
    parser.add_argument("--training-dir", type=Path, required=True, help="Phase 3B training directory")
    parser.add_argument("--english-training-dir", type=Path, help="Phase 3A English training directory")
    parser.add_argument("--context-intervals", type=Path, help="Optional context-cluster bootstrap CSV")
    parser.add_argument("--output", type=Path, default=PHASE / "results")
    args = parser.parse_args()
    try:
        count = export(args)
        print(f"Exported {count} files to {args.output}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Phase 3 export: {error}\n")
