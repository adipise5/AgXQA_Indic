"""Verify the pinned inputs for Phase 3A or Phase 3B."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.qa_release import VARIANTS, load_qa_release

EXPERIMENTS = {"3a": "experiment_3A_zero_shot", "3b": "experiment_3B_multilingual"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="3b")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config or Path(__file__).parent / EXPERIMENTS[args.experiment] / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        splits, identity = load_qa_release(config, args.bundle_dir, args.variant)
        report = {
            "experiment": args.experiment, "dataset": identity,
            "selected_rows": {
                split: dict(Counter(row["language"] for row in rows if row["language"] in config[selection]))
                for split, rows, selection in (
                    ("train", splits["train"], "train_languages"),
                    ("validation", splits["validation"], "train_languages"),
                    ("test", splits["test"], "eval_languages"),
                )
            },
            "checkpoint_requirement": config["checkpoint_requirement"],
            "experiment_executed": False,
        }
        if args.experiment == "3a" and identity["variant"] == "document_disjoint":
            report["checkpoint_requirement"] = "English XLM-R trained only on the document-disjoint subset"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Phase 3 preflight: {error}\n")


if __name__ == "__main__":
    main()
