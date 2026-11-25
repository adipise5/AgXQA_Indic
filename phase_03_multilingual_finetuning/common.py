"""Inputs and artifacts shared across Phase 3 commands."""
import csv
import json
from pathlib import Path

from shared.qa_release import load_qa_release, sha256


PHASE = Path(__file__).resolve().parent
LANGUAGES = ("en", "hi", "mr", "bn", "ta")
EXPERIMENTS = {"3a": PHASE / "experiment_3A_zero_shot", "3b": PHASE / "experiment_3B_multilingual"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_settings(filename):
    import yaml

    return yaml.safe_load((PHASE / "configs" / filename).read_text(encoding="utf-8"))


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_data(experiment, bundle_dir, variant):
    return load_qa_release(read_json(EXPERIMENTS[experiment] / "config.json"), bundle_dir, variant)


def file_manifest(directory):
    return {str(path.relative_to(directory)): sha256(path) for path in sorted(directory.rglob("*")) if path.is_file()}
