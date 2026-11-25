"""Model-free dataset verification, preparation, and tokenizer audits."""

import argparse
from collections import Counter
from pathlib import Path

from shared.qa_release import VARIANTS, sha256
from .common import LANGUAGES, file_manifest, load_data, read_settings, write_json, write_jsonl
from .features import prepare_features, summarize_tokenization


def dataset_report(splits, identity):
    train = {language: set() for language in LANGUAGES}
    for row in splits["train"]:
        train[row["language"]].add(row["context"])
    return {
        "dataset": identity,
        "counts_by_language": {split: dict(Counter(row["language"] for row in rows)) for split, rows in splits.items()},
        "test_rows_with_exact_train_context": {
            lang: sum(row["language"] == lang and row["context"] in train[lang] for row in splits["test"])
            for lang in LANGUAGES
        },
        "no_cross_split_question_ids": True,
        "all_stored_spans_exact": True,
        "notes": "A successful integrity audit may still describe a provisional release with inherited passage overlap.",
    }


def main(operation):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="span_checked")
    parser.add_argument("--output", type=Path, required=True)
    if operation == "tokenize":
        parser.add_argument("--tokenizer-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        splits, identity = load_data("3b", args.bundle_dir, args.variant)
        report = dataset_report(splits, identity)
        if operation == "prepare":
            args.output.mkdir(parents=True, exist_ok=False)
            for split, rows in splits.items():
                write_jsonl(args.output / f"{split}.jsonl", rows)
            report["prepared_file_sha256"] = file_manifest(args.output)
            write_json(args.output / "dataset_integrity_report.json", report)
        elif operation == "tokenize":
            from transformers import AutoTokenizer

            settings = read_settings("evaluation_config.yaml")
            observed = {name: sha256(args.tokenizer_dir / name) for name in settings["tokenizer_file_sha256"]}
            if observed != settings["tokenizer_file_sha256"]:
                raise ValueError("Tokenizer files differ from the pinned base revision")
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True, local_files_only=True)
            report["tokenization"] = settings["tokenization"]
            report["tokenizer_files"] = observed
            report["tokenizer_revision"] = settings["base_revision"]
            report["tokenizer_class"] = type(tokenizer).__name__
            report["by_split_and_language"] = {}
            for split, rows in splits.items():
                _, statistics = prepare_features(rows, tokenizer, settings["tokenization"])
                report["by_split_and_language"][split] = summarize_tokenization(
                    statistics, settings["tokenization"]["max_answer_length"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, report)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, report)
        print(f"Wrote {args.output}")
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f"Phase 3 {operation}: {error}\n")
