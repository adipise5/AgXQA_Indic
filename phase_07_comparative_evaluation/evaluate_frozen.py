"""Frozen-checkpoint extractive QA inference on the Native IndicAgriQA test file (Phase 7).

The dataset, its overlap audit and every checkpoint file are hash-verified before inference, and any mismatch stops
the run. Decoding uses the pinned Phase 3 settings. predictions.jsonl follows the shared prediction contract
({"id", "language", "prediction_text"}); offsets and tokenization diagnostics go to predictions_detail.jsonl.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

from phase_03_multilingual_finetuning.common import file_manifest, read_json, write_json, write_jsonl
from phase_03_multilingual_finetuning.features import decode_logits, prepare_features
from phase_03_multilingual_finetuning.model_runs import PROVENANCE
from shared.qa_release import sha256
from shared.validate import read_jsonl, validate


PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parent
CHECKPOINTS = PHASE / "configs" / "checkpoints.json"
DECODING = {"max_seq_length": 384, "doc_stride": 128, "max_answer_length": 64, "n_best_size": 20}
NATIVE_LANGUAGES = ("hi", "mr", "bn", "ta")
PROVENANCE_FIELDS = ("seed", "base_model", "base_revision", "training_data")


# ---------------------------------------------------------------- dataset

def load_native(manifest_path):
    """Return the Step 1 manifest, the verified Native rows and a text-free dataset record."""
    manifest = read_json(manifest_path)
    step1 = manifest_path.parent
    dataset = manifest["native_evaluation"]
    frozen = step1 / dataset["frozen_filename"]
    digest = sha256(frozen)
    if digest != dataset["frozen_sha256"]:
        raise ValueError(f"Native dataset hash mismatch: {frozen}")
    rows = read_jsonl(frozen)
    validate(rows)
    languages = Counter(row["language"] for row in rows)
    if len(rows) != dataset["records"] or {row["split"] for row in rows} != {"test"} or set(languages) != set(
            NATIVE_LANGUAGES):
        raise ValueError("The Native file does not match the manifest's record count, test split and four languages")
    audit_path = step1 / manifest["overlap_audit_filename"]
    if read_json(audit_path)["requires_overlap_review"]:
        raise ValueError("The Step 1 overlap audit has unresolved flags")
    record = {
        "frozen_filename": dataset["frozen_filename"], "sha256": digest, "records": len(rows),
        "languages": {language: languages[language] for language in NATIVE_LANGUAGES},
        "source_documents": len({row["source_document_id"] for row in rows}),
        "passages": len({row["context"] for row in rows}),
        "human_reviewed_records": sum(1 for row in rows if (row.get("metadata") or {}).get("human_validated") is True),
        "manifest_sha256": sha256(manifest_path), "overlap_audit_sha256": sha256(audit_path),
    }
    return manifest, rows, record


def model_rows(rows):
    """Convert shared-contract rows to the Phase 3 feature-builder layout (parallel answer arrays)."""
    converted = []
    for row in rows:
        answers = row["answers"]
        converted.append({
            "id": row["id"], "language": row["language"], "question": row["question"], "context": row["context"],
            "source_document_id": row["source_document_id"],
            "answers": {"text": [answer["text"] for answer in answers],
                        "answer_start": [answer["answer_start"] for answer in answers]},
        })
    return converted


def smoke_rows(rows):
    """One fixed question per Native language: the lowest ID in each language."""
    return [min((row for row in rows if row["language"] == language), key=lambda row: row["id"])
            for language in NATIVE_LANGUAGES]


# ---------------------------------------------------------------- checkpoints

def checkpoint_spec(manifest, model):
    pinned = read_json(CHECKPOINTS)["models"]
    if model not in pinned:
        raise ValueError(f"Unknown model {model}; expected one of {', '.join(pinned)}")
    spec = dict(pinned[model], model=model)
    if spec["role"] == "primary":
        expected = manifest["checkpoints"].get(model)
        if expected is None:
            raise ValueError(f"The Step 1 manifest has no checkpoint entry for {model}")
        recorded = read_json(ROOT / spec["recorded_provenance"])["checkpoint"]
        if (any(expected[field] != recorded[field] for field in PROVENANCE_FIELDS)
                or expected["selected_epoch_checkpoint"] != recorded["computational_statistics"]["selected_epoch_checkpoint"]
                or expected["file_sha256"] != recorded["checkpoint_file_sha256"]):
            raise ValueError(f"The Step 1 manifest entry for {model} disagrees with the recorded Phase 3 provenance")
        spec.update(private_server_path=expected["private_server_path"], recorded=recorded,
                    selected_epoch_checkpoint=expected["selected_epoch_checkpoint"], file_sha256=expected["file_sha256"])
    return spec


def verify_checkpoint(spec, directory):
    """Fail unless the directory holds exactly the pinned files and the pinned provenance."""
    provenance_path = directory / PROVENANCE
    if not provenance_path.is_file():
        raise ValueError(f"Missing checkpoint provenance: {provenance_path}")
    provenance = read_json(provenance_path)
    if spec["role"] == "primary":
        if provenance != spec["recorded"]:
            raise ValueError(f"{spec['model']}: provenance differs from the recorded Phase 3 run manifest")
    elif (sha256(provenance_path) != spec["provenance_sha256"]
          or any(provenance[field] != spec[field] for field in PROVENANCE_FIELDS)):
        raise ValueError(f"{spec['model']}: provenance differs from the pinned Phase 5 checkpoint")
    if provenance["computational_statistics"]["selected_epoch_checkpoint"] != spec["selected_epoch_checkpoint"]:
        raise ValueError(f"{spec['model']}: wrong selected checkpoint")
    files = file_manifest(directory)
    files.pop(PROVENANCE)
    if provenance["checkpoint_file_sha256"] != spec["file_sha256"] or files != spec["file_sha256"]:
        changed = sorted(set(files) ^ set(spec["file_sha256"]) | {
            name for name in set(files) & set(spec["file_sha256"]) if files[name] != spec["file_sha256"][name]})
        raise ValueError(f"{spec['model']}: missing, extra or changed checkpoint files: {', '.join(changed) or 'provenance'}")
    return provenance


# ---------------------------------------------------------------- inference

def build_predictions(rows, decoded, statistics):
    """Shared-contract predictions plus private diagnostics, checking coverage and character offsets."""
    if set(decoded) != {row["id"] for row in rows} or len(decoded) != len(rows):
        raise ValueError("Missing or extra predictions")
    predictions, details = [], []
    for row in rows:
        span = decoded[row["id"]]
        start, end, text = span["predicted_start"], span["predicted_end"], span["predicted_answer"]
        if not 0 <= start < end <= len(row["context"]) or row["context"][start:end] != text:
            raise ValueError(f"Prediction offsets do not match the passage: {row['id']}")
        stat = statistics[row["id"]]
        predictions.append({"id": row["id"], "language": row["language"], "prediction_text": text})
        details.append({
            "id": row["id"], "language": row["language"], "source_document_id": row["source_document_id"],
            "predicted_start": start, "predicted_end": end, "span_score": span["span_score"],
            "windows": stat["windows"], "positive_windows": stat["positive_windows"],
            "context_tokens": stat["context_tokens"], "gold_token_length": stat["best_gold_token_length"],
            "gold_exceeds_decode_limit": (stat["best_gold_token_length"] or 0) > DECODING["max_answer_length"],
        })
    return predictions, details


def infer(rows, directory, device_name, batch_size):
    import torch
    from transformers import DataCollatorWithPadding
    from phase_03_multilingual_finetuning.model_runs import get_device, load_model_and_tokenizer, weights_metadata

    device = get_device(torch, device_name)
    model, tokenizer = load_model_and_tokenizer(str(directory), None, True, trained=True)
    model.to(device).eval()
    features, statistics = prepare_features(rows, tokenizer, DECODING)
    if sum(stat["windows"] for stat in statistics.values()) != len(features) or any(
            stat["windows"] < 1 for stat in statistics.values()):
        raise ValueError("Incomplete tokenization windows")
    batches = torch.utils.data.DataLoader([feature["inputs"] for feature in features], batch_size=batch_size,
                                          shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    starts, ends = [], []
    with torch.inference_mode():
        for batch in batches:
            output = model(**{key: tensor.to(device) for key, tensor in batch.items()})
            starts.extend(output.start_logits.float().cpu().tolist())
            ends.extend(output.end_logits.float().cpu().tolist())
    elapsed = perf_counter() - started
    if len(starts) != len(features):
        raise ValueError("Incomplete window logits")
    decoded = decode_logits(rows, features, starts, ends, DECODING)
    computation = {**weights_metadata(model, device), "windows": len(features), "inference_seconds": elapsed,
                   "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    return decoded, statistics, computation


def source_revision():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True)
        return {"commit": head.stdout.strip(), "tracked_changes": bool(dirty.stdout.strip()), "source": "git"}
    except (OSError, subprocess.CalledProcessError):
        marker = ROOT / ".source_commit"
        return {**json.loads(marker.read_text(encoding="utf-8")), "source": ".source_commit"} if marker.is_file() else None


def run(args):
    manifest, rows, dataset = load_native(args.manifest)
    spec = checkpoint_spec(manifest, args.model)
    directory = args.checkpoint_dir or Path(spec["private_server_path"])
    provenance = verify_checkpoint(spec, directory)
    print(f"{args.model}: {spec['selected_epoch_checkpoint']}, all checkpoint files verified "
          f"(model {spec['file_sha256']['model.safetensors'][:12]})")
    print(f"Verified the frozen Native dataset: {dataset['records']} rows, {dataset['source_documents']} documents")
    if args.verify_only:
        return
    selected = smoke_rows(rows) if args.smoke else rows
    inputs = model_rows(selected)
    args.output.mkdir(parents=True, exist_ok=False)
    decoded, statistics, computation = infer(inputs, directory, args.device, args.batch_size)
    predictions, details = build_predictions(inputs, decoded, statistics)
    write_jsonl(args.output / "predictions.jsonl", predictions)
    write_jsonl(args.output / "predictions_detail.jsonl", details)
    import torch
    import transformers
    write_json(args.output / "run_manifest.json", {
        "phase": 7, "smoke": args.smoke, "model": args.model, "role": spec["role"], "label": spec["label"],
        "dataset": dataset, "evaluated_ids": len(predictions),
        "evaluated_languages": dict(Counter(row["language"] for row in predictions)),
        "checkpoint": {"directory": directory.as_posix(), "selected_epoch_checkpoint": spec["selected_epoch_checkpoint"],
                       "model_sha256": spec["file_sha256"]["model.safetensors"], "file_sha256": spec["file_sha256"],
                       "provenance_sha256": sha256(directory / PROVENANCE), "seed": provenance["seed"],
                       "experiment": provenance["experiment"]},
        "decoding": DECODING, "batch_size": args.batch_size, "device": args.device,
        "source": source_revision(), "command": [Path(sys.executable).name, "-m",
                                                  "phase_07_comparative_evaluation.evaluate_frozen", *sys.argv[1:]],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "computation": computation, "long_gold_spans": sum(detail["gold_exceeds_decode_limit"] for detail in details),
        "prediction_count": len(predictions), "predictions_sha256": sha256(args.output / "predictions.jsonl"),
        "predictions_detail_sha256": sha256(args.output / "predictions_detail.jsonl"),
    })
    print(f"Wrote {len(predictions)} {'smoke ' if args.smoke else ''}predictions to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Step 1 manifest naming the data and checkpoints")
    parser.add_argument("--model", required=True, help="phase_3a, phase_3b, phase_5_alpha_0 or phase_5_alpha_1")
    parser.add_argument("--checkpoint-dir", type=Path, help="Actual checkpoint directory, verified against the same hashes")
    parser.add_argument("--output", type=Path, help="Fresh run directory for this model")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="One fixed question per Native language")
    parser.add_argument("--verify-only", action="store_true", help="Verify the dataset and checkpoint, then stop")
    args = parser.parse_args()
    if not args.verify_only and args.output is None:
        parser.error("--output is required unless --verify-only is set")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        run(args)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f"Phase 7 evaluate_frozen: {error}\n")


if __name__ == "__main__":
    main()
