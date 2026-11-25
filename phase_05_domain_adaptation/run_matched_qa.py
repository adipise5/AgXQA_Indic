"""Fine-tune an Indic-AEC-adapted XLM-R encoder under the recorded Phase 3B QA protocol."""

import argparse
from hashlib import sha256 as make_digest
import json
from pathlib import Path
from types import SimpleNamespace

from phase_03_multilingual_finetuning import model_runs
from phase_03_multilingual_finetuning.common import load_data, read_json, read_settings, write_json
from phase_03_multilingual_finetuning.scoring import aggregate, compare_inputs
from shared.qa_release import sha256
from shared.validate import read_jsonl


PHASE = Path(__file__).resolve().parent
PHASE3 = PHASE.parent / "phase_03_multilingual_finetuning"
REFERENCE = PHASE3 / "results" / "phase_3b"
TRAINING_CONFIG = PHASE3 / "configs" / "cuda_hpc.yaml"
DEFAULT_AUDIT = PHASE / "results" / "phase3_span_checked_overlap.json"
EXPERIMENT = "phase_5_adapted"


def rebuild_qa_splits(source_dir, export_dir, control_dataset, audit):
    from phase_02_multilingual_benchmark.gemini_audit import ALL_LANGUAGES, validate_parallel
    from phase_02_multilingual_benchmark.gemini_exports import EXPORTS, import_rows, make_candidate
    from phase_02_multilingual_benchmark.pipeline import answer_issues, load_source

    for filename, _ in EXPORTS.values():
        if sha256(export_dir / filename) != audit["translated_export_sha256"][filename]:
            raise ValueError(f"Gemini export checksum mismatch: {filename}")
    sources, source_audit = load_source(source_dir)
    if source_audit["file_sha256"] != audit["source_sha256"]:
        raise ValueError("Official English files differ from the audited source")
    raw, cohort, _ = import_rows(export_dir, {row["id"]: row for row in sources})
    candidates = [
        make_candidate(source, source if lang == "en" else raw[f"{source['id']}_{lang}"], lang)[0]
        for source in cohort for lang in ALL_LANGUAGES
    ]
    validate_parallel(candidates, cohort)
    pending = {row["original_id"] for row in candidates if answer_issues(row)}
    rows = [row for row in candidates if row["original_id"] not in pending]
    validate_parallel(rows, [source for source in cohort if source["id"] not in pending])
    splits = {split: [row for row in rows if row["split"] == split]
              for split in control_dataset["split_sha256"]}
    for split, examples in splits.items():
        digest = make_digest()
        for row in examples:
            digest.update((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        if digest.hexdigest() != control_dataset["split_sha256"][split]:
            raise ValueError(f"Rebuilt {split} bytes differ from the exact Phase 3 split")
    return splits


def verified_inputs(args):
    settings = read_settings("evaluation_config.yaml")
    control = read_json(REFERENCE / "run_manifest.json")
    audit = read_json(args.audit)
    if args.export_dir is None:
        splits, identity = load_data("3b", args.bundle_dir, "span_checked")
        data_verification = "original_manifest_and_split_checksums"
    else:
        identity = control["dataset"]
        splits = rebuild_qa_splits(args.source_dir, args.export_dir, identity, audit)
        data_verification = "export_rebuild_matched_split_checksums"
    adaptation_dir = args.adaptation_dir
    if (control["experiment"] != "phase_3b" or control["seed"] != 42
            or identity != control["dataset"] or settings["metric_protocol"] != control["protocol"]
            or settings["tokenization"] != control["tokenization"]
            or model_runs.training_configuration(TRAINING_CONFIG) != control["checkpoint"]["training_configuration"]):
        raise ValueError("Phase 5 must match the recorded Phase 3B data, seed, training settings and protocol")
    predictions_path = REFERENCE / "predictions.jsonl"
    predictions_digest = sha256(predictions_path)
    if predictions_digest != control["output_file_sha256"]["predictions.jsonl"]:
        raise ValueError("Phase 3B predictions differ from their published checksum")
    control_predictions = read_jsonl(predictions_path)
    if len(control_predictions) != len(splits["test"]) or {r["id"] for r in control_predictions} != {
            r["id"] for r in splits["test"]}:
        raise ValueError("Phase 3B predictions do not match the pinned test row IDs")

    if (audit["status"] != "no_literal_overlap_flags" or audit["phase6_reserved_source_hits"]
            or audit["dataset"]["variant"] != "span_checked"
            or {split: data["sha256"] for split, data in audit["dataset"]["splits"].items()}
            != identity["split_sha256"]):
        raise ValueError("The overlap audit must cover the same verified QA split files")
    corpus = audit["corpus"]
    metrics = read_json(adaptation_dir / "metrics.json")
    weights = adaptation_dir / "final_encoder_layer.safetensors"
    digest = sha256(weights)
    if (metrics["status"] != "local_pretrained_mlm_pilot"
            or metrics["model_id"] != settings["base_model"]
            or metrics["model_revision"] != settings["base_revision"]
            or metrics["phase4_version"] != corpus["version"]
            or metrics["input_manifest_sha256"] != corpus["input_manifest_sha256"]
            or digest != metrics["checkpoint_sha256"] or digest != corpus["adapted_layer_sha256"]):
        raise ValueError("Adapted encoder weights do not match the audited Phase 5 pilot")
    initialization = {
        "method": "indic_aec_mlm_final_encoder_layer",
        "phase4_version": corpus["version"],
        "phase4_train_sha256": corpus["train_sha256"],
        "adaptation_sha256": digest,
        "mlm_metrics_sha256": sha256(adaptation_dir / "metrics.json"),
        "overlap_audit_sha256": sha256(args.audit),
        "qa_data_verification": data_verification,
    }
    return settings, splits, identity, control_predictions, predictions_digest, weights, initialization


def run(args):
    settings, splits, identity, control_predictions, control_hash, weights, initialization = verified_inputs(args)
    if args.check_only:
        print("Verified the Phase 3B settings, QA split checksums, baseline predictions, overlap audit and adapted layer.")
        return

    from safetensors.torch import load_file
    import torch

    model_runs.get_device(torch, "cuda")
    adapted_layer = load_file(str(weights), device="cpu")

    def initialize_encoder(model):
        model.roberta.encoder.layer[-1].load_state_dict(adapted_layer, strict=True)
        return initialization

    args.output.mkdir(parents=True, exist_ok=False)
    training = args.output / "training"
    training.mkdir()
    train_args = SimpleNamespace(
        training_config=TRAINING_CONFIG, pilot_steps=None, device="cuda", mps_memory_fraction=None,
        seed=42, offline=args.offline, output=training,
    )
    provenance = {
        "kind": "phase2_pinned", "bundle_manifest_sha256": identity["bundle_manifest_sha256"],
        "variant": identity["variant"],
        "split_sha256": {split: identity["split_sha256"][split] for split in ("train", "validation")},
    }
    model_runs.train_qa(train_args, settings, splits, provenance, EXPERIMENT, settings["languages"],
                        initialize_encoder=initialize_encoder)

    evaluation = args.output / "evaluation"
    evaluation.mkdir()
    eval_args = SimpleNamespace(
        checkpoint=training / "best_checkpoint", provenance=None, device="cuda", batch_size=8,
        save_window_logits=False, output=evaluation,
    )
    model_runs.evaluate_qa(eval_args, settings, splits["test"], identity, EXPERIMENT)
    adapted_predictions = read_jsonl(evaluation / "predictions.jsonl")
    control_groups, _ = compare_inputs(control_predictions, adapted_predictions)
    control, adapted = aggregate(control_predictions), aggregate(adapted_predictions)
    write_json(args.output / "comparison.json", {
        "protocol": settings["metric_protocol"], "dataset": identity, "seed": 42,
        "paired_test_groups": len(control_groups), "initialization": initialization,
        "phase_3b_predictions_sha256": control_hash,
        "phase_5_predictions_sha256": sha256(evaluation / "predictions.jsonl"),
        "phase_3b": control, "phase_5_adapted": adapted,
        "delta_percentage_points": {language: {metric: adapted[language][metric] - control[language][metric]
                                               for metric in ("exact_match", "f1")} for language in control},
    })
    print(f"Saved matched comparison to {args.output / 'comparison.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--bundle-dir", type=Path, help="Original pinned Gemini v2 bundle used in Phase 3")
    source.add_argument("--export-dir", type=Path, help="Directory with the three original Gemini flat exports")
    parser.add_argument("--source-dir", type=Path, help="Original English AgXQA JSONLs for --export-dir")
    parser.add_argument("--adaptation-dir", type=Path, required=True, help="Private saved Phase 5 MLM run")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="Use the already cached XLM-R base")
    parser.add_argument("--check-only", action="store_true", help="Verify pinned files without starting training")
    args = parser.parse_args()
    if (args.export_dir is None) != (args.source_dir is None):
        parser.error("--export-dir and --source-dir must be provided together")
    try:
        run(args)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f"Phase 5 matched QA: {error}\n")


if __name__ == "__main__":
    main()
