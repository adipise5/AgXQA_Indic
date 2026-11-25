"""Validation-selected adaptation-strength sweep for the Phase 5 final encoder layer.

Every arm uses the recorded Phase 3B data, tokenizer, training configuration, checkpoint selection rule
and scorer. Only the initial final-encoder layer changes: base + alpha * (adapted - base). Training and
validation never receive the test split. Test evaluation is limited to the alpha-0 control and the
candidate frozen in selection.json.
"""

import argparse
import csv
import os
from pathlib import Path
import shutil
from statistics import mean
from types import SimpleNamespace

from phase_03_multilingual_finetuning import model_runs
from phase_03_multilingual_finetuning.common import LANGUAGES, read_json, read_settings, write_csv, write_json
from phase_03_multilingual_finetuning.features import decode_logits, prepare_features
from phase_03_multilingual_finetuning.scoring import aggregate, bootstrap, compare_inputs, scored_predictions
from shared.qa_release import sha256
from shared.validate import read_jsonl
from .run_matched_qa import (DEFAULT_AUDIT, REFERENCE, TRAINING_CONFIG, adapted_initializer, experiment_name,
                             interpolate_layer, tensor_digest, verified_inputs)


CONTROL_ALPHA = 0.0
MIN_FREE_BYTES = 15 * 2**30
POPULATIONS = LANGUAGES + ("indic_macro", "overall_macro")
SELECTION_RULE = ("An adapted candidate (alpha > 0) at the control seed qualifies only if, on validation, its English F1 "
                  ">= the matched alpha-0 control's AND its Indic macro F1 > the control's. Among qualifying "
                  "candidates pick the highest Indic macro validation F1; ties go to the smaller alpha. If none "
                  "qualifies, keep the alpha-0 control and skip the extra adaptation runs.")


# ---------------------------------------------------------------- run records

def checkpoint_record(run):
    return read_json(run / "training" / "best_checkpoint" / model_runs.PROVENANCE)


def run_alpha(record):
    initialization = record.get("adapted_initialization")
    if initialization is None:
        raise ValueError("The run has no recorded Phase 5 initialization")
    if "alpha" in initialization:
        return float(initialization["alpha"])
    if initialization.get("method") == "indic_aec_mlm_final_encoder_layer":
        return 1.0  # Recorded before alpha existed: the adapted tensors were loaded unchanged.
    raise ValueError("Unknown Phase 5 initialization")


def split_dir(run, split):
    if split == "validation":
        return run / "validation"
    return run / "test" if (run / "test").exists() or not (run / "evaluation").exists() else run / "evaluation"


def run_summary(run, split):
    record = checkpoint_record(run)
    directory = split_dir(run, split)
    manifest = read_json(directory / "run_manifest.json")
    if manifest["checkpoint"] != record:
        raise ValueError(f"{directory} did not evaluate the selected checkpoint of {run}")
    if sha256(directory / "predictions.jsonl") != manifest["output_file_sha256"]["predictions.jsonl"]:
        raise ValueError(f"{directory} predictions differ from their manifest")
    statistics = record["computational_statistics"]
    return {
        "run": run.as_posix(), "split_dir": directory.as_posix(), "alpha": run_alpha(record), "seed": record["seed"],
        "experiment": record["experiment"], "selected_checkpoint": statistics["selected_epoch_checkpoint"],
        "checkpoint_model_sha256": record["checkpoint_file_sha256"]["model.safetensors"],
        "initial_layer_digest": record["adapted_initialization"].get("initial_layer_digest"),
        "predictions_sha256": manifest["output_file_sha256"]["predictions.jsonl"],
        "metrics": read_json(directory / "metrics.json")["by_language_and_macro"],
        "dataset": manifest["dataset"], "training_configuration": record["training_configuration"],
        "runtime": {key: statistics.get(key) for key in ("device_name", "platform", "versions")},
    }


def trainer_validation(run, summary):
    """Per-language validation scores the Trainer logged for the selected epoch."""
    step = int(summary["selected_checkpoint"].removeprefix("checkpoint-"))
    with (run / "training" / "validation_metrics.csv").open(encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["step"]) == step]
    if {row["language"] for row in rows} != set(LANGUAGES):
        raise ValueError(f"{run} has no complete Trainer validation record for {summary['selected_checkpoint']}")
    return {row["language"]: {"exact_match": float(row["em"]), "f1": float(row["f1"])} for row in rows}


def compact(summary):
    metrics = summary["metrics"]
    return {key: summary[key] for key in ("run", "alpha", "seed", "selected_checkpoint", "checkpoint_model_sha256",
                                          "initial_layer_digest", "predictions_sha256")} | {
        "scores": {name: {metric: metrics[name][metric] for metric in ("exact_match", "f1")} for name in POPULATIONS}}


def qualifies(control, candidate):
    return (candidate["metrics"]["en"]["f1"] >= control["metrics"]["en"]["f1"]
            and candidate["metrics"]["indic_macro"]["f1"] > control["metrics"]["indic_macro"]["f1"])


def choose(control, candidates):
    """Apply SELECTION_RULE to validation summaries. Returns the selected candidate or None."""
    eligible = [candidate for candidate in candidates if qualifies(control, candidate)]
    return max(eligible, key=lambda c: (c["metrics"]["indic_macro"]["f1"], -c["alpha"]), default=None)


def check_matched(summaries):
    first = summaries[0]
    for summary in summaries[1:]:
        if summary["dataset"] != first["dataset"] or summary["training_configuration"] != first["training_configuration"]:
            raise ValueError(f"{summary['run']} does not share the control's data and training configuration")


# ---------------------------------------------------------------- neural steps

def gpu_record(torch):
    properties = torch.cuda.get_device_properties(0)
    return {"name": properties.name, "uuid": str(getattr(properties, "uuid", "")),
            "total_memory_bytes": properties.total_memory, "torch_cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def evaluate_split(run, split, settings, splits, identity, save_logits=False):
    record = checkpoint_record(run)
    output = run / split
    output.mkdir(exist_ok=False)
    eval_args = SimpleNamespace(checkpoint=run / "training" / "best_checkpoint", provenance=None, device="cuda",
                                batch_size=8, save_window_logits=save_logits, output=output)
    model_runs.evaluate_qa(eval_args, settings, splits[split], identity, record["experiment"])
    print(f"Wrote {output}", flush=True)


def check_initialization(settings, adapted, offline, seed, alphas=(0.0, 0.5, 1.0)):
    """Build the seeded QA model per alpha and confirm only the final layer moves, by exactly the formula."""
    import torch
    from transformers import set_seed

    reference, results, tokenizer = None, {}, None
    for alpha in alphas:
        set_seed(seed)
        model, tokenizer = model_runs.load_model_and_tokenizer(
            settings["base_model"], settings["base_revision"], offline, trained=False)
        prefix = f"roberta.encoder.layer.{len(model.roberta.encoder.layer) - 1}."
        before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        record = adapted_initializer(adapted, alpha, {})(model)
        after = model.state_dict()
        base = {name[len(prefix):]: tensor for name, tensor in before.items() if name.startswith(prefix)}
        others = {name: tensor for name, tensor in before.items() if not name.startswith(prefix)}
        reference = reference or others
        expected = interpolate_layer(base, adapted, alpha)
        checks = {
            "layer_matches_formula": all(torch.equal(after[prefix + name], tensor) for name, tensor in expected.items()),
            "other_tensors_unchanged": all(torch.equal(after[name], tensor) for name, tensor in others.items()),
            "qa_head_and_other_layers_match_alpha_0": all(torch.equal(reference[name], tensor)
                                                          for name, tensor in others.items()),
        }
        if alpha == 0:
            checks["endpoint_is_base_exactly"] = all(torch.equal(after[prefix + n], t) for n, t in base.items())
        if alpha == 1:
            checks["endpoint_is_adapted_exactly"] = all(torch.equal(after[prefix + n], t) for n, t in adapted.items())
        if not all(checks.values()):
            raise ValueError(f"Initialization check failed for alpha {alpha:g}: {checks}")
        results[f"{alpha:g}"] = {**checks, "initial_layer_digest": record["initial_layer_digest"],
                                 "qa_head_digest": tensor_digest({n: t for n, t in after.items()
                                                                  if n.startswith("qa_outputs.")}),
                                 "max_abs_change_from_base": max(float((after[prefix + n] - t).abs().max())
                                                                 for n, t in base.items())}
        del model, before, after
    return results, tokenizer


def gold_over_cap(rows, statistics, cap):
    return {language: sum(1 for row in rows if row["language"] == language
                          and (statistics[row["id"]]["best_gold_token_length"] or 0) > cap) for language in LANGUAGES}


def cmd_check(args):
    settings, splits, identity, _, _, weights, initialization = verified_inputs(args)
    import torch
    from safetensors.torch import load_file

    report = {"verified": "Phase 3B settings, QA split checksums, baseline predictions, overlap audit, adapted layer",
              "dataset": identity, "adaptation_sha256": initialization["adaptation_sha256"],
              "training_configuration_file": TRAINING_CONFIG.relative_to(TRAINING_CONFIG.parents[2]).as_posix(),
              "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if args.require_cuda:
        model_runs.get_device(torch, "cuda")
        report["gpu"] = gpu_record(torch)
    target = (args.record or Path.cwd()).resolve()
    while not target.exists():
        target = target.parent
    report["free_bytes"] = shutil.disk_usage(target).free
    if args.require_cuda and report["free_bytes"] < MIN_FREE_BYTES:
        raise ValueError(f"Only {report['free_bytes'] / 2**30:.1f} GiB free; budget {MIN_FREE_BYTES / 2**30:.0f} GiB")
    report["initialization_check"], tokenizer = check_initialization(
        settings, load_file(str(weights), device="cpu"), args.offline, seed=42)
    cap = settings["tokenization"]["max_answer_length"]
    _, statistics = prepare_features(splits["validation"], tokenizer, settings["tokenization"])
    lengths = [record["best_gold_token_length"] or 0 for record in statistics.values()]
    report["validation_span_cap"] = {
        "current_cap": cap, "rows": len(lengths), "max_gold_token_length": max(lengths),
        "gold_exceeding_current_cap": gold_over_cap(splits["validation"], statistics, cap),
        f"gold_exceeding_{args.larger_cap}": gold_over_cap(splits["validation"], statistics, args.larger_cap),
    }
    if args.record:
        write_json(args.record, report)
    print({key: report[key] for key in ("dataset", "validation_span_cap") if key in report} | {
        "initialization_check": {alpha: {k: v for k, v in result.items() if not k.endswith("digest")}
                                 for alpha, result in report["initialization_check"].items()}})


def cmd_train(args):
    settings, splits, identity, _, _, weights, initialization = verified_inputs(args)
    import torch
    from safetensors.torch import load_file

    model_runs.get_device(torch, "cuda")
    initialize = adapted_initializer(load_file(str(weights), device="cpu"), args.alpha, initialization)
    experiment = experiment_name(args.alpha)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "sweep_run.json", {
        "alpha": args.alpha, "seed": args.seed, "experiment": experiment,
        "role": "control" if args.alpha == CONTROL_ALPHA else "candidate", "gpu": gpu_record(torch),
        "training_configuration_file": TRAINING_CONFIG.relative_to(TRAINING_CONFIG.parents[2]).as_posix(),
    })
    training = args.output / "training"
    training.mkdir()
    train_args = SimpleNamespace(training_config=TRAINING_CONFIG, pilot_steps=None, device="cuda",
                                 mps_memory_fraction=None, seed=args.seed, offline=args.offline, output=training)
    provenance = {"kind": "phase2_pinned", "bundle_manifest_sha256": identity["bundle_manifest_sha256"],
                  "variant": identity["variant"],
                  "split_sha256": {split: identity["split_sha256"][split] for split in ("train", "validation")}}
    model_runs.train_qa(train_args, settings, {split: splits[split] for split in ("train", "validation")},
                        provenance, experiment, settings["languages"], initialize_encoder=initialize)
    evaluate_split(args.output, "validation", settings, splits, identity, args.save_validation_logits)
    if not args.keep_epoch_checkpoints:
        shutil.rmtree(training / "epoch_checkpoints")


def cmd_validate(args):
    settings, splits, identity, *_ = verified_inputs(args)
    checkpoint_record(args.run)
    evaluate_split(args.run, "validation", settings, splits, identity, args.save_validation_logits)


def cmd_test(args):
    selection = read_json(args.selection)
    alpha = run_alpha(checkpoint_record(args.run))
    allowed = {CONTROL_ALPHA} | ({selection["selected_alpha"]} if selection["selected_alpha"] is not None else set())
    if alpha not in allowed:
        raise ValueError(f"alpha {alpha:g} is neither the control nor the frozen candidate in {args.selection}")
    settings, splits, identity, *_ = verified_inputs(args)
    evaluate_split(args.run, "test", settings, splits, identity)


# ---------------------------------------------------------------- analysis (no neural execution)

def cmd_select(args):
    control = run_summary(args.control, "validation")
    candidates = [run_summary(run, "validation") for run in args.candidates]
    if control["alpha"] != CONTROL_ALPHA or any(c["alpha"] == CONTROL_ALPHA for c in candidates):
        raise ValueError("The control must be alpha 0 and every candidate must have alpha > 0")
    if len({c["alpha"] for c in candidates}) != len(candidates) or any(c["seed"] != control["seed"] for c in candidates):
        raise ValueError("Candidates need distinct alphas at the control seed")
    check_matched([control, *candidates])
    selected = choose(control, candidates)
    entries = []
    for summary in (control, *candidates):
        trainer = trainer_validation(Path(summary["run"]), summary)
        entry = compact(summary)
        entry["trainer_validation_max_abs_difference"] = max(
            abs(trainer[lang][metric] - summary["metrics"][lang][metric])
            for lang in LANGUAGES for metric in ("exact_match", "f1"))
        if summary is not control:
            entry.update(
                english_f1_at_least_control=summary["metrics"]["en"]["f1"] >= control["metrics"]["en"]["f1"],
                indic_macro_f1_above_control=summary["metrics"]["indic_macro"]["f1"]
                > control["metrics"]["indic_macro"]["f1"],
                eligible=qualifies(control, summary))
        entries.append(entry)
    write_json(args.output, {
        "rule": SELECTION_RULE, "split": "validation", "seed": control["seed"], "control": entries[0],
        "candidates": entries[1:], "decision": "adapted_candidate" if selected else "keep_alpha_0_control",
        "selected_run": selected["run"] if selected else None,
        "selected_alpha": selected["alpha"] if selected else None,
    })
    print(f"Selection: {'alpha ' + format(selected['alpha'], 'g') if selected else 'keep alpha-0 control'} "
          f"-> {args.output}")


def cmd_span_cap(args):
    settings, splits, identity, *_ = verified_inputs(args)
    from transformers import AutoTokenizer

    record = checkpoint_record(args.run)
    directory = args.run / "validation"
    manifest = read_json(directory / "run_manifest.json")
    if manifest["checkpoint"] != record or sha256(directory / "window_logits.jsonl") != manifest[
            "output_file_sha256"]["window_logits.jsonl"]:
        raise ValueError("Saved validation logits do not belong to this run's selected checkpoint")
    rows, token_settings = splits["validation"], settings["tokenization"]
    cap = token_settings["max_answer_length"]
    if not cap < args.larger_cap <= token_settings["max_seq_length"]:
        raise ValueError("The comparison cap must exceed the current cap and fit in a window")
    tokenizer = AutoTokenizer.from_pretrained(str(args.run / "training" / "best_checkpoint"), use_fast=True,
                                              local_files_only=True)
    features, statistics = prepare_features(rows, tokenizer, token_settings)
    saved = read_jsonl(directory / "window_logits.jsonl")
    if len(saved) != len(features) or any(
            window["id"] != feature["id"] or [tuple(p) if p else None for p in window["offsets"]] != feature["offsets"]
            for window, feature in zip(saved, features)):
        raise ValueError("Saved logits do not align with the rebuilt validation windows")
    starts, ends = [window["start_logits"] for window in saved], [window["end_logits"] for window in saved]

    def score(limit):
        decoded = decode_logits(rows, features, starts, ends, {**token_settings, "max_answer_length": limit})
        return scored_predictions(rows, decoded, statistics, record["experiment"], record["seed"])

    current, larger = score(cap), score(args.larger_cap)
    spans = lambda records: {row["id"]: (row["predicted_start"], row["predicted_end"]) for row in records}
    if spans(current) != spans(read_jsonl(directory / "predictions.jsonl")):
        raise ValueError("Decoding the saved logits does not reproduce the validation predictions")
    before, after = spans(current), spans(larger)
    a_groups, b_groups = compare_inputs(current, larger)
    current_scores, larger_scores = aggregate(current), aggregate(larger)
    write_json(args.output, {
        "check": "decoder span cap on saved validation logits; separate from the adaptation sweep",
        "run": args.run.as_posix(), "alpha": run_alpha(record), "seed": record["seed"], "split": "validation",
        "current_cap": cap, "larger_cap": args.larger_cap, "rows": len(rows),
        "gold_exceeding_current_cap": gold_over_cap(rows, statistics, cap),
        "gold_exceeding_larger_cap": gold_over_cap(rows, statistics, args.larger_cap),
        "changed_predictions": {lang: sum(1 for row in current if row["language"] == lang
                                          and before[row["id"]] != after[row["id"]]) for lang in LANGUAGES},
        "current": current_scores, "larger": larger_scores,
        "delta_percentage_points": {name: {metric: larger_scores[name][metric] - current_scores[name][metric]
                                           for metric in ("exact_match", "f1")} for name in POPULATIONS},
        "intervals_source_document": bootstrap(a_groups, b_groups, unit="document", **settings["bootstrap"]),
    })
    print(f"Wrote {args.output}")


def paired(control_run, candidate_run, split, settings):
    a = read_jsonl(split_dir(control_run, split) / "predictions.jsonl")
    b = read_jsonl(split_dir(candidate_run, split) / "predictions.jsonl")
    groups = compare_inputs(a, b)
    scores_a, scores_b = aggregate(a), aggregate(b)
    return {
        "split": split, "control": control_run.as_posix(), "candidate": candidate_run.as_posix(),
        "paired_groups": len(groups[0]),
        "delta_percentage_points": {name: {metric: scores_b[name][metric] - scores_a[name][metric]
                                           for metric in ("exact_match", "f1")} for name in POPULATIONS},
        "intervals_source_document": bootstrap(*groups, unit="document", **settings["bootstrap"]),
    }


def stability(pairs, split):
    rows = [pair for pair in pairs if pair["split"] == split]
    result = {}
    for name in ("en", "indic_macro", "overall_macro"):
        values = [pair["delta_percentage_points"][name]["f1"] for pair in rows]
        result[f"{name}_f1"] = {"per_seed": values, "mean": mean(values), "min": min(values), "max": max(values),
                                "positive_seeds": sum(value > 0 for value in values)}
    return result


def table_row(summary, split, arm):
    metrics = summary["metrics"]
    return {"split": split, "seed": summary["seed"], "arm": arm, "alpha": f"{summary['alpha']:g}",
            "selected_checkpoint": summary["selected_checkpoint"],
            **{f"{name}_{short}": round(metrics[name][metric], 2) for name in POPULATIONS
               for short, metric in (("em", "exact_match"), ("f1", "f1"))}}


def markdown(rows, pairs):
    names = {"indic_macro": "Indic", "overall_macro": "Overall"}
    head = ["Split", "Seed", "Arm", "α", "Checkpoint"] + [f"{names.get(n, n)} EM/F1" for n in POPULATIONS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        lines.append("| " + " | ".join([row["split"], str(row["seed"]), row["arm"], row["alpha"],
                                        row["selected_checkpoint"]] + [f"{row[f'{n}_em']:.2f} / {row[f'{n}_f1']:.2f}"
                                                                       for n in POPULATIONS]) + " |")
    if not pairs:
        return "\n".join(lines) + "\n\nNo candidate was frozen, so there are no paired candidate − control deltas.\n"
    lines += ["", "Paired deltas (candidate − control, percentage points; 95% CI resampling source documents):", "",
              "| Split | Control → Candidate | " + " | ".join(f"Δ {names.get(n, n)} EM / F1" for n in POPULATIONS) + " |",
              "|" + "---|" * (2 + len(POPULATIONS))]
    for pair in pairs:
        intervals = {(i["language"], i["metric"]): i for i in pair["intervals_source_document"]}
        cells = []
        for name in POPULATIONS:
            em, f1 = intervals[(name, "exact_match")], intervals[(name, "f1")]
            cells.append(f"{em['delta_pp']:+.2f} / {f1['delta_pp']:+.2f} [{f1['lower_pp']:+.2f}, {f1['upper_pp']:+.2f}]")
        lines.append(f"| {pair['split']} | {Path(pair['control']).name} → {Path(pair['candidate']).name} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def cmd_compare(args):
    settings = read_settings("evaluation_config.yaml")
    selection = read_json(args.selection)
    if len(args.control_runs) != len(args.candidate_runs):
        raise ValueError("Pass one candidate run per control run")
    sweep = [Path(selection["control"]["run"])] + [Path(entry["run"]) for entry in selection["candidates"]]
    rows, pairs, runtime = [], [], {}
    reference_runtime = read_json(REFERENCE / "computational_statistics.json")["training"]
    for run in sweep:
        summary = run_summary(run, "validation")
        role = "control" if summary["alpha"] == CONTROL_ALPHA else (
            "selected" if summary["run"] == selection["selected_run"] else "sweep candidate")
        rows.append(table_row(summary, "validation", role))
    for control, candidate in zip(args.control_runs, args.candidate_runs):
        summaries = [run_summary(control, "validation"), run_summary(candidate, "validation")]
        check_matched(summaries)
        if summaries[0]["alpha"] != CONTROL_ALPHA or summaries[1]["alpha"] != selection["selected_alpha"] or \
                summaries[0]["seed"] != summaries[1]["seed"]:
            raise ValueError(f"{control} and {candidate} are not a paired control and frozen candidate")
        if summaries[0]["seed"] != selection["seed"]:
            rows += [table_row(summaries[0], "validation", "control"), table_row(summaries[1], "validation", "selected")]
        pairs.append(paired(control, candidate, "validation", settings))
    test_pairs = []
    for control, candidate in zip(args.control_runs, args.candidate_runs):
        rows += [table_row(run_summary(control, "test"), "test", "control"),
                 table_row(run_summary(candidate, "test"), "test", "selected")]
        test_pairs.append(paired(control, candidate, "test", settings))
    control_42 = Path(selection["control"]["run"])
    reproduction = None
    if split_dir(control_42, "test").exists():
        recorded = read_jsonl(REFERENCE / "predictions.jsonl")
        rerun = read_jsonl(split_dir(control_42, "test") / "predictions.jsonl")
        groups = compare_inputs(recorded, rerun)
        reproduction = {
            "recorded_phase_3b_predictions_sha256": sha256(REFERENCE / "predictions.jsonl"),
            "alpha_0_predictions_sha256": sha256(split_dir(control_42, "test") / "predictions.jsonl"),
            "identical_spans": [(r["id"], r["predicted_start"], r["predicted_end"]) for r in recorded]
                               == [(r["id"], r["predicted_start"], r["predicted_end"]) for r in rerun],
            "phase_3b": aggregate(recorded), "alpha_0": aggregate(rerun),
            "intervals_source_document": bootstrap(*groups, unit="document", **settings["bootstrap"]),
        }
        if not any(Path(c) == control_42 for c in args.control_runs):
            rows.append(table_row(run_summary(control_42, "test"), "test", "control"))
    for run in sweep[1:]:
        if run.as_posix() != selection["selected_run"] and split_dir(run, "test").exists():
            rows.append(table_row(run_summary(run, "test"), "test", "context: test seen before the sweep"))
    for run in {*sweep, *args.control_runs, *args.candidate_runs}:
        summary = run_summary(run, "validation")
        sweep_record = run / "sweep_run.json"
        runtime[run.as_posix()] = {**summary["runtime"], "gpu": read_json(sweep_record)["gpu"] if sweep_record.exists() else None,
                                   "differs_from_phase_3b": {key: [reference_runtime.get(key), summary["runtime"][key]]
                                                             for key in summary["runtime"]
                                                             if summary["runtime"][key] != reference_runtime.get(key)}}
    all_pairs = pairs + test_pairs
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "comparison.json", {
        "protocol": settings["metric_protocol"], "selection": selection, "table": rows, "paired": all_pairs,
        "seed_stability": {**({"validation": stability(pairs, "validation")} if pairs else {}),
                           **({"test": stability(test_pairs, "test")} if test_pairs else {})},
        "alpha_0_reproduces_phase_3b": reproduction, "runtime": runtime,
    })
    write_csv(args.output / "comparison_table.csv", tuple(rows[0]), rows)
    (args.output / "comparison_table.md").write_text(markdown(rows, all_pairs), encoding="utf-8")
    print(f"Wrote {args.output}")


# ---------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = argparse.ArgumentParser(add_help=False)
    source = data.add_mutually_exclusive_group()
    source.add_argument("--bundle-dir", type=Path, help="Original pinned Gemini v2 bundle used in Phase 3")
    source.add_argument("--export-dir", type=Path, help="Directory with the three original Gemini flat exports")
    data.add_argument("--source-dir", type=Path, help="Original English AgXQA JSONLs for --export-dir")
    data.add_argument("--adaptation-dir", type=Path, required=True, help="Private saved Phase 5 MLM run")
    data.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    data.add_argument("--offline", action="store_true", help="Use the already cached XLM-R base")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", parents=[data], help="Verify inputs, initialization endpoints and span-cap counts")
    check.add_argument("--record", type=Path, help="Write the preflight record here")
    check.add_argument("--require-cuda", action="store_true")
    check.add_argument("--larger-cap", type=int, default=160)
    train = commands.add_parser("train", parents=[data], help="Train one arm and evaluate it on validation")
    train.add_argument("--alpha", type=float, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--save-validation-logits", action="store_true")
    train.add_argument("--keep-epoch-checkpoints", action="store_true")
    validate = commands.add_parser("validate", parents=[data], help="Evaluate an existing run on validation")
    validate.add_argument("--run", type=Path, required=True)
    validate.add_argument("--save-validation-logits", action="store_true")
    select = commands.add_parser("select", help="Freeze the candidate from validation results")
    select.add_argument("--control", type=Path, required=True)
    select.add_argument("--candidates", type=Path, nargs="+", required=True)
    select.add_argument("--output", type=Path, required=True)
    test = commands.add_parser("test", parents=[data], help="Evaluate the control or frozen candidate on test")
    test.add_argument("--run", type=Path, required=True)
    test.add_argument("--selection", type=Path, required=True)
    span = commands.add_parser("span-cap", parents=[data], help="Compare decoder caps on saved validation logits")
    span.add_argument("--run", type=Path, required=True)
    span.add_argument("--larger-cap", type=int, default=160)
    span.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="Paired validation and test comparison table")
    compare.add_argument("--selection", type=Path, required=True)
    compare.add_argument("--control-runs", type=Path, nargs="*", default=[])
    compare.add_argument("--candidate-runs", type=Path, nargs="*", default=[])
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, "export_dir", None) is not None and args.source_dir is None:
        parser.error("--export-dir and --source-dir must be provided together")
    if getattr(args, "alpha", None) is not None and not 0 <= args.alpha <= 1:
        parser.error("--alpha must be in [0, 1]")
    handlers = {"check": cmd_check, "train": cmd_train, "validate": cmd_validate, "select": cmd_select,
                "test": cmd_test, "span-cap": cmd_span_cap, "compare": cmd_compare}
    try:
        handlers[args.command](args)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f"Phase 5 alpha sweep {args.command}: {error}\n")


if __name__ == "__main__":
    main()
