"""Opt-in XLM-R training and evaluation for matched Phase 3 experiments."""

import argparse
from importlib.metadata import version
from pathlib import Path
import platform
from time import perf_counter

from shared.qa_release import VARIANTS, sha256
from phase_01_original_baseline.src.agxqa.metrics import score as score_english
from phase_02_multilingual_benchmark.pipeline import HASHES, REVISION, load_source
from .common import file_manifest, load_data, read_json, read_settings, write_csv, write_json, write_jsonl
from .features import decode_logits, prepare_features, summarize_tokenization
from .scoring import aggregate, consistency, scored_predictions


PROVENANCE = "phase3_checkpoint.json"
TRAINER_CHECKPOINT_FILES = ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")


def load_checkpoint_provenance(path, supplied, settings, experiment, identity):
    provenance_path = supplied or path / PROVENANCE
    record = read_json(provenance_path)
    files = file_manifest(path)
    if provenance_path.resolve().is_relative_to(path.resolve()):
        files.pop(str(provenance_path.resolve().relative_to(path.resolve())))
    if files != record["checkpoint_file_sha256"]:
        raise ValueError("Checkpoint files differ from their recorded hashes")
    if (record["base_model"], record["base_revision"]) != (settings["base_model"], settings["base_revision"]):
        raise ValueError("Checkpoint is not from the pinned XLM-R Base initialization")
    if not record["training_configuration"] or not record["computational_statistics"]:
        raise ValueError("Checkpoint training configuration and computational statistics are required")
    expected = ["en"] if experiment == "phase_3a" else settings["languages"]
    if record["training_languages"] != expected or record["selection_languages"] != expected:
        raise ValueError("Checkpoint training or model-selection languages violate the experiment")

    data = record["training_data"]
    if experiment == "phase_3a" and data["kind"] == "phase1_official_english":
        if (identity["variant"] == "document_disjoint" or data["revision"] != REVISION
                or data["split_sha256"] != {split: HASHES[split] for split in ("train", "validation")}):
            raise ValueError("English checkpoint provenance is incompatible with this dataset variant")
    elif (data["kind"] != "phase2_pinned"
          or data["bundle_manifest_sha256"] != identity["bundle_manifest_sha256"]
          or data["variant"] != identity["variant"]
          or data["split_sha256"] != {split: identity["split_sha256"][split] for split in ("train", "validation")}):
        raise ValueError("Checkpoint was not trained on this experiment's pinned train and validation data")
    return record


def get_device(torch, requested):
    available = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    selected = next(name for name in ("cuda", "mps", "cpu") if available[name]) if requested == "auto" else requested
    if not available[selected]:
        raise ValueError(f"Requested device {selected} is unavailable")
    return torch.device(selected)


def configure_mps_memory(torch, device, fraction):
    if fraction is None:
        return None
    if device.type != "mps" or not 0 < fraction <= 1:
        raise ValueError("An MPS memory cap requires the MPS device and a fraction in (0, 1]")
    recommended = torch.mps.recommended_max_memory()
    torch.mps.set_per_process_memory_fraction(fraction)
    limit = {"fraction": fraction, "recommended_working_set_bytes": recommended,
             "allocator_limit_bytes": int(fraction * recommended)}
    print(f"MPS allocation capped at {limit['allocator_limit_bytes'] / (1 << 30):.2f} GiB", flush=True)
    return limit


def pilot_training_settings(settings, steps):
    if steps < 1:
        raise ValueError("A memory pilot must have at least one optimizer step")
    return {**settings, "max_steps": steps, "eval_strategy": "no", "save_strategy": "no",
            "metric_for_best_model": None, "load_best_model_at_end": False, "logging_steps": 1,
            "report_to": []}


def load_model_and_tokenizer(reference, revision, offline, trained):
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    kwargs = {"revision": revision, "local_files_only": offline, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(reference, use_fast=True, **kwargs)
    model, loading = AutoModelForQuestionAnswering.from_pretrained(
        reference, output_loading_info=True, use_safetensors=None if trained else True, weights_only=True, **kwargs)
    config = model.config
    architecture = (config.model_type, config.hidden_size, config.num_hidden_layers, config.vocab_size, config.num_labels)
    if architecture != ("xlm-roberta", 768, 12, 250002, 2):
        raise ValueError("Expected an XLM-RoBERTa Base extractive QA model")
    missing = set(loading["missing_keys"])
    allowed_missing = set() if trained else {"qa_outputs.weight", "qa_outputs.bias"}
    if missing != allowed_missing or loading["mismatched_keys"] or loading.get("error_msgs"):
        raise ValueError(f"Unexpected checkpoint load results: {loading}")
    if tokenizer.padding_side != "right" or len(tokenizer) != config.vocab_size:
        raise ValueError("Tokenizer is not compatible with the XLM-R checkpoint")
    return model, tokenizer


def weights_metadata(model, device):
    import torch

    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
    else:
        name = platform.processor() or platform.machine()
    return {"parameters": sum(parameter.numel() for parameter in model.parameters()),
            "device": str(device), "device_name": name,
            "platform": platform.platform(),
            "versions": {package: version(package) for package in ("torch", "transformers", "accelerate", "tokenizers", "numpy")}}


def log_training(output, history, seed, languages):
    rows, validation = [], []
    learning_rate = None
    for event in history:
        if "learning_rate" in event:
            learning_rate = event["learning_rate"]
        if "loss" in event or "eval_loss" in event:
            rows.append({"seed": seed, "epoch": event.get("epoch"), "step": event.get("step"),
                         "train_loss": event.get("loss"), "val_loss": event.get("eval_loss"),
                         "val_em": event.get("eval_macro_em"), "val_f1": event.get("eval_macro_f1"),
                         "learning_rate": event.get("learning_rate", learning_rate)})
        if "eval_loss" in event:
            for language in languages:
                validation.append({"seed": seed, "epoch": event["epoch"], "step": event["step"], "language": language,
                                   "em": event[f"eval_{language}_em"], "f1": event[f"eval_{language}_f1"]})
    write_csv(output / "training_logs.csv", ("seed", "epoch", "step", "train_loss", "val_loss",
                                             "val_em", "val_f1", "learning_rate"), rows)
    write_csv(output / "validation_metrics.csv", ("seed", "epoch", "step", "language", "em", "f1"), validation)
    write_json(output / "trainer_log_history.json", history)


def training_configuration(override):
    training = read_settings("multilingual_config.yaml")["training"]
    if override:
        import yaml

        training.update(yaml.safe_load(override.read_text(encoding="utf-8"))["training"])
    return training


def read_resume_state(output, checkpoint):
    checkpoint_root = (output / "epoch_checkpoints").resolve()
    if checkpoint.resolve().parent != checkpoint_root:
        raise ValueError("Resume from an epoch checkpoint in the original run output directory")
    missing = [name for name in TRAINER_CHECKPOINT_FILES
               if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size]
    if missing:
        raise ValueError(f"Epoch checkpoint is incomplete. Missing or empty files: {', '.join(missing)}")
    state = read_json(checkpoint / "trainer_state.json")
    step = state["global_step"]
    if (checkpoint.name != f"checkpoint-{step}" or not 0 < step < state["max_steps"]
            or not (state["epoch"] > 0 and state["epoch"] == int(state["epoch"]))):
        raise ValueError("Resume requires an unfinished run's completed epoch checkpoint")
    if not state["best_model_checkpoint"]:
        raise ValueError("The epoch checkpoint has no validation selection state")
    best = Path(state["best_model_checkpoint"]).resolve()
    if best.parent != checkpoint_root or not (best / "model.safetensors").is_file():
        raise ValueError("The previous best validation checkpoint is missing from the original run")
    later_steps = (path.name.removeprefix("checkpoint-") for path in checkpoint_root.glob("checkpoint-*")
                   if path.is_dir())
    if any(value.isdecimal() and int(value) > step for value in later_steps) or (output / "best_checkpoint").exists():
        raise ValueError("The original run has newer training artifacts")
    return state


def verify_resume_inputs(output, checkpoint, settings, training, identity, seed):
    state = read_resume_state(output, checkpoint)
    launch = read_json(output / "launch_inputs.json")
    previous_training = training_configuration(Path(launch["training_configuration_override"]))
    for key in ("gradient_checkpointing", "gradient_checkpointing_kwargs"):
        previous_training.pop(key, None)
    current_training = {key: value for key, value in training.items()
                        if key not in ("gradient_checkpointing", "gradient_checkpointing_kwargs")}
    if previous_training != current_training:
        raise ValueError("Resume must preserve the training configuration except for gradient checkpointing")
    initialization = launch["initialization"]
    if (launch["experiment"] != "phase_3b" or launch["seed"] != seed
            or launch["dataset"] != identity or launch["optimizer_steps_planned"] != state["max_steps"]
            or initialization["model"] != settings["base_model"]
            or initialization["revision"] != settings["base_revision"]):
        raise ValueError("Resume inputs differ from the original run")
    return {"checkpoint": str(checkpoint), "optimizer_step": state["global_step"],
            "previous_device": launch["device"], "previous_training_configuration": launch["training_configuration_override"]}


def train_qa(args, settings, splits, provenance, experiment, training_languages, resume=None, *,
             initialize_encoder=None):
    import torch
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

    settings_training = training_configuration(args.training_config)
    token_settings = settings["tokenization"]
    padding_multiple = settings_training.get("pad_to_multiple_of")
    if padding_multiple is not None and (type(padding_multiple) is not int or padding_multiple < 1
                                        or token_settings["max_seq_length"] % padding_multiple):
        raise ValueError("pad_to_multiple_of must be a positive divisor of max_seq_length")
    if args.pilot_steps is None and settings_training.get("max_steps", -1) != -1:
        raise ValueError("Use --pilot-steps for a bounded training pilot")
    for key, value in {"metric_for_best_model": "macro_f1", "eval_strategy": "epoch", "save_strategy": "epoch",
                       "load_best_model_at_end": True, "greater_is_better": True}.items():
        if settings_training[key] != value:
            raise ValueError(f"Training must use {key}={value} for the Phase 3 checkpoint selection protocol")
    device = get_device(torch, args.device)
    mps_limit = configure_mps_memory(torch, device, args.mps_memory_fraction)
    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(settings["base_model"], settings["base_revision"], args.offline, trained=False)
    initialization = None
    if initialize_encoder is not None:
        if resume:
            raise ValueError("Adapted initialization is only supported for a fresh run")
        initialization = initialize_encoder(model)
    pilot = args.pilot_steps is not None
    prepared = {split: prepare_features(splits[split], tokenizer, token_settings, labels=True)
                for split in (("train",) if pilot else ("train", "validation"))}
    for split, (_, stats) in prepared.items():
        unaligned = [identifier for identifier, stat in stats.items() if not stat["positive_windows"]]
        if unaligned:
            raise ValueError(f"Gold spans fit no token window in {split}: {', '.join(unaligned)}")

    def metrics(result):
        decoded = decode_logits(splits["validation"], prepared["validation"][0], *result.predictions, token_settings)
        scored = scored_predictions(splits["validation"], decoded, prepared["validation"][1], experiment, args.seed)
        values = aggregate(scored)
        return {"macro_em": values["overall_macro"]["exact_match"], "macro_f1": values["overall_macro"]["f1"],
                **{f"{lang}_{name}": values[lang][key] for lang in training_languages
                   for name, key in (("em", "exact_match"), ("f1", "f1"))}}

    runtime_settings = pilot_training_settings(settings_training, args.pilot_steps) if pilot else settings_training
    trainer_settings = {key: value for key, value in runtime_settings.items() if key != "pad_to_multiple_of"}
    trainer_args = TrainingArguments(
        output_dir=str(args.output / "epoch_checkpoints"), seed=args.seed, data_seed=args.seed,
        use_cpu=device.type == "cpu", label_names=["start_positions", "end_positions"],
        include_num_input_tokens_seen="non_padding", save_safetensors=True, **trainer_settings,
    )
    if trainer_args.device.type != device.type or trainer_args.world_size != 1 or trainer_args.n_gpu > 1:
        raise ValueError("The configured experiment uses one device. Select the requested backend and expose one GPU")
    trainer = Trainer(
        model=model, args=trainer_args, processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=padding_multiple),
        train_dataset=[feature["inputs"] for feature in prepared["train"][0]],
        eval_dataset=None if pilot else [feature["inputs"] for feature in prepared["validation"][0]],
        compute_metrics=None if pilot else metrics,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if resume:
        print(f"Resuming from optimizer step {resume['optimizer_step']} on {device.type}", flush=True)
    started = perf_counter()
    trainer.train(resume_from_checkpoint=resume["checkpoint"] if resume else None)
    elapsed = perf_counter() - started
    if pilot:
        if trainer.state.global_step != args.pilot_steps:
            raise ValueError("The memory pilot did not complete its requested optimizer steps")
        write_json(args.output / "memory_pilot.json", {
            "mode": "memory_pilot", "optimizer_steps": trainer.state.global_step, "elapsed_seconds": elapsed,
            "training_configuration": runtime_settings, "mps_memory_limit": mps_limit, "training_data": provenance,
            "mps_driver_bytes_at_completion": torch.mps.driver_allocated_memory() if device.type == "mps" else None,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "checkpoint_saved": False, "evaluation_performed": False,
        })
        write_json(args.output / "trainer_log_history.json", trainer.state.log_history)
        return
    if not trainer.state.best_model_checkpoint or trainer.state.best_metric is None:
        raise ValueError("Training finished without a checkpoint selected by validation macro-F1")
    checkpoint = args.output / "best_checkpoint"
    trainer.save_model(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    log_training(args.output, trainer.state.log_history, args.seed, training_languages)
    training_metrics = {
        **weights_metadata(model, device), "training_rows": len(splits["train"]),
        "tokenized_training_features": len(prepared["train"][0]), "optimizer_steps": trainer.state.global_step,
        "training_input_tokens": trainer.state.num_input_tokens_seen, "training_seconds": elapsed,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "validation_best_macro_f1": trainer.state.best_metric, "mps_memory_limit": mps_limit,
        "selected_epoch_checkpoint": Path(trainer.state.best_model_checkpoint).name,
    }
    if resume:
        training_metrics["resumed_from"] = resume
        training_metrics["training_seconds_scope"] = "resumed_segment"
    write_json(args.output / "computational_statistics.json", training_metrics)
    write_json(args.output / "tokenization_statistics.json", {
        split: summarize_tokenization(stats, token_settings["max_answer_length"])
        for split, (_, stats) in prepared.items()
    })
    if provenance["kind"] == "phase1_official_english":
        validation = trainer.predict(trainer.eval_dataset)
        predictions = decode_logits(splits["validation"], prepared["validation"][0], *validation.predictions, token_settings)
        baseline, _ = score_english(splits["validation"], {key: span["predicted_answer"] for key, span in predictions.items()})
        write_json(args.output / "english_source_validation_metrics.json", baseline)
    record = {
        "experiment": experiment, "base_model": settings["base_model"], "base_revision": settings["base_revision"],
        "seed": args.seed, "training_languages": training_languages, "selection_languages": training_languages,
        "training_data": provenance, "training_configuration": settings_training, "tokenization": token_settings,
        "computational_statistics": training_metrics, "checkpoint_file_sha256": file_manifest(checkpoint),
    }
    if initialization is not None:
        record["adapted_initialization"] = initialization
    write_json(checkpoint / PROVENANCE, record)


def evaluate_qa(args, settings, rows, identity, experiment):
    import torch
    from transformers import DataCollatorWithPadding

    provenance = load_checkpoint_provenance(args.checkpoint, args.provenance, settings, experiment, identity)
    if "tokenization" in provenance and provenance["tokenization"] != settings["tokenization"]:
        raise ValueError("Checkpoint tokenization and the evaluation configuration differ")
    device = get_device(torch, args.device)
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, None, True, trained=True)
    model.to(device).eval()
    token_settings = settings["tokenization"]
    features, statistics = prepare_features(rows, tokenizer, token_settings)
    batches = torch.utils.data.DataLoader(
        [feature["inputs"] for feature in features], batch_size=args.batch_size,
        shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    start_logits, end_logits = [], []
    with torch.inference_mode():
        for batch in batches:
            result = model(**{key: tensor.to(device) for key, tensor in batch.items()})
            start_logits.extend(result.start_logits.float().cpu().tolist())
            end_logits.extend(result.end_logits.float().cpu().tolist())
    elapsed = perf_counter() - started
    decoded = decode_logits(rows, features, start_logits, end_logits, token_settings)
    seed = provenance["seed"]
    predictions = scored_predictions(rows, decoded, statistics, experiment, seed)
    scores = aggregate(predictions)
    english = [row for row in rows if row["language"] == "en"]
    squad_scores, _ = score_english(english, {row["id"]: decoded[row["id"]]["predicted_answer"] for row in english})
    computation = {
        **weights_metadata(model, device), "inference_window_batches": len(batches), "test_windows": len(features),
        "inference_seconds": elapsed, "average_inference_ms_per_question_row": 1000 * elapsed / len(rows),
        "inference_peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "training": provenance["computational_statistics"],
    }
    write_jsonl(args.output / "predictions.jsonl", predictions)
    if args.save_window_logits:
        write_jsonl(args.output / "window_logits.jsonl", (
            {"id": feature["id"], "offsets": feature["offsets"], "start_logits": starts, "end_logits": ends}
            for feature, starts, ends in zip(features, start_logits, end_logits)
        ))
    write_json(args.output / "metrics.json", {"protocol": settings["metric_protocol"],
                                            "by_language_and_macro": scores, "matched_test_groups": len(english),
                                            "english_squad_v1": squad_scores, "joint": consistency(predictions)})
    write_csv(args.output / "language_wise_results.csv", ("language", "count", "exact_match", "f1"),
              [{"language": lang, **score} for lang, score in scores.items()])
    write_json(args.output / "computational_statistics.json", computation)
    write_json(args.output / "run_manifest.json", {
        "experiment": experiment, "seed": seed, "dataset": identity, "protocol": settings["metric_protocol"],
        "tokenization": token_settings, "checkpoint": provenance,
        "output_file_sha256": file_manifest(args.output),
    })


def main(operation):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-model", action="store_true", help="Explicitly enable neural execution")
    parser.add_argument("--variant", choices=VARIANTS, default="final")
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.set_defaults(resume_from_checkpoint=None)
    if operation.startswith("train"):
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--offline", action="store_true")
        parser.add_argument("--training-config", type=Path)
        parser.add_argument("--mps-memory-fraction", type=float)
        parser.add_argument("--pilot-steps", type=int, help="Run a bounded training pilot without saving or evaluation")
        if operation == "train_english":
            parser.add_argument("--source-dir", type=Path, help="Pinned official English files for the Phase 1 fallback")
        else:
            parser.add_argument("--resume-from-checkpoint", type=Path,
                                help="Continue a complete epoch checkpoint in the original output directory")
    else:
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--provenance", type=Path, help="Optional external Phase 1 checkpoint provenance JSON")
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--save-window-logits", action="store_true")
    args = parser.parse_args()
    if not args.run_model:
        parser.error("--run-model is required for training or inference")
    if operation.startswith("train") and args.pilot_steps is not None and args.pilot_steps < 1:
        parser.error("--pilot-steps must be positive")
    if args.resume_from_checkpoint and args.pilot_steps is not None:
        parser.error("Resume is available for full training runs")
    try:
        settings = read_settings("evaluation_config.yaml")
        experiment = "3a" if operation in ("evaluate_zero_shot", "train_english") else "3b"
        control = read_settings("zero_shot_config.yaml" if experiment == "3a" else "multilingual_config.yaml")
        languages = ["en"] if experiment == "3a" else settings["languages"]
        if (control["experiment"] != f"phase_{experiment}" or control["training_languages"] != languages
                or control["selection_languages"] != languages):
            raise ValueError("Training and selection languages do not match the Phase 3 experiment")
        if operation == "train_english" and args.source_dir:
            if args.variant == "document_disjoint":
                raise ValueError("Use the document-disjoint bundle and omit --source-dir for the sensitivity checkpoint")
            source, audit = load_source(args.source_dir)
            splits = {split: [{**row, "id": f"{row['id']}_en", "language": "en"}
                              for row in source if row["split"] == split] for split in ("train", "validation")}
            provenance = {"kind": "phase1_official_english", "revision": audit["revision"],
                          "split_sha256": {split: audit["file_sha256"][split] for split in splits}}
            args.output.mkdir(parents=True, exist_ok=False)
            train_qa(args, settings, splits, provenance, "phase_3a", languages)
            print(f"Wrote {args.output}")
            return

        splits, identity = load_data(experiment, args.bundle_dir, args.variant)
        resume = None
        if args.resume_from_checkpoint:
            resume = verify_resume_inputs(args.output, args.resume_from_checkpoint, settings,
                                          training_configuration(args.training_config), identity, args.seed)
        else:
            args.output.mkdir(parents=True, exist_ok=False)
        if operation.startswith("train"):
            train_splits = {split: [row for row in splits[split] if row["language"] in languages]
                            for split in ("train", "validation")}
            provenance = {"kind": "phase2_pinned", "bundle_manifest_sha256": identity["bundle_manifest_sha256"],
                          "variant": identity["variant"],
                          "split_sha256": {split: identity["split_sha256"][split] for split in train_splits}}
            train_qa(args, settings, train_splits, provenance, f"phase_{experiment}", languages, resume)
        else:
            if args.batch_size < 1:
                raise ValueError("Evaluation batch size must be positive")
            evaluate_qa(args, settings, splits["test"], identity, f"phase_{experiment}")
        print(f"Wrote {args.output}")
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f"Phase 3 {operation}: {error}\n")
    except RuntimeError as error:
        message = str(error)
        if getattr(args, "mps_memory_fraction", None) is not None and "mps" in message.lower() and "out of memory" in message.lower():
            parser.exit(2, f"Phase 3 {operation}: the MPS allocator could not fit this workload under the requested memory cap.\n{message}\n")
        raise
