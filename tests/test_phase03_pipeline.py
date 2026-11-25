"""Synthetic Phase 3 invariants, using no neural model."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from phase_03_multilingual_finetuning.common import LANGUAGES, file_manifest, write_json
from phase_03_multilingual_finetuning.features import answer_bounds, decode_logits, label_span
from phase_03_multilingual_finetuning.model_runs import (
    TRAINER_CHECKPOINT_FILES, load_checkpoint_provenance, pilot_training_settings,
    training_configuration, verify_resume_inputs,
)
from phase_03_multilingual_finetuning.scoring import bootstrap, compare_inputs, consistency, scored_predictions


def fixture():
    return [{"id": f"1_{language}", "original_id": "1", "language": language,
             "context": "Water reaches roots.", "question": "Where does the water go?",
             "answers": {"text": ["roots"], "answer_start": [14]}, "source_document_id": "example"}
            for language in LANGUAGES]


class Phase3Tests(unittest.TestCase):
    def test_resume_requires_matching_inputs_and_optimizer_state(self):
        configs = Path(__file__).resolve().parents[1] / "phase_03_multilingual_finetuning/configs"
        training = training_configuration(configs / "mac_cpu_resume.yaml")
        identity = {"variant": "span_checked", "bundle_manifest_sha256": "fixture"}
        settings = {"base_model": "fixture", "base_revision": "test-revision"}
        with TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "epoch_checkpoints" / "checkpoint-448"
            checkpoint.mkdir(parents=True)
            write_json(output / "launch_inputs.json", {
                "experiment": "phase_3b", "seed": 42, "dataset": identity, "device": "mps",
                "optimizer_steps_planned": 2240,
                "initialization": {"model": "fixture", "revision": "test-revision"},
                "training_configuration_override": str(configs / "mac_mps_fixed_padding.yaml"),
            })
            write_json(checkpoint / "trainer_state.json", {
                "global_step": 448, "max_steps": 2240, "epoch": 1.0,
                "best_model_checkpoint": str(checkpoint),
            })
            for name in TRAINER_CHECKPOINT_FILES:
                if name != "trainer_state.json":
                    (checkpoint / name).write_bytes(b"synthetic")

            self.assertEqual(verify_resume_inputs(output, checkpoint, settings, training, identity, 42)["optimizer_step"], 448)
            with self.assertRaisesRegex(ValueError, "preserve the training configuration"):
                verify_resume_inputs(output, checkpoint, settings, dict(training, pad_to_multiple_of=128), identity, 42)
            with self.assertRaisesRegex(ValueError, "original run"):
                verify_resume_inputs(output, checkpoint, settings, training, dict(identity, variant="final"), 42)
            (checkpoint / "optimizer.pt").unlink()
            with self.assertRaisesRegex(ValueError, "optimizer.pt"):
                verify_resume_inputs(output, checkpoint, settings, training, identity, 42)

    def test_memory_pilot_is_bounded_without_changing_effective_batch(self):
        full = {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 16,
                "eval_strategy": "epoch", "save_strategy": "epoch", "load_best_model_at_end": True}
        pilot = pilot_training_settings(full, 5)
        self.assertEqual(pilot["max_steps"], 5)
        self.assertEqual(pilot["per_device_train_batch_size"] * pilot["gradient_accumulation_steps"], 16)
        self.assertEqual((pilot["eval_strategy"], pilot["save_strategy"]), ("no", "no"))
        self.assertEqual(pilot["logging_steps"], 1)
        self.assertFalse(pilot["load_best_model_at_end"])
        self.assertEqual(full["eval_strategy"], "epoch")
        with self.assertRaises(ValueError):
            pilot_training_settings(full, 0)

    def test_labels_ignore_question_tokens_and_align_boundary_whitespace(self):
        row = {**fixture()[0], "answers": {"text": [" roots"], "answer_start": [13]}}
        span, = answer_bounds(row)
        self.assertEqual(span, (14, 19))
        self.assertEqual(label_span([None, None, (13, 17), (17, 19), None], *span), (2, 3))
        self.assertIsNone(label_span([None, (13, 17), None], *span))

    def test_decoder_selects_best_exact_span_across_windows(self):
        row = fixture()[0]
        features = [{"id": row["id"], "offsets": [None, (0, 5), (6, 13)]},
                    {"id": row["id"], "offsets": [None, (14, 19), (19, 20)]}]
        decoded = decode_logits([row], features, [[99, 2, 1], [99, 4, 1]],
                                [[99, 2, 1], [99, 3, 1]], {"n_best_size": 2, "max_answer_length": 1})
        self.assertEqual(decoded[row["id"]]["predicted_answer"], "roots")
        self.assertEqual(decoded[row["id"]]["predicted_end"], 19)

    def test_joint_accuracy_and_bootstrap_use_matched_language_groups(self):
        rows = fixture()
        statistics = {row["id"]: {"context_tokens": 4, "windows": 1} for row in rows}
        correct = {row["id"]: {"predicted_answer": "roots", "predicted_start": 14, "predicted_end": 19} for row in rows}
        partly_correct = {**correct, "1_hi": {"predicted_answer": "Water", "predicted_start": 0, "predicted_end": 5}}
        a = scored_predictions(rows, partly_correct, statistics, "phase_3a", 42)
        b = scored_predictions(rows, correct, statistics, "phase_3b", 42)
        self.assertEqual(consistency(a)["correct_in_n_languages"][4]["percent"], 100)
        intervals = bootstrap(*compare_inputs(a, b), 8, .95, 42, "context")
        indic = next(row for row in intervals if row["language"] == "indic_macro" and row["metric"] == "f1")
        self.assertEqual((indic["delta_pp"], indic["lower_pp"], indic["upper_pp"]), (25, 25, 25))
        with self.assertRaises(ValueError):
            compare_inputs(a, b[:-1])

    def test_english_checkpoint_rejects_mismatched_file_hash_or_dataset(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text('{"synthetic": true}', encoding="utf-8")
            identity = {"variant": "document_disjoint", "bundle_manifest_sha256": "fixture",
                        "split_sha256": {"train": "a", "validation": "b", "test": "c"}}
            settings = {"base_model": "fixture", "base_revision": "fixture"}
            record = {
                **settings, "training_languages": ["en"], "selection_languages": ["en"],
                "training_configuration": {"epochs": 1}, "computational_statistics": {"steps": 1},
                "training_data": {"kind": "phase2_pinned", "variant": "document_disjoint",
                                  "bundle_manifest_sha256": "fixture", "split_sha256": {"train": "a", "validation": "b"}},
                "checkpoint_file_sha256": file_manifest(checkpoint),
            }
            write_json(checkpoint / "phase3_checkpoint.json", record)
            self.assertEqual(load_checkpoint_provenance(checkpoint, None, settings, "phase_3a", identity), record)
            with self.assertRaisesRegex(ValueError, "pinned train and validation"):
                load_checkpoint_provenance(checkpoint, None, settings, "phase_3a", dict(identity, variant="final"))
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recorded hashes"):
                load_checkpoint_provenance(checkpoint, None, settings, "phase_3a", identity)
