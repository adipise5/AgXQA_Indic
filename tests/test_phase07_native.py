"""Phase 7 Native evaluation invariants, using synthetic data and no neural model."""

import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from phase_03_multilingual_finetuning.common import write_json, write_jsonl
from phase_07_comparative_evaluation import evaluate_frozen as frozen
from phase_07_comparative_evaluation import statistical_analysis as stats
from shared.evaluate import evaluate
from shared.qa_release import sha256

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("hi", "mr", "bn", "ta")


def native_rows(per_language=3, documents=2):
    rows = []
    for language in LANGS:
        for index in range(per_language):
            context = f"{language} passage {index}: apply 25 kg urea per hectare before sowing."
            rows.append({"id": f"NATIVE_{language.upper()}_{index:06d}", "language": language, "split": "test",
                         "source_document_id": f"{language.upper()}_DOC_{index % documents}",
                         "context": context, "question": f"How much urea? ({index})",
                         "answers": [{"text": "25 kg urea", "answer_start": context.index("25 kg urea")}],
                         "metadata": {"human_validated": False}})
    return rows


def step1(directory, rows, **changes):
    write_jsonl(directory / "phase6_test.shared.jsonl", rows)
    write_json(directory / "native_overlap_audit.json", {"requires_overlap_review": changes.get("flags", False)})
    manifest = {"native_evaluation": {"frozen_filename": "phase6_test.shared.jsonl",
                                      "frozen_sha256": changes.get("sha", sha256(directory / "phase6_test.shared.jsonl")),
                                      "records": changes.get("records", len(rows))},
                "overlap_audit_filename": "native_overlap_audit.json", "checkpoints": changes.get("checkpoints", {})}
    write_json(directory / "manifest.json", manifest)
    return directory / "manifest.json", manifest


def predictions(rows, correct):
    return [{"id": row["id"], "language": row["language"],
             "prediction_text": "25 kg urea" if correct(row) else "before sowing"} for row in rows]


class DatasetTests(unittest.TestCase):
    def test_model_rows_convert_answers_to_parallel_arrays(self):
        rows = native_rows(1)
        converted = frozen.model_rows(rows)
        self.assertEqual(converted[0]["answers"], {"text": ["25 kg urea"], "answer_start": [rows[0]["answers"][0]["answer_start"]]})
        for key in ("id", "language", "question", "context", "source_document_id"):
            self.assertEqual(converted[0][key], rows[0][key])
        start = converted[0]["answers"]["answer_start"][0]
        self.assertEqual(converted[0]["context"][start:start + len("25 kg urea")], "25 kg urea")

    def test_smoke_rows_take_the_lowest_id_per_language(self):
        rows = list(reversed(native_rows(3)))
        smoke = frozen.smoke_rows(rows)
        self.assertEqual([row["language"] for row in smoke], list(LANGS))
        self.assertTrue(all(row["id"].endswith("000000") for row in smoke))

    def test_dataset_checks_fail_closed(self):
        rows = native_rows()
        with TemporaryDirectory() as directory:
            path, _ = step1(Path(directory), rows)
            _, loaded, record = frozen.load_native(path)
            self.assertEqual((len(loaded), record["languages"], record["human_reviewed_records"]),
                             (12, {lang: 3 for lang in LANGS}, 0))
        for change, message in (({"sha": "0" * 64}, "hash mismatch"), ({"flags": True}, "overlap audit"),
                                ({"records": 13}, "record count")):
            with TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, message):
                frozen.load_native(step1(Path(directory), rows, **change)[0])


class CheckpointTests(unittest.TestCase):
    def make_checkpoint(self, directory):
        directory.mkdir()
        for name, content in (("config.json", b"{}"), ("model.safetensors", b"weights")):
            (directory / name).write_bytes(content)
        files = {name: sha256(directory / name) for name in ("config.json", "model.safetensors")}
        provenance = {"seed": 42, "base_model": "base", "base_revision": "rev", "training_data": {"kind": "fixture"},
                      "experiment": "phase_5_alpha_0", "computational_statistics": {"selected_epoch_checkpoint": "checkpoint-896"},
                      "checkpoint_file_sha256": files}
        write_json(directory / "phase3_checkpoint.json", provenance)
        spec = {"model": "phase_5_alpha_0", "role": "diagnostic", "provenance_sha256": sha256(directory / "phase3_checkpoint.json"),
                "seed": 42, "base_model": "base", "base_revision": "rev", "training_data": {"kind": "fixture"},
                "selected_epoch_checkpoint": "checkpoint-896", "file_sha256": files}
        return spec

    def test_diagnostic_checkpoint_needs_exact_files_and_provenance(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_checkpoint"
            spec = self.make_checkpoint(checkpoint)
            frozen.verify_checkpoint(spec, checkpoint)
            (checkpoint / "extra.bin").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "extra.bin"):
                frozen.verify_checkpoint(spec, checkpoint)
            (checkpoint / "extra.bin").unlink()
            (checkpoint / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "model.safetensors"):
                frozen.verify_checkpoint(spec, checkpoint)
            with self.assertRaisesRegex(ValueError, "provenance"):
                frozen.verify_checkpoint({**spec, "provenance_sha256": "0" * 64}, checkpoint)

    def test_primary_manifest_entry_must_match_recorded_phase3_provenance(self):
        recorded = json.loads((ROOT / "phase_03_multilingual_finetuning/results/phase_3a/run_manifest.json")
                              .read_text(encoding="utf-8"))["checkpoint"]
        entry = {"private_server_path": "outputs/x", "selected_epoch_checkpoint": "checkpoint-500",
                 "file_sha256": recorded["checkpoint_file_sha256"],
                 **{field: recorded[field] for field in frozen.PROVENANCE_FIELDS}}
        spec = frozen.checkpoint_spec({"checkpoints": {"phase_3a": entry}}, "phase_3a")
        self.assertEqual(spec["file_sha256"]["model.safetensors"],
                         "3d350d925798c5f67c27ee4f07a46f1149549b583cfcd5ac885fcba74ea1deea")
        for change in ({"selected_epoch_checkpoint": "checkpoint-448"}, {"seed": 7},
                       {"file_sha256": {**entry["file_sha256"], "model.safetensors": "0" * 64}}):
            with self.assertRaisesRegex(ValueError, "disagrees"):
                frozen.checkpoint_spec({"checkpoints": {"phase_3a": {**entry, **change}}}, "phase_3a")
        with self.assertRaisesRegex(ValueError, "no checkpoint entry"):
            frozen.checkpoint_spec({"checkpoints": {}}, "phase_3b")
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            frozen.checkpoint_spec({"checkpoints": {}}, "phase_9")

    def test_pinned_diagnostics_match_the_phase5_summary(self):
        pinned = json.loads((ROOT / "phase_07_comparative_evaluation/configs/checkpoints.json").read_text(encoding="utf-8"))
        summary = json.loads((ROOT / "phase_05_domain_adaptation/results/alpha_sweep_summary.json").read_text(encoding="utf-8"))
        recorded = {row["alpha"]: row["checkpoint_model_sha256"] for row in summary["validation"]}
        for model, alpha in (("phase_5_alpha_0", 0.0), ("phase_5_alpha_1", 1.0)):
            self.assertEqual(pinned["models"][model]["file_sha256"]["model.safetensors"], recorded[alpha])
        self.assertEqual(pinned["native_adaptation_diagnostic"]["declared"], "include")


class PredictionTests(unittest.TestCase):
    def test_predictions_require_full_coverage_and_matching_offsets(self):
        rows = frozen.model_rows(native_rows(1))
        stats_ = {row["id"]: {"windows": 1, "positive_windows": 1, "context_tokens": 20, "best_gold_token_length": 70}
                  for row in rows}
        decoded = {row["id"]: {"predicted_answer": "25 kg urea", "predicted_start": row["context"].index("25"),
                               "predicted_end": row["context"].index("25") + 10, "span_score": 1.0} for row in rows}
        predictions, details = frozen.build_predictions(rows, decoded, stats_)
        self.assertEqual([set(p) for p in predictions], [{"id", "language", "prediction_text"}] * 4)
        self.assertTrue(all(detail["gold_exceeds_decode_limit"] for detail in details))
        with self.assertRaisesRegex(ValueError, "Missing or extra"):
            frozen.build_predictions(rows, {k: v for k, v in list(decoded.items())[1:]}, stats_)
        broken = dict(decoded)
        broken[rows[0]["id"]] = {**decoded[rows[0]["id"]], "predicted_start": 0}
        with self.assertRaisesRegex(ValueError, "offsets"):
            frozen.build_predictions(rows, broken, stats_)


class StatisticsTests(unittest.TestCase):
    def test_document_bootstrap_pairs_models_within_language(self):
        gold = native_rows(4, documents=2)
        control = predictions(gold, lambda row: row["id"].endswith(("0", "1")))
        candidate = predictions(gold, lambda row: True)
        rows = {(r["language"], r["metric"]): r for r in stats.native_paired(gold, control, candidate, 500, 0.95, 42)}
        self.assertAlmostEqual(rows[("hi", "em")]["delta_pp"], 50.0)
        self.assertAlmostEqual(rows[("macro", "f1")]["candidate"], 100.0)
        self.assertEqual((rows[("macro", "em")]["documents"], rows[("macro", "em")]["questions"]), (8, 16))
        self.assertLessEqual(rows[("hi", "em")]["lower_pp"], 50.0)
        same = {(r["language"], r["metric"]): r for r in stats.native_paired(gold, control, control, 200, 0.95, 42)}
        self.assertEqual({(r["delta_pp"], r["lower_pp"], r["upper_pp"]) for r in same.values()}, {(0.0, 0.0, 0.0)})
        with self.assertRaises(ValueError):
            stats.native_paired(gold, control, candidate[1:], 10, 0.95, 42)

    def write_run(self, directory, gold, correct):
        directory.mkdir()
        write_jsonl(directory / "predictions.jsonl", predictions(gold, correct))
        write_jsonl(directory / "predictions_detail.jsonl", [
            {"id": row["id"], "gold_exceeds_decode_limit": False, "windows": 2} for row in gold])

    def test_error_sample_is_reproducible_and_counts_cases(self):
        gold = native_rows(4)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_run(root / "a", gold, lambda row: row["id"].endswith(("0", "1")))
            self.write_run(root / "b", gold, lambda row: row["id"].endswith(("1", "2")))
            counts, sample = stats.error_sample(gold, root / "a", root / "b", 1, 42)
            again = stats.error_sample(gold, root / "a", root / "b", 1, 42)
        self.assertEqual((counts, sample), again)
        self.assertEqual({k: counts["hi"][k] for k in ("both_correct", "both_wrong", "candidate_recovered",
                                                       "candidate_newly_missed")},
                         {"both_correct": 1, "both_wrong": 1, "candidate_recovered": 1, "candidate_newly_missed": 1})
        self.assertTrue(all(record["gold_has_numeral"] and record["multi_window_passage"] for record in sample))

    def test_report_and_plots_from_a_synthetic_run(self):
        gold = native_rows(4)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run, step = root / "run", root / "step1"
            step.mkdir()
            manifest_path, _ = step1(step, gold)
            _, _, dataset = frozen.load_native(manifest_path)
            (run / "analysis").mkdir(parents=True)
            for model, correct in (("phase_3a", lambda r: r["id"].endswith("0")), ("phase_3b", lambda r: True)):
                (run / model).mkdir()
                write_jsonl(run / model / "predictions.jsonl", predictions(gold, correct))
                write_jsonl(run / model / "predictions_detail.jsonl", [
                    {"id": r["id"], "gold_exceeds_decode_limit": False, "windows": 1} for r in gold])
                report = evaluate(gold, predictions(gold, correct))
                report["input_sha256"] = {"gold": dataset["sha256"], "predictions": sha256(run / model / "predictions.jsonl")}
                write_json(run / model / "metrics.json", report)
                write_json(run / model / "run_manifest.json", {
                    "smoke": False, "dataset": dataset, "predictions_sha256": sha256(run / model / "predictions.jsonl"),
                    "role": "primary", "label": model, "checkpoint": {}, "batch_size": 8, "device": "cpu", "source": None,
                    "command": [], "environment": {}, "prediction_count": len(gold), "long_gold_spans": 0, "decoding": frozen.DECODING,
                    "evaluated_languages": {lang: 4 for lang in LANGS}})
            stats.cmd_native(SimpleNamespace(gold=step / "phase6_test.shared.jsonl", control=run / "phase_3a/predictions.jsonl",
                                             candidate=run / "phase_3b/predictions.jsonl", control_label="phase_3a",
                                             candidate_label="phase_3b", iterations=100, seed=42,
                                             output=run / "analysis/native_phase_3b_vs_phase_3a.json"))
            stats.cmd_errors(SimpleNamespace(gold=step / "phase6_test.shared.jsonl", control_run=run / "phase_3a",
                                             candidate_run=run / "phase_3b", per_case=2, seed=42,
                                             private_output=run / "analysis/native_error_sample.private.csv",
                                             counts_output=run / "analysis/native_error_counts.json"))
            with mock.patch.object(stats, "bootstrap_settings", return_value=(200, 0.95, 42)):
                stats.cmd_translated(SimpleNamespace(output=run / "analysis/translated.json"))
            stats.cmd_report(SimpleNamespace(run=run, output_dir=root / "report"))
            tables = (root / "report/comparison_tables.md").read_text(encoding="utf-8")
            self.assertIn("55.15 | 78.89", tables)
            self.assertIn("| Hindi | 4 | 1 | 0 | 3 | 0 |", tables)
            public = "".join(path.read_text(encoding="utf-8") for path in (root / "report").glob("*.*"))
            self.assertNotIn("passage 0", public)
            if importlib.util.find_spec("matplotlib"):
                from phase_07_comparative_evaluation import generate_plots
                with mock.patch("sys.argv", ["generate_plots", "--results", str(root / "report"),
                                             "--output-dir", str(root / "plots")]):
                    generate_plots.main()
                self.assertEqual(len(list((root / "plots").glob("*.png"))), 3)


if __name__ == "__main__":
    unittest.main()
