"""Phase 5 adaptation-strength sweep invariants, using no pretrained model."""

import csv
import importlib.util
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from phase_03_multilingual_finetuning.common import LANGUAGES, read_json, write_json, write_jsonl
from phase_03_multilingual_finetuning.scoring import aggregate
from phase_05_domain_adaptation import alpha_sweep
from shared.qa_release import sha256

HAS_TORCH = importlib.util.find_spec("torch") is not None


def predictions(seed, quality):
    rng = random.Random(seed)
    rows = []
    for group in range(6):
        for language in LANGUAGES:
            exact = int(rng.random() < quality)
            rows.append({"id": f"{group}_{language}", "original_id": str(group), "language": language,
                         "experiment": "fixture", "question": f"q{group}", "gold_answers": ["a"],
                         "predicted_answer": "a", "predicted_start": rng.randrange(5), "predicted_end": 9,
                         "exact_match": exact, "f1": 1.0 if exact else rng.random() / 2,
                         "source_document_id": f"doc{group // 2}", "english_context_sha256": f"c{group}"})
    return rows


def make_run(root, name, alpha, seed, quality):
    run = root / name
    checkpoint = run / "training" / "best_checkpoint"
    checkpoint.mkdir(parents=True)
    record = {"experiment": f"phase_5_alpha_{alpha:g}", "seed": seed, "training_configuration": {"lr": 3e-5},
              "computational_statistics": {"selected_epoch_checkpoint": "checkpoint-896", "device_name": "fixture",
                                           "platform": "fixture", "versions": {"torch": "x"}},
              "checkpoint_file_sha256": {"model.safetensors": f"{name}-weights"},
              "adapted_initialization": {"alpha": alpha, "initial_layer_digest": f"{name}-layer"}}
    write_json(checkpoint / "phase3_checkpoint.json", record)
    for split, offset in (("validation", 0), ("test", 1000)):
        directory = run / split
        directory.mkdir()
        rows = predictions(seed + offset + int(alpha * 10), quality)
        write_jsonl(directory / "predictions.jsonl", rows)
        scores = aggregate(rows)
        write_json(directory / "metrics.json", {"by_language_and_macro": scores})
        write_json(directory / "run_manifest.json", {
            "checkpoint": record, "dataset": {"variant": "fixture"},
            "output_file_sha256": {"predictions.jsonl": sha256(directory / "predictions.jsonl")}})
        if split == "validation":
            with (run / "training" / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("seed", "epoch", "step", "language", "em", "f1"))
                for language in LANGUAGES:
                    writer.writerow((seed, 2.0, 896, language, scores[language]["exact_match"], scores[language]["f1"]))
    return run


def summary(alpha, english, indic):
    return {"alpha": alpha, "metrics": {"en": {"f1": english}, "indic_macro": {"f1": indic}}}


class SelectionRuleTests(unittest.TestCase):
    control = summary(0.0, 70.0, 65.0)

    def test_candidate_with_lower_or_equal_indic_f1_is_rejected(self):
        equal_indic, lower_indic = summary(0.5, 72.0, 65.0), summary(1.0, 72.0, 64.9)
        self.assertFalse(alpha_sweep.qualifies(self.control, equal_indic))
        self.assertFalse(alpha_sweep.qualifies(self.control, lower_indic))
        self.assertIsNone(alpha_sweep.choose(self.control, [equal_indic, lower_indic]))

    def test_candidate_with_lower_english_f1_is_rejected(self):
        lower_english = summary(0.5, 69.99, 70.0)
        self.assertFalse(alpha_sweep.qualifies(self.control, lower_english))
        self.assertIsNone(alpha_sweep.choose(self.control, [lower_english]))

    def test_qualifying_candidate_is_selected(self):
        qualifying = summary(0.5, 70.0, 65.01)  # English equal to the control is enough.
        lower_english = summary(1.0, 69.0, 80.0)
        self.assertTrue(alpha_sweep.qualifies(self.control, qualifying))
        self.assertIs(alpha_sweep.choose(self.control, [lower_english, qualifying]), qualifying)
        better = summary(0.25, 71.0, 66.0)
        self.assertIs(alpha_sweep.choose(self.control, [qualifying, better]), better)

    def test_indic_tie_goes_to_the_smaller_alpha(self):
        tie_low, tie_high = summary(0.25, 71.0, 66.0), summary(0.75, 72.0, 66.0)
        self.assertIs(alpha_sweep.choose(self.control, [tie_high, tie_low]), tie_low)
        self.assertIs(alpha_sweep.choose(self.control, [tie_low, tie_high]), tie_low)

    def test_alpha_of_runs_recorded_before_alpha_existed(self):
        self.assertEqual(alpha_sweep.run_alpha({"adapted_initialization": {
            "method": "indic_aec_mlm_final_encoder_layer"}}), 1.0)
        self.assertEqual(alpha_sweep.run_alpha({"adapted_initialization": {"alpha": 0.5}}), 0.5)
        with self.assertRaises(ValueError):
            alpha_sweep.run_alpha({})

    def test_select_and_compare_pair_control_with_frozen_candidate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            control = make_run(root, "alpha0-seed42", 0.0, 42, 0.5)
            half = make_run(root, "alpha0.5-seed42", 0.5, 42, 0.6)
            selection = root / "selection.json"
            alpha_sweep.cmd_select(SimpleNamespace(control=control, candidates=[half], output=selection))
            chosen = read_json(selection)
            self.assertEqual(chosen["control"]["trainer_validation_max_abs_difference"], 0)
            entry = chosen["candidates"][0]
            self.assertEqual(entry["eligible"],
                             entry["english_f1_at_least_control"] and entry["indic_macro_f1_above_control"])
            expected = half if entry["eligible"] else None
            self.assertEqual(chosen["selected_run"], expected and expected.as_posix())
            self.assertEqual(chosen["decision"], "adapted_candidate" if expected else "keep_alpha_0_control")
            if expected is None:
                half = make_run(root, "alpha0.5-forced", 0.5, 42, 0.6)
                chosen.update(selected_alpha=0.5, selected_run=half.as_posix())
                selection.unlink()
                write_json(selection, chosen)
            with self.assertRaises(ValueError):
                alpha_sweep.cmd_test(SimpleNamespace(run=make_run(root, "alpha1-seed42", 1.0, 42, 0.5),
                                                     selection=selection))
            controls = [control, make_run(root, "alpha0-seed123", 0.0, 123, 0.5)]
            candidates = [half, make_run(root, "alpha0.5-seed123", 0.5, 123, 0.6)]
            reference = root / "reference"
            reference.mkdir()
            write_jsonl(reference / "predictions.jsonl", predictions(1042, 0.5))
            write_json(reference / "computational_statistics.json", {"training": {"device_name": "NVIDIA L4"}})
            with mock.patch.object(alpha_sweep, "REFERENCE", reference):
                alpha_sweep.cmd_compare(SimpleNamespace(selection=selection, control_runs=controls,
                                                        candidate_runs=candidates, output=root / "comparison"))
            result = read_json(root / "comparison" / "comparison.json")
            self.assertEqual([pair["split"] for pair in result["paired"]], ["validation"] * 2 + ["test"] * 2)
            self.assertEqual(len(result["seed_stability"]["test"]["indic_macro_f1"]["per_seed"]), 2)
            self.assertTrue(result["alpha_0_reproduces_phase_3b"]["identical_spans"] in (True, False))
            self.assertEqual({row["split"] for row in result["table"]}, {"validation", "test"})
            self.assertIn("Δ Indic EM / F1", (root / "comparison" / "comparison_table.md").read_text(encoding="utf-8"))


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class InterpolationTests(unittest.TestCase):
    def setUp(self):
        import torch

        generator = torch.Generator().manual_seed(0)
        self.base = {"weight": torch.randn(64, 64, generator=generator), "bias": torch.randn(64, generator=generator)}
        self.adapted = {name: tensor + 1e-3 * torch.randn(tensor.shape, generator=generator) + 7.0
                        for name, tensor in self.base.items()}

    def test_endpoints_are_exact_and_midpoint_follows_the_formula(self):
        import torch

        from phase_05_domain_adaptation.run_matched_qa import interpolate_layer

        zero, one = interpolate_layer(self.base, self.adapted, 0), interpolate_layer(self.base, self.adapted, 1)
        self.assertTrue(all(torch.equal(zero[n], self.base[n]) for n in self.base))
        self.assertTrue(all(torch.equal(one[n], self.adapted[n]) for n in self.base))
        naive = {n: self.base[n] + 1.0 * (self.adapted[n] - self.base[n]) for n in self.base}
        self.assertFalse(all(torch.equal(naive[n], self.adapted[n]) for n in self.base),
                         "fixture should show why the alpha=1 endpoint is special-cased")
        half = interpolate_layer(self.base, self.adapted, 0.5)
        for name in self.base:
            expected = (self.base[name].double() + 0.5 * (self.adapted[name].double() - self.base[name].double()))
            self.assertTrue(torch.equal(half[name], expected.float()))
            self.assertIsNot(zero[name], self.base[name])
        with self.assertRaises(ValueError):
            interpolate_layer(self.base, self.adapted, 1.5)
        with self.assertRaises(ValueError):
            interpolate_layer(self.base, {"weight": self.adapted["weight"]}, 0.5)

    def test_initializer_changes_only_the_final_layer(self):
        import torch

        from phase_05_domain_adaptation.run_matched_qa import adapted_initializer, tensor_digest

        torch.manual_seed(1)
        layers = torch.nn.ModuleList([torch.nn.Linear(64, 64), torch.nn.Linear(64, 64)])
        layers[-1].load_state_dict(self.base)
        model = SimpleNamespace(roberta=SimpleNamespace(encoder=SimpleNamespace(layer=layers)))
        first = {n: t.clone() for n, t in layers[0].state_dict().items()}
        record = adapted_initializer(self.adapted, 0.5, {"method": "fixture"})(model)
        self.assertTrue(all(torch.equal(layers[0].state_dict()[n], t) for n, t in first.items()))
        self.assertEqual(record["alpha"], 0.5)
        self.assertEqual(record["base_layer_digest"], tensor_digest(self.base))
        self.assertEqual(record["initial_layer_digest"], tensor_digest(layers[-1].state_dict()))
        self.assertNotEqual(record["initial_layer_digest"], record["base_layer_digest"])


if __name__ == "__main__":
    unittest.main()
