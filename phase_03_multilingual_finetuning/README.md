# Phase 3 — Multilingual Fine-Tuning and Cross-Lingual Evaluation

**Owner:** Saunak
**Status:** The Phase 2 input checks are implemented. The reviewed release, English XLM-R checkpoint and model execution remain pending.

## Research objective
Compare 3A (English training, all-language evaluation) against 3B (five-language training, all-language evaluation), using identical held-out question groups.

## Inputs and outputs
- Inputs: Validated phase 2 splits.
- Deliverables: Versioned XLM-R checkpoints, predictions and per-language metrics.
- Shared contracts: [schemas](../schemas/README.md), [evaluation protocol](../docs/evaluation.md).

## Verify Phase 3 inputs

From the repository root, check the same frozen test set for both experiments.

```sh
python3 phase_03_multilingual_finetuning/train.py --experiment 3a
python3 phase_03_multilingual_finetuning/train.py --experiment 3b
```

Both commands verify the [Phase 2 version and split hashes](../phase_02_multilingual_benchmark/configs/qa_release.json). The preflight selects English train and validation rows for 3A and all five languages for 3B. 3A requires the Phase 1 English-only XLM-R QA checkpoint and its training provenance. 3B starts from the original pretrained XLM-R Base. Both use the same five-language test. The commands report data integrity and checkpoint requirements without executing the experiments.

Use `--variant final` once human review has produced the final build. Use `--variant document_disjoint` to inspect the separate sensitivity dataset. For that comparison, 3A needs an English checkpoint trained on the disjoint subset. The original English-only checkpoint retains the primary protocol's inherited passage overlap.

## Completion checklist
- [ ] Resolve input paths, source permissions, dataset/model revisions and configuration.
- [ ] Implement and review phase-specific experiment steps.
- [ ] Validate IDs, answer offsets and split/document leakage.
- [ ] Record commands, environment, seed, hardware and artifact hashes.
- [ ] Save predictions and actual results with review coverage and limitations.
- [ ] Hand off deliverables to downstream owners; never label a dry run as complete.
