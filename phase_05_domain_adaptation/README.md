# Phase 5 — Domain Adaptation Using Indic-AEC

**Owner:** Aditya Pise
**Status:** The QA dataset preflight is implemented. The adaptation corpus and verified Phase 6 context reconstruction are required to run the overlap audit.

## Research objective
Continue XLM-R masked-language pretraining on Indic-AEC, then fine-tune QA with the same data and settings as 3B. Exclude native test documents before adaptation.

## Inputs and outputs
- Inputs: Phase 4 training corpus and phase 3 QA training configuration.
- Deliverables: MLM and QA checkpoints, losses, predictions and controlled comparison.
- Shared contracts: [schemas](../schemas/README.md), [evaluation protocol](../docs/evaluation.md).

## Verify Phase 5 inputs

From the repository root, verify the split hashes shared with Phase 3 and audit the exact text released to MLM.

```sh
python3 phase_05_domain_adaptation/finetune_qa.py \
  --corpus path/to/indic-aec-train.jsonl \
  --phase6-contexts outputs/phase6_audit/data/processed_contexts \
  --output outputs/phase_05_domain_adaptation/overlap-audit.json
```

The audit checks Phase 6 reserved URLs and hashes and normalized word 5-gram overlap against Phase 2 validation and test passages and all verified Phase 6 contexts and reconstructed document windows. It checks all retained held-out Phase 2 groups, including pending English rows and any revised evaluation passages. Missing inputs or overlap produce an unsuccessful audit. The earlier Hindi pilot report covers a different, smaller corpus. [Phase 4 dependencies and reconstruction instructions](../phase_04_indic_aec_corpus/README.md)

Use `--variant final` for the reviewed evaluation set when it exists. This preflight does not execute adaptation or QA training.

## Completion checklist
- [ ] Resolve input paths, source permissions, dataset/model revisions and configuration.
- [ ] Implement and review phase-specific experiment steps.
- [ ] Validate IDs, answer offsets and split/document leakage.
- [ ] Record commands, environment, seed, hardware and artifact hashes.
- [ ] Save predictions and actual results with review coverage and limitations.
- [ ] Hand off deliverables to downstream owners; never label a dry run as complete.
