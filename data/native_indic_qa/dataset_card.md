# IndicAgriQA-Native: dataset card

The full dataset card lives with the Phase 6 code: [`phase_06_native_indic_qa/docs/dataset_card.md`](../../phase_06_native_indic_qa/docs/dataset_card.md).

This folder contains the release:

- `final/*.jsonl`: 2,351 QA pairs. Each has its question, answers, offsets, `context_id` and `context_sha256`, but no passage text.
- `metadata.csv`: per-ID metadata.

To rebuild the full files with passages into `full/` (git-ignored), run `phase_06_native_indic_qa/scripts/hydrate_dataset.py` (see the Phase 6 README, §24).
