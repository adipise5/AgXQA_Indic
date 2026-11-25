# Phase 7 deliverables

The deliverables are the frozen-checkpoint evaluator, the aggregate comparison, and enough provenance to reproduce the scores from the private run. **Phase 3B is the selected model.** The Native results are provisional: **0 of 2,351 questions are human-reviewed** in this frozen dataset.

## Deliverables

| Deliverable | Where |
|---|---|
| Reproducible workflow | [evaluate_frozen.py](evaluate_frozen.py), [configs/checkpoints.json](configs/checkpoints.json), [statistical_analysis.py](statistical_analysis.py), [generate_plots.py](generate_plots.py), [hpc/run_phase7.sh](hpc/run_phase7.sh), [tests/test_phase07_native.py](../tests/test_phase07_native.py). The README has the [GPU commands](README.md#run) and the [report rebuild commands](README.md#rebuild-the-report-from-saved-predictions). |
| English and translated results | [comparison_tables.md §1–2](results/comparison_tables.md#1-original-english-agxqa). English is its own table, scored with `squad_v1_english`. The translated section has per-language EM and F1, Indic macro, paired 3B − 3A with document intervals, the English–Indic gap and joint correctness, each with its n = 322 denominator. Data: [translated.json](results/translated.json). |
| Native results | [comparison_tables.md §3–4](results/comparison_tables.md#3-native-indicagriqa-provisional). 3A and recorded 3B on the same 2,351 frozen rows: Hindi 1,107, Marathi 581, Bengali 420 and Tamil 243, from 70 documents. Each language shows EM, F1 and its question and document counts; the Indic macro is equal-weight, and paired differences come with 95% intervals. Data: [native_scores.json](results/native_scores.json), [native_phase_3b_vs_phase_3a.json](results/native_phase_3b_vs_phase_3a.json). |
| Phase 5 diagnostic | Secondary Native comparisons, both checkpoint-896, seed 42: α 1 vs its α-0 control ([JSON](results/native_phase_5_alpha_1_vs_alpha_0.json)), and the α-0 rerun vs recorded 3B ([JSON](results/native_phase_5_alpha_0_vs_phase_3b.json)). |
| Figures and error summary | [Four figures](results/plots/), linked from [§5](results/comparison_tables.md#5-figures). [§6](results/comparison_tables.md#6-native-error-summary-3a-vs-3b-exact-match) has the paired outcome counts, sampling coverage and per-category automated flags, labelled provisional. Reviewer status: 0 of 289 sampled cases. Data: [error_counts.json](results/error_counts.json). |
| Reproducibility manifest | [phase7_manifest.json](results/phase7_manifest.json) records the following, with no passage text, predictions, account names or server paths:<ul><li>the run revision and report-generation revision, with per-file git blob IDs and the files changed since the run;</li><li>the dataset and checkpoint hashes, and the prediction, metrics and paired-analysis hashes;</li><li>row counts per language, the scorers and bootstrap settings;</li><li>review coverage;</li><li>repository-relative commands.</li></ul> |

Native intervals pair predictions by `(id, language)` and resample source documents within each language, using the same draws for both models: 10,000 draws, 95% percentile interval, seed 42. They describe uncertainty over source documents for these fixed checkpoints, not variation between training runs.

## Private run artifacts (git-ignored)

```text
outputs/phase_07_comparative_evaluation/
├── step1/                                   # phase6_test.shared.jsonl, manifest.json, native_overlap_audit.json
└── runs/
    ├── native-20261125T202902Z.tar          # private transfer archive
    └── native-20261125T202902Z/             # predictions, per-example scores, logs, native_error_sample.private.csv
```

The full model checkpoints stay in their Phase 3 and Phase 5 locations on the GPU server. Share the frozen passages, predictions, the private error sample and the raw logs only through the team's private storage.

## Acceptance checklist

- [x] **Inputs match the pins.** The frozen dataset (`7ff7cc15…6336`), overlap audit (`requires_overlap_review: false`) and all four checkpoint directories were verified against the pinned hashes on the server before inference. The Step 1 manifest entries for 3A and 3B match the committed Phase 3 provenance.
- [x] **Every reported run is complete and reproducible.**
  - Each of the four runs covers the same record IDs and all four language counts (1,107 / 581 / 420 / 243).
  - Scores were recomputed from the saved predictions and match exactly.
  - The paired intervals were recomputed and match exactly.
  - Tables and figures were regenerated from the saved files.
- [x] **Links and status are recorded.** The README links the results and records Phase 3B as the selected model. `results/status.json` records the completed evaluation and the pending reviewer decisions.
- [x] **Review coverage is reported.** 0 of 2,351 Native questions are human-reviewed. `native_error_sample.private.csv` has 0 of 289 sampled cases reviewed, so the error categories are labelled automated and provisional.
- [x] **Tests pass and the staged files are clean.** The Phase 7 tests pass, and the staged files were checked for private artifacts: no passage text, predictions, account names or absolute paths. The runner, tests, report, figures, aggregate JSON and status updates are staged together on `research/phase7-native-evaluation`.

## Pending reviewer decisions

- Human review of the 2,351 frozen Native questions. Scores stay provisional until then.
- Reviewer categories in `native_error_sample.private.csv`, for alternative phrasing, units, numerals and span boundaries. Only reviewer-confirmed categories should be reported as such.
