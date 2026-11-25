# Phase 5 — Domain Adaptation Using Indic-AEC

**Owner:** Aditya Pise  
**Status:** Complete, with seed 42 on the provisional test set. **The sweep selected α 0, and the project retains the recorded Phase 3B model.**

## Objective

Phase 5 tests whether masked-language-model (MLM) adaptation on the reviewed Indic-AEC agricultural corpus improves Indic QA over Phase 3B while preserving English. The adaptation updates the final encoder layer of the pinned XLM-R Base. Each QA model then trains under the exact Phase 3B recipe and is compared, paired, on the same test questions.

## Final result

| Decision | Outcome |
|---|---|
| Sweep selection (validation only) | **α 0.** No adapted candidate kept English validation F1 at least equal to the control's while raising Indic macro validation F1 above it. |
| Retained model | **The recorded Phase 3B model**: Indic macro F1 64.87, English F1 71.64. |
| Matched control | The **α-0 rerun** of the Phase 3B recipe, on the same GPU type and seed. It is used only for paired comparisons and is not reported as a new model. |
| Decoder span cap | Stays at **64 tokens**. Raising it to 160 changed validation F1 by −0.01 (Indic) and +0.01 (overall). |

Key test numbers, as F1 with 95% intervals from resampling source documents:

| Comparison | English F1 | Indic macro F1 |
|---|---|---|
| Recorded Phase 3B | 71.64 | 64.87 |
| α-0 rerun − recorded 3B: how much an identical rerun varies | −0.78 [−3.97, +2.53] | +0.29 [−2.18, +2.35] |
| Full adaptation (α 1) − α-0 control, context only | −2.43 [−4.82, −0.04] | −0.20 [−1.66, +1.52] |

**Findings:**
- **Adaptation did not improve Indic QA.** Its effect is within the variation between identical reruns.
- **Full adaptation costs some English**: −4.55 on validation and −2.43 on test, from a single seed.
- **The adapted layer barely moved.** It differs from the base by at most 5.9e-4 per weight, after one epoch over six excerpts. A real test of domain adaptation needs a substantially larger MLM run.

Full tables, covering validation and test for every run, per language, plus the span-cap check: [results/comparison_table.md](results/comparison_table.md). Hashes and intervals: [results/alpha_sweep_summary.json](results/alpha_sweep_summary.json).

## Layout

Checked in (code and the results needed to understand or rerun the experiment):

```
phase_05_domain_adaptation/
├── README.md                 # Objective, run commands and final result
├── config.json
├── prepare_mlm_data.py       # Prepare Indic-AEC MLM inputs
├── train_mlm.py              # Train the adapted final encoder layer
├── finetune_qa.py            # Verify QA data and audit overlap
├── run_matched_qa.py         # Train and evaluate under the Phase 3B recipe (--alpha)
├── alpha_sweep.py            # Validation-selected adaptation-strength sweep
├── hpc/run_alpha_sweep.sh    # Run the whole sweep on the GPU server
└── results/
    ├── mlm_input_manifest.json         # Input counts and hashes
    ├── mlm_pilot_metrics.json          # Measured masked-token loss
    ├── earlier_mlm_pilot_metrics.json  # Earlier corpus pilot, kept for the record
    ├── phase3_span_checked_overlap.json
    ├── alpha_sweep_summary.json        # Aggregate scores, selection and hashes
    └── comparison_table.md             # Results for readers
tests/test_phase05_alpha_sweep.py
results/status.json                     # Phase 5 complete, Phase 3B retained
```

Private run artifacts stay in the git-ignored `outputs/` folder on the server:

```
outputs/phase_05_domain_adaptation/
├── mlm_runs/<run-id>/        # final_encoder_layer.safetensors, metrics.json
├── matched-qa-seed42/        # first matched run (α 1), referenced by the sweep
└── alpha-sweep/
    ├── preflight.json, selection.json, span_cap_validation.json
    ├── comparison/comparison.json
    ├── alpha0-seed42/, alpha0.5-seed42/
    ├── logs/
    └── alpha-sweep-results.tar         # weight-free archive to copy back
```

The recorded pilot's run-id is `3f3481d6662a2b69-len256-last-layer-1ep-lr2e-05-seed42-cpu`. Its weights have SHA-256 `1e7122873ee756397f1dc1dbcf6a4e788a429967124c6fc7d9af12ff53e7cd76`. Raw inputs, predictions and weights stay private. Prediction redistribution follows the dataset's source terms.

## Run commands

Run every command from the repository root.

**1. MLM adaptation (CPU is fine).** This needs the verified Phase 4 package, its pinned tokenizer cache, and a local directory with the pinned base weights and config. `train_mlm.py` writes `outputs/phase_05_domain_adaptation/mlm_runs/<run-id>/`.

```sh
python phase_05_domain_adaptation/prepare_mlm_data.py \
  --corpus-release /path/to/releases/3f3481d6662a2b69 \
  --tokenizer-cache /path/to/tokenizer_cache

HF_HUB_OFFLINE=1 python phase_05_domain_adaptation/train_mlm.py \
  --inputs outputs/phase_05_domain_adaptation/mlm_inputs/3f3481d6662a2b69-len256 \
  --cache-dir /path/to/tokenizer_cache \
  --model-dir /path/to/pinned/xlm-roberta-base \
  --device cpu --epochs 1
```

The overlap audit binds the recorded pilot's checkpoint hash. A reproduced checkpoint with a different hash needs its own provenance and a separately labelled comparison.

**2. Single matched QA run (CUDA, about 30 minutes on an L4).** Use the Phase 3 environment and budget about 12 GiB for checkpoints.
- `--alpha` defaults to 1, which loads the adapted layer unchanged.
- Add `--offline` when the pinned base model is already in the Hugging Face cache.
- QA data defaults to the Phase 3 bundle at `phase_02_multilingual_benchmark/outputs/gemini-v2`. Pass `--bundle-dir` for another location, or `--export-dir` with `--source-dir` to rebuild the splits from the three original Gemini exports. The rebuild must reproduce the same split SHA-256s.

```sh
run=outputs/phase_05_domain_adaptation/mlm_runs/3f3481d6662a2b69-len256-last-layer-1ep-lr2e-05-seed42-cpu
python -m phase_05_domain_adaptation.run_matched_qa --adaptation-dir $run \
  --output outputs/phase_05_domain_adaptation/matched-qa-seed42 --offline --check-only
python -m phase_05_domain_adaptation.run_matched_qa --adaptation-dir $run \
  --output outputs/phase_05_domain_adaptation/matched-qa-seed42 --offline
```

**3. Adaptation-strength sweep (GPU server).** This takes about 1 hour when no candidate qualifies, or about 3 hours when one does. Completed steps are skipped on rerun.

```sh
setsid nohup bash phase_05_domain_adaptation/hpc/run_alpha_sweep.sh \
  > outputs/phase_05_domain_adaptation/alpha-sweep-driver.log 2>&1 < /dev/null &
```

The script's defaults match the layout above. Override them with these environment variables:

| Variable | Sets |
|---|---|
| `OUT` | Sweep output folder |
| `ADAPTATION_DIR` | Adaptation run folder |
| `ALPHA1_RUN` | Existing α-1 run |
| `SWEEP_ALPHA` | Alpha to try alongside the control (default 0.5) |
| `EXTRA_SEEDS` | Paired seeds for a selected candidate (default 123 and 2026) |
| `LARGER_CAP` | Span-cap comparison value (default 160) |
| `BUNDLE_DIR` | QA bundle location |

The individual steps are `python -m phase_05_domain_adaptation.alpha_sweep {check,train,validate,select,span-cap,test,compare}`.

## Method

**Initialization.** Only the final encoder layer changes: `initial = base + α × (adapted − base)`.
- α 0 keeps the pinned base tensors bit for bit.
- α 1 loads the saved adapted tensors exactly. Both endpoints are special-cased because floating-point `base + (adapted − base)` need not equal `adapted`.
- The seeded QA head and every other tensor are identical across runs. `alpha_sweep.py check` confirms this on the real model.
- Each run records the base, adapted and initial layer digests in its checkpoint provenance.

**Fixed across runs.** Everything except the initial final layer:
- Phase 3B data and split checksums;
- tokenizer and `configs/cuda_hpc.yaml`;
- validation macro-F1 checkpoint selection;
- decoder settings and the `agxqa-indic-v1` scorer;
- seed 42 and one GPU.

**Selection on validation.**
- **Arms:** α 0 (the control) and α 0.5 are trained. The existing α-1 run is scored on validation as a third candidate.
- **Rule:** an adapted candidate qualifies only if its English validation F1 is at least the control's **and** its Indic macro validation F1 is strictly above it. Among qualifying candidates, the highest Indic F1 wins, and ties go to the smaller α. The result is frozen in `selection.json`.
- **If one qualifies,** it and the control are repeated with paired seeds 123 and 2026.
- **If none qualifies,** α 0 is selected and no extra adaptation runs are made.
- **Test** runs only after selection. The CLI refuses any run other than the control and the frozen candidate. Paired intervals resample source documents.
- **Context:** the α-1 test scores were seen before the sweep, so they are reported only as context.

**Decoder span cap (separate check).** 35 of 1,735 validation rows (2.0%) have gold answers longer than the 64-token cap; the longest is 154 tokens. The check re-decodes the α-0 control's saved validation scores at caps 64 and 160. It first confirms that cap 64 reproduces the saved predictions. This check is not part of the adaptation sweep.

## Measured MLM and overlap work

**MLM pilot.**
- **Run:** updated the final encoder layer of pinned XLM-R Base on reviewed Indic-AEC release `3f3481d6662a2b69`, with seed 42, 39 training sequences and 77 validation sequences.
- **Result:** on the same 2,948 masked validation positions, loss moved from **1.745622** to **1.738547** after one epoch. Reloading the saved layer reproduced the final loss.
- [Pilot metrics](results/mlm_pilot_metrics.json) · [Input manifest](results/mlm_input_manifest.json) · [Earlier corpus pilot](results/earlier_mlm_pilot_metrics.json)

**Split verification.** The three Gemini v2 exports plus the pinned English source reproduced all three Phase 3 split checksums.

**Overlap audit.**
- **Literal overlap:** the six exact training excerpts produced zero word 5-gram or character 10-gram flags, at thresholds 0.30, 0.40 and 0.50. They were checked against **3,370** retained Phase 2 validation and test candidates and **1,960** Phase 6 reference passages and windows. The Phase 2 count includes the five provisional test groups held out of scoring.
- **Phase 6 reservations:** the reservation scan produced zero hits.
- **Caveat:** four reconstructed Native source pages differ from their original HTML snapshots.
- These checks cover literal reuse only. [Text-free overlap audit](results/phase3_span_checked_overlap.json)

[Phase 4 instructions](../phase_04_indic_aec_corpus/README.md) · [Phase 3 instructions](../phase_03_multilingual_finetuning/README.md)
