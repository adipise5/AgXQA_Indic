# Phase 5 results: matched QA and adaptation-strength sweep

All numbers come from [alpha_sweep_summary.json](alpha_sweep_summary.json), which also records the checkpoint, prediction and source-file hashes.

- **Setup:** seed 42; the recorded Phase 3B data, tokenizer and training configuration (`configs/cuda_hpc.yaml`); validation macro-F1 checkpoint selection; one NVIDIA L4. Package versions match Phase 3B.
- **Scoring:** `agxqa-indic-v1`, on the provisional gemini-v2 `span_checked` release.
- **Test set:** 322 question groups per language, from 39 source documents.
- **Initialization:** each run starts the final encoder layer at `base + α × (adapted − base)`, and every other tensor is identical across runs.

## Decisions

1. **The sweep selected α 0.** No adapted candidate had English validation F1 at least the control's and Indic macro validation F1 above it.
2. **The project retains the recorded Phase 3B model.** The α-0 rerun is the matched control for these comparisons. It is not a new reported model.
3. **The decoder span cap stays at 64 tokens.**

## Validation (selection split)

| Run | Checkpoint | EN F1 | HI F1 | MR F1 | BN F1 | TA F1 | Indic EM / F1 | Overall EM / F1 | Qualifies |
|---|---|---|---|---|---|---|---|---|---|
| α 0 (matched control) | epoch 2 | 74.24 | 69.75 | 67.11 | 68.63 | 67.74 | 39.99 / 68.31 | 41.90 / 69.50 | control |
| α 0.5 | epoch 3 | 74.14 | 67.95 | 64.69 | 67.22 | 67.53 | 39.63 / 66.85 | 41.50 / 68.30 | No: English and Indic below control |
| α 1 | epoch 2 | 69.69 | 68.40 | 68.22 | 67.49 | 70.40 | 43.16 / 68.63 | 43.05 / 68.84 | No: English below control |

## Test (evaluated after selection)

| Run | Checkpoint | EN F1 | HI F1 | MR F1 | BN F1 | TA F1 | Indic EM / F1 | Overall EM / F1 |
|---|---|---|---|---|---|---|---|---|
| Phase 3B, recorded (retained) | epoch 4 | 71.64 | 65.94 | 63.91 | 66.48 | 63.17 | 35.95 / 64.87 | 37.45 / 66.23 |
| α 0, matched control | epoch 2 | 70.86 | 67.13 | 65.35 | 65.11 | 63.05 | 35.40 / 65.16 | 36.71 / 66.30 |
| α 1, context only* | epoch 2 | 68.43 | 68.52 | 63.33 | 66.28 | 61.71 | 36.41 / 64.96 | 37.02 / 65.65 |

\* The α-1 test scores come from the first matched run, which was made before the sweep. They took no part in selection.

## Paired test differences (F1, percentage points, 95% CI from resampling source documents)

| Comparison | EN | HI | MR | BN | TA | Indic | Overall |
|---|---|---|---|---|---|---|---|
| α 0 rerun − recorded 3B | -0.78 [-3.97, +2.53] | +1.19 [-1.97, +3.91] | +1.45 [-1.68, +4.08] | -1.37 [-4.93, +1.71] | -0.11 [-2.79, +2.27] | +0.29 [-2.18, +2.35] | +0.07 [-2.14, +1.97] |
| α 1 − α 0 control (context) | -2.43 [-4.82, -0.04] | +1.40 [-1.01, +4.51] | -2.02 [-5.11, +0.50] | +1.16 [-2.53, +5.49] | -1.34 [-3.88, +1.12] | -0.20 [-1.66, +1.52] | -0.65 [-1.75, +0.63] |

The first row measures how much an identical rerun of the 3B recipe varies. The rerun selected a different epoch and produced different predictions.

## Decoder span cap (separate validation check on the α-0 control)

- **Long answers:** 35 of 1,735 validation rows have gold answers longer than 64 tokens. At 160 tokens, 0 do.
- **Effect of the larger cap:** 26 predictions change. Indic F1 moves by -0.01, and overall F1 by +0.01. The cap stays at 64.
