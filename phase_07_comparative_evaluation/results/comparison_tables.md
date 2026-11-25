# Phase 7 comparative results

Scores are percentages. All models are single runs with seed 42. **Phase 3B is the selected model.** Phase 5's sweep found no qualifying adaptation.

- **Native review coverage:** 0 of 2,351 questions human-reviewed. Native results are provisional.
- **Error-sample review:** 0 of 289 sampled cases have a reviewer decision. The error categories below are automated and provisional.
- **Translated release:** the Gemini v2 `span_checked` release is unreviewed (provisional).

## 1. Original English AgXQA

330 official test items, scored with `squad_v1_english` (SQuAD 1.1 normalization, EM and token-overlap F1).

| System | EM | F1 |
|---|---|---|
| Original AgRoBERTa (published) | 55.15 | 78.89 |
| Phase 1 reimplementation | 54.24 | 78.23 |

## 2. Translated AgXQA-Indic

322 aligned question groups in each of 5 languages (1,610 rows), from 39 source documents. Scored with `agxqa-indic-v1` (EM and token-overlap F1).

| System | English EM / F1 (n=322) | Hindi EM / F1 (n=322) | Marathi EM / F1 (n=322) | Bengali EM / F1 (n=322) | Tamil EM / F1 (n=322) | Indic macro EM / F1 |
|---|---|---|---|---|---|---|
| XLM-R English-only (3A, checkpoint-500, seed 42) | 36.96 / 66.79 | 27.95 / 51.51 | 26.71 / 52.57 | 25.47 / 53.16 | 25.78 / 50.42 | 26.48 / 51.92 |
| XLM-R multilingual (3B, checkpoint-1792, seed 42; selected) | 43.48 / 71.64 | 37.89 / 65.94 | 35.71 / 63.91 | 37.27 / 66.48 | 32.92 / 63.17 | 35.95 / 64.87 |

Paired 3B − 3A (95% intervals, 39 source documents resampled, 10,000 draws, seed 42):

| Metric | English | Hindi | Marathi | Bengali | Tamil | Indic macro |
|---|---|---|---|---|---|---|
| F1 | +4.86 [+1.43, +10.14] | +14.43 [+10.91, +17.90] | +11.34 [+6.21, +16.01] | +13.32 [+8.24, +19.78] | +12.74 [+7.71, +19.16] | +12.96 [+9.56, +17.07] |
| EM | +6.52 [+1.22, +14.29] | +9.94 [+5.08, +16.46] | +9.01 [+3.69, +15.04] | +11.80 [+4.80, +20.80] | +7.14 [+0.52, +15.68] | +9.47 [+5.18, +15.62] |

English − language F1 gap on the same 322 aligned groups, and joint correctness (a group counts if all five languages are exact):

| System | EN − Hindi | EN − Marathi | EN − Bengali | EN − Tamil | All five exact |
|---|---|---|---|---|---|
| XLM-R English-only (3A, checkpoint-500, seed 42) | 15.27 | 14.22 | 13.63 | 16.36 | 22 of 322 (6.83%) |
| XLM-R multilingual (3B, checkpoint-1792, seed 42; selected) | 5.70 | 7.74 | 5.16 | 8.48 | 46 of 322 (14.29%) |

Translated diagnostic, Phase 5 α 1 − α 0 (both checkpoint-896, seed 42; the α 1 test scores were seen before the Phase 5 sweep): F1 English -2.43 [-4.82, -0.04]; Indic macro -0.20 [-1.66, +1.52].

## 3. Native IndicAgriQA (provisional)

2,351 questions from 70 source documents and 826 passages, with identical frozen rows for every model. Scored with `agxqa-indic-v1`. Indic macro is the equal-weight mean of the four languages.

| System | Hindi EM / F1 (n=1,107, docs=21) | Marathi EM / F1 (n=581, docs=20) | Bengali EM / F1 (n=420, docs=18) | Tamil EM / F1 (n=243, docs=11) | Indic macro EM / F1 |
|---|---|---|---|---|---|
| XLM-R English-only (3A, checkpoint-500, seed 42) | 35.23 / 61.42 | 32.19 / 56.52 | 34.52 / 62.64 | 26.34 / 56.43 | 32.07 / 59.25 |
| XLM-R multilingual (3B, checkpoint-1792, seed 42; selected) | 46.61 / 74.17 | 48.54 / 72.80 | 52.14 / 78.21 | 46.09 / 72.70 | 48.35 / 74.47 |
| Phase 5 α 0 rerun of the 3B recipe (checkpoint-896, seed 42; diagnostic control) | 49.59 / 75.59 | 49.40 / 72.20 | 52.14 / 78.17 | 48.97 / 72.29 | 50.03 / 74.56 |
| Phase 5 α 1 Indic-AEC adapted (checkpoint-896, seed 42; diagnostic) | 47.61 / 73.48 | 46.30 / 69.52 | 54.52 / 75.65 | 44.44 / 68.77 | 48.22 / 71.85 |

## 4. Native paired differences

Pairs are matched by `(id, language)`. Source documents are resampled within each language with the same draws for both models (10,000 draws, 95% percentile interval, seed 42). The intervals describe uncertainty over source documents for these fixed checkpoints, not variation between training runs. The diagnostic rows are secondary.

| Comparison | Metric | Hindi | Marathi | Bengali | Tamil | Indic macro |
|---|---|---|---|---|---|---|
| 3B − 3A (primary) | F1 | +12.75 [+10.69, +15.62] | +16.28 [+13.72, +18.80] | +15.57 [+11.68, +20.06] | +16.27 [+12.10, +22.63] | +15.22 [+13.48, +17.34] |
| 3B − 3A (primary) | EM | +11.38 [+7.71, +15.62] | +16.35 [+13.20, +19.65] | +17.62 [+11.94, +24.09] | +19.75 [+12.11, +30.40] | +16.28 [+13.55, +19.61] |
| α 1 − α 0 (diagnostic) | F1 | -2.11 [-3.51, -0.68] | -2.68 [-4.69, -0.76] | -2.52 [-4.73, -0.72] | -3.52 [-6.49, -0.03] | -2.71 [-3.84, -1.55] |
| α 1 − α 0 (diagnostic) | EM | -1.99 [-4.40, -0.10] | -3.10 [-5.63, -0.94] | +2.38 [-1.65, +5.71] | -4.53 [-9.75, +0.00] | -1.81 [-3.68, -0.20] |
| α 0 rerun − recorded 3B (rerun variation) | F1 | +1.42 [+0.44, +2.30] | -0.60 [-2.41, +1.60] | -0.04 [-2.99, +2.60] | -0.41 [-1.96, +0.81] | +0.09 [-0.89, +1.04] |
| α 0 rerun − recorded 3B (rerun variation) | EM | +2.98 [-0.00, +5.85] | +0.86 [-1.71, +3.69] | +0.00 [-5.29, +4.87] | +2.88 [+0.00, +5.32] | +1.68 [-0.10, +3.38] |

## 5. Figures

- [Translated F1, 3A vs 3B](plots/translated_f1.png)
- [Native F1, 3A vs 3B](plots/native_f1.png)
- [Native 3B − 3A paired differences](plots/native_delta_3b_vs_3a.png)
- [Native diagnostic α 1 − α 0 paired differences](plots/native_delta_alpha1_vs_alpha0.png)

## 6. Native error summary (3A vs 3B, exact match)

Paired outcomes over all questions:

| Language | Questions | Both correct | Both wrong | 3B recovered | 3B newly missed |
|---|---|---|---|---|---|
| Hindi | 1107 | 264 | 465 | 252 | 126 |
| Marathi | 581 | 144 | 256 | 138 | 43 |
| Bengali | 420 | 117 | 173 | 102 | 28 |
| Tamil | 243 | 50 | 117 | 62 | 14 |

Sampling coverage: up to 25 cases per outcome and language, drawn with seed 42. Automated flags (**provisional**) describe the model that missed; for both-wrong cases, 3B. Review status: 0 of 289 sampled cases reviewed.

| Language | Outcome | Sampled / available | gold has numeral | partial overlap | prediction longer than gold | prediction inside gold | gold exceeds decode limit | multi window passage |
|---|---|---|---|---|---|---|---|---|
| Hindi | 3B recovered | 25 / 252 | 8 | 19 | 3 | 15 | 0 | 3 |
| Hindi | 3B newly missed | 25 / 126 | 8 | 24 | 20 | 3 | 0 | 1 |
| Hindi | Both wrong | 25 / 465 | 10 | 23 | 18 | 3 | 0 | 2 |
| Marathi | 3B recovered | 25 / 138 | 12 | 20 | 7 | 13 | 0 | 8 |
| Marathi | 3B newly missed | 25 / 43 | 10 | 22 | 14 | 5 | 0 | 8 |
| Marathi | Both wrong | 25 / 256 | 13 | 20 | 14 | 3 | 0 | 5 |
| Bengali | 3B recovered | 25 / 102 | 11 | 20 | 8 | 10 | 0 | 5 |
| Bengali | 3B newly missed | 25 / 28 | 10 | 24 | 21 | 2 | 0 | 2 |
| Bengali | Both wrong | 25 / 173 | 6 | 20 | 14 | 8 | 0 | 2 |
| Tamil | 3B recovered | 25 / 62 | 12 | 20 | 13 | 7 | 0 | 7 |
| Tamil | 3B newly missed | 14 / 14 | 7 | 14 | 8 | 6 | 0 | 4 |
| Tamil | Both wrong | 25 / 117 | 6 | 20 | 9 | 9 | 0 | 8 |

Passage text and predictions stay in the private run folder.
