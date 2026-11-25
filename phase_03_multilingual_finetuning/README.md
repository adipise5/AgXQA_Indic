# Phase 3 — Multilingual Fine-Tuning and Cross-Lingual Evaluation

**Owner:** Saunak

## 1. Research Objective

Compare English-only XLM-R transfer (3A) with five-language supervised training (3B) on the same agricultural QA questions. Phase 3B supplies the QA baseline for Phase 5 domain adaptation.

## 2. Research Questions

Measure English-to-Indic transfer, five-language fine-tuning gains, which languages lose the most F1, how much the English–Indic gap changes, question-wise consistency, and whether multilingual training changes English performance.

## 3. Experimental Hypotheses

Multilingual supervision may improve Indic F1 and joint exact match and reduce the gap with English. The effect on English and the relative difficulty of the four target languages are empirical questions.

## 4. Phase 1 and Phase 2 Dependencies

Phase 1 did not train an English XLM-R QA checkpoint, so 3A uses the planned fallback. XLM-R Base is fine-tuned here on the pinned official English train split (1,503 questions) and selected on English validation (353 questions). Both experiments use the provisional `gemini-v2` `span_checked` release. The reviewed Phase 2 release is still pending, so every result below is provisional. [Phase 1 checkpoint status](../phase_01_original_baseline/checkpoints/README.md) · [Phase 2 methodology](../phase_02_multilingual_benchmark/README.md)

## 5. Dataset Description

| Provisional split | Question groups | Rows |
|---|---:|---:|
| Train | 1,304 | 6,520 |
| Validation | 347 | 1,735 |
| Test | 322 | 1,610 |

Each group contains English, Hindi, Marathi, Bengali, and Tamil. The final test targets 327 groups once the five English source cases and native reviews are resolved.

## 6. Dataset Integrity Validation

All three files match their pinned checksums. The audit verifies exact spans, exactly one row per language and question group, and zero cross-split question IDs. It records inherited exact passage overlap separately. [Dataset integrity report](results/dataset_integrity_report.json) · [Evaluation contract](../phase_02_multilingual_benchmark/configs/qa_release.json)

## 7. Model Architecture

Both experiments use XLM-RoBERTa Base with an extractive start/end QA head. Base model and tokenizer are pinned to `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`. The scripts verify architecture and checkpoint file hashes, and reject missing trained QA parameters during evaluation. [Pinned XLM-R revision](https://huggingface.co/FacebookAI/xlm-roberta-base/tree/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089)

## 8. Tokenization and Preprocessing

Use the fast tokenizer, context-only truncation, 384 tokens, 128-token stride and dynamic right padding. Use the first encodable gold reference in each window and CLS labels where none is fully contained. Decode only context spans across all windows using 20 start and end candidates and a 64-token maximum answer. [Tokenizer audit](results/tokenization_statistics.json)

The provisional audit measured 7,163 train, 1,825 validation and 1,701 test windows. Every row has an encodable answer in at least one window. On test, 38 rows have all encodable gold spans longer than the decode cap. This limits exact-span recovery for those examples.

## 9. Experiment 3A — Zero-Shot Cross-Lingual Transfer

Reuse a verified Phase 1 English-only XLM-R QA checkpoint and evaluate all five languages directly. Training and checkpoint selection use English only. The fallback script supports the pinned official English train and validation data if the checkpoint is unavailable. A document-disjoint comparison requires a separate English run on the disjoint subset.

## 10. Experiment 3B — Multilingual Fine-Tuning

Initialize from the original pinned XLM-R Base weights. Train all model parameters on the five-language train split and choose the best epoch by validation macro-F1. Keep test evaluation separate. Each language has equal row counts in the current cohort, so the seeded shuffle uses the full split without rebalancing.

## 11. Training Configuration

The proposed defaults are five epochs, learning rate 3e-5, batch size 8, gradient accumulation 2, weight decay 0.01, warmup ratio 0.1, linear schedule, AdamW and gradient clipping 1.0. Save epoch checkpoints with a limit of two and retain the best. Training events are logged every 20 steps and validation metrics at each epoch. Use the verified Phase 1 settings as the reference for controlled comparison. [Multilingual config](configs/multilingual_config.yaml)

The measured 3A and 3B runs used batch size 1 with gradient accumulation 16, which keeps the effective batch of 16, and no gradient checkpointing. All other settings are the defaults above. [CUDA override](configs/cuda_hpc.yaml)

## 12. Evaluation Metrics

Use `agxqa-indic-v1` for all five languages. NFC normalization, case folding, Unicode punctuation separation and whitespace tokenization preserve combining marks. Each metric takes its best gold-reference score independently. The English-only SQuAD v1 score is reported separately. [Shared scoring protocol](../docs/evaluation.md)

## 13. Experimental Results

Both experiments ran once, with seed 42, on 25 November 2026. They ran on one NVIDIA L4 (24 GB) with PyTorch 2.8.0+cu128 and Transformers 4.57.6, using `hpc/run_phase3.sh`. The whole pipeline took 33 minutes. 3B selected epoch 4 (step 1,792) at validation macro-F1 68.48. 3A selected its final epoch 5 (step 500) at English validation F1 70.13, or 69.33 under SQuAD scoring. Both evaluations cover the same 322 test question groups in five languages (1,610 rows). [Exported results](results/) · [Export manifest](results/export_manifest.json)

Earlier Mac MPS and CPU attempts stopped before their first epoch checkpoint and produced no scores. [Mac training notes](docs/methodology.md#memory-capped-mac-pilot)

## 14. Language-Wise Performance Comparison

Scores use `agxqa-indic-v1`. Δ F1 is 3B minus 3A in percentage points.

| Language | 3A EM | 3A F1 | 3B EM | 3B F1 | Δ F1 |
|---|---:|---:|---:|---:|---:|
| English | 36.96 | 66.79 | 43.48 | 71.64 | +4.86 |
| Hindi | 27.95 | 51.51 | 37.89 | 65.94 | +14.43 |
| Marathi | 26.71 | 52.57 | 35.71 | 63.91 | +11.34 |
| Bengali | 25.47 | 53.16 | 37.27 | 66.48 | +13.32 |
| Tamil | 25.78 | 50.42 | 32.92 | 63.17 | +12.74 |
| Indic macro | 26.48 | 51.92 | 35.95 | 64.87 | +12.96 |
| Overall macro | 28.57 | 54.89 | 37.45 | 66.23 | +11.34 |

Under SQuAD v1 scoring, English is 38.82 EM / 66.35 F1 for 3A and 45.65 EM / 71.43 F1 for 3B. [Comparative results](results/comparative_results.csv) · [3A metrics](results/phase_3a/metrics.json) · [3B metrics](results/phase_3b/metrics.json)

## 15. Cross-Lingual Performance Gap

Each gap is English F1 minus the language's F1.

| Language | 3A gap | 3B gap | Gap reduction |
|---|---:|---:|---:|
| Hindi | 15.27 | 5.70 | 9.57 |
| Marathi | 14.22 | 7.74 | 6.48 |
| Bengali | 13.63 | 5.16 | 8.46 |
| Tamil | 16.36 | 8.48 | 7.89 |

[Performance gap](results/performance_gap.csv)

## 16. Cross-Lingual Joint Correctness

Each count is the number of the 322 question groups answered with an exact match in that many languages.

| Exact in | 3A | 3B |
|---|---:|---:|
| All five languages | 22 (6.8%) | 46 (14.3%) |
| Four | 27 (8.4%) | 36 (11.2%) |
| Three | 31 (9.6%) | 32 (9.9%) |
| Two | 40 (12.4%) | 37 (11.5%) |
| One | 69 (21.4%) | 59 (18.3%) |
| None | 133 (41.3%) | 112 (34.8%) |

[Joint correctness](results/cross_lingual_consistency.csv) · [Pairwise language agreement](results/pairwise_agreement.json)

## 17. Statistical Significance

The analysis uses 10,000 paired bootstrap replicates with seed 42 and percentile 95% intervals. Whole question groups are resampled, so the five translations of a question stay together.

| Language | Δ F1 | 95% interval | Δ EM | 95% interval |
|---|---:|---|---:|---|
| English | +4.86 | [1.29, 8.48] | +6.52 | [1.86, 11.49] |
| Hindi | +14.43 | [10.00, 19.04] | +9.94 | [4.04, 15.84] |
| Marathi | +11.34 | [6.44, 16.20] | +9.01 | [2.80, 15.22] |
| Bengali | +13.32 | [8.20, 18.35] | +11.80 | [5.59, 18.32] |
| Tamil | +12.74 | [8.15, 17.29] | +7.14 | [1.24, 13.04] |
| Indic macro | +12.96 | [9.40, 16.55] | +9.47 | [4.97, 13.90] |

Resampling by shared English passage gives nearly identical intervals, for example Indic macro F1 [9.46, 16.60]. Only one training seed was run, so the intervals cover test-sample uncertainty but not seed variation. [Question-group intervals](results/confidence_intervals.csv) · [Passage-cluster intervals](results/confidence_intervals_context.csv)

## 18. Error Analysis

Sample up to 50 matched erroneous question groups per Indic language and include both predictions for each. Reviewers code terminology, span boundary, numerical, semantic, long-context, morphology, translation, distractor, tokenization and partial-answer errors. Automated slices report matched EM and F1 by context length, token count, answer length, question type, numeric answer, source category and review status. Question type (numerical, temporal, causal, procedural, location, entity, conditional or factual) is a rule-based label from the English question and applies to all five translations. Every provisional test question has the category Irrigation and none is human-reviewed yet, so domain and raw-versus-reviewed comparisons are not yet possible.

The error worksheet holds 200 question–language pairs (50 per Indic language) with both models' predictions. Its `error_category`, `reviewer` and `comments` columns await human coding. Averaged over the four Indic languages, numerical questions gain the most among types with at least 20 groups (32 groups, 43.0 → 63.7 F1). Location questions gain the least (20 groups, 60.6 → 61.3). Most question-type groups have fewer than 35 questions and no intervals, so treat these slices as descriptive. [Error worksheet](results/error_analysis.csv) · [Matched slices](results/matched_slices.csv)

## 19. Computational Efficiency

| Parameter | 3A | 3B |
|---|---:|---:|
| Training examples | 1,503 | 6,520 |
| Tokenized training features | 1,594 | 7,163 |
| Model parameters | 277,454,594 | 277,454,594 |
| Optimizer steps | 500 | 2,240 |
| Training time | 5.7 min | 26.1 min |
| GPU | NVIDIA L4 | NVIDIA L4 |
| Peak GPU memory (training) | 5.19 GiB | 5.19 GiB |
| Inference per question row | 6.76 ms | 7.45 ms |

Inference timing covers batched model execution and conversion of logits to CPU. [Computational statistics](results/computational_statistics.json)

## 20. Research Findings

On this provisional cohort:

- English-only training transfers part of the way. The Indic languages reach 50.4–53.2 F1, 13.6–16.4 points below English.
- Five-language training raises F1 in every language. Indic F1 rises by 11.3–14.4 points, English by 4.9, and every 95% interval excludes zero.
- The English–Indic gap shrinks from 13.6–16.4 points to 5.2–8.5. Hindi gains the most and Marathi the least.
- Questions answered exactly in all five languages double from 22 to 46, and those answered in none fall from 133 to 112.
- Tamil has the lowest Indic F1 under both models (50.42 and 63.17) and the largest remaining gap after 3B (8.48), followed by Marathi (7.74).

## 21. Limitations

- **Unequal training budgets.** 3A made 500 optimizer updates and 3B made 2,240. 3A's English validation F1 was still rising at its final epoch (68.30 → 70.13), so part of the 3B advantage, especially on English, may reflect training budget rather than multilingual supervision. A compute-matched English baseline was not run.
- **One seed.** Variation across training seeds is unmeasured.
- **Provisional data.** The test set is the 322-group `span_checked` subset of machine translations awaiting human review. The final test targets 327 groups.
- **Passage overlap with training.** 295 of the 322 English test passages also appear in the training split. This overlap is inherited from the official AgXQA split. The separate document-disjoint subset supports an unseen-passage analysis with separately trained checkpoints, which has not been run.
- The exports differ in English and Indic reference counts.

## 22. Reproducibility Instructions

Run from the AgXQA-Indic checkout. The local Phase 3 environment uses Python 3.11. The lock records the resolved macOS arm64 packages.

```sh
phase=phase_03_multilingual_finetuning
out=outputs/phase_03_multilingual_finetuning
py="$phase/.venv/bin/python"
python3.11 -m venv "$phase/.venv"
"$py" -m pip install -r "$phase/requirements-lock.txt"
"$py" "$phase/train.py" --experiment 3a
"$py" "$phase/train.py" --experiment 3b
"$py" "$phase/scripts/validate_dataset.py" --output "$out/dataset_integrity_report.json"
"$py" "$phase/scripts/tokenize_dataset.py" --tokenizer-dir "$out/tokenizer-e73636d" \
  --output "$out/tokenization_statistics.json"
```

The tokenizer-only directory contains `config.json`, `tokenizer_config.json` and `tokenizer.json` from the pinned revision. The [methodology](docs/methodology.md) has the model-run commands, input requirements and reporting workflow.

On a Linux GPU server, `hpc/setup_env.sh` builds the environment and `hpc/run_phase3.sh` runs the whole seed in order: preflight, 3B training and test, 3A English training and zero-shot test, and the comparison. `hpc/slurm_phase3.sbatch` wraps it for SLURM. GPUs with at least 10 GB use `configs/cuda_hpc.yaml`, which is the same batch recipe without gradient checkpointing.

After a run, copy its artifacts into `results/` and open the notebooks in `notebooks/`:

```sh
run=cuda-span-checked-42
"$py" "$phase/scripts/export_results.py" --a "$out/phase_3a-$run" --b "$out/phase_3b-$run" \
  --comparison "$out/comparison-$run" --training-dir "$out/training-3b-$run" \
  --english-training-dir "$out/english-fallback-$run" --context-intervals "$out/comparison-$run-context-intervals.csv"
```

The export verifies both prediction files against their run manifests and writes `results/export_manifest.json` with each file's source and hash.

## 23. Model Checkpoint Links

Both checkpoints are stored on the training server and have not been uploaded, so no public links exist. Their weight hashes and full provenance are recorded in the exported run manifests. [Checkpoint guide](checkpoints/README.md) · [3A run manifest](results/phase_3a/run_manifest.json) · [3B run manifest](results/phase_3b/run_manifest.json)

## 24. References

- XLM-RoBERTa Base pinned model assets. [Hugging Face](https://huggingface.co/FacebookAI/xlm-roberta-base/tree/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089)
- Extractive QA tokenization and span-label conventions. [Transformers documentation](https://huggingface.co/docs/transformers/main/tasks/question_answering)

## 25. Contributors

Saunak owns Phase 3. Dhiraj supplies the Phase 1 English checkpoint and uses the final comparisons in Phase 7. Aditya Pise uses the 3B configuration and frozen evaluation artifacts for Phase 5.
