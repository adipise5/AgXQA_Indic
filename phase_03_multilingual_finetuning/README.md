# Phase 3 — Multilingual Fine-Tuning and Cross-Lingual Evaluation

**Owner:** Saunak

## 1. Research Objective

Compare English-only XLM-R transfer (3A) with five-language supervised training (3B) on the same agricultural QA questions. Phase 3B supplies the QA baseline for Phase 5 domain adaptation.

## 2. Research Questions

Measure English-to-Indic transfer, five-language fine-tuning gains, which languages lose the most F1, how much the English–Indic gap changes, question-wise consistency, and whether multilingual training changes English performance.

## 3. Experimental Hypotheses

Multilingual supervision may improve Indic F1 and joint exact match and reduce the gap with English. The effect on English and the relative difficulty of the four target languages are empirical questions.

## 4. Phase 1 and Phase 2 Dependencies

3A requires the English-only XLM-R QA checkpoint and training provenance from Phase 1. The repository currently records that checkpoint as pending. Both experiments require the identical reviewed Phase 2 release for primary results. The provisional `gemini-v2` subset is available for data and tokenizer checks. [Phase 1 checkpoint status](../phase_01_original_baseline/checkpoints/README.md) · [Phase 2 methodology](../phase_02_multilingual_benchmark/README.md)

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

## 12. Evaluation Metrics

Use `agxqa-indic-v1` for all five languages. NFC normalization, case folding, Unicode punctuation separation and whitespace tokenization preserve combining marks. Each metric takes its best gold-reference score independently. The English-only SQuAD v1 score is reported separately. [Shared scoring protocol](../docs/evaluation.md)

## 13. Experimental Results

The first Phase 3B run used seed 42 on the local span-checked bundle, MPS, batch size 2, and gradient accumulation 8. It stopped after two optimizer steps because of memory pressure. The log is preserved locally. No checkpoint or model scores were produced.

The capped MPS retry exceeded the 7.70 GiB allocation limit before the first optimizer step. The four-thread CPU pilot completed all five steps in 26 seconds. The full CPU run uses the same low-memory configuration and the span-checked bundle. No validation or test scores are available yet. [Mac training commands](docs/methodology.md#memory-capped-mac-pilot)

## 14. Language-Wise Performance Comparison

The reporters emit per-language EM/F1, counts, Indic macro, overall macro and 3B-minus-3A gains in percentage points after the two runs.

## 15. Cross-Lingual Performance Gap

For each Indic language, subtract its F1 from English F1. Report the difference in this gap from 3A to 3B.

## 16. Cross-Lingual Joint Correctness

Use complete parallel question IDs. Report the distribution of groups with 0–5 exact matches and pairwise language correctness agreement.

## 17. Statistical Significance

Run 10,000 paired bootstrap replicates with seed 42 and percentile 95% intervals. Resample whole question groups by default. The original English passage and cited document are also available as cluster units. Report seed variation separately with the seed summary if independent runs are available.

## 18. Error Analysis

Sample up to 50 matched erroneous question groups per Indic language and include both predictions for each. Reviewers code terminology, span boundary, numerical, semantic, long-context, morphology, translation, distractor, tokenization and partial-answer errors. Automated slices use context length, token count, answer length, source category and review status.

## 19. Computational Efficiency

Record train rows and features, nonpadding training tokens, model parameters, optimizer steps, wall time, environment, device and CUDA peak allocated memory. Inference timing covers batched model execution and conversion of logits to CPU, with a per-question-row average. MPS peak memory remains unmeasured by these scripts.

## 20. Research Findings

The integrity and tokenizer audits are complete for the provisional cohort. Cross-lingual performance differences remain unmeasured.

## 21. Limitations

The primary split retains inherited English passage reuse. The separate document-disjoint dataset supports an unseen-passage analysis with separately trained checkpoints. Equal epochs do not equate to equal optimizer updates between 3A and 3B. The current exports also differ in English and Indic reference counts.

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

## 23. Model Checkpoint Links

The English and multilingual XLM-R checkpoint links are pending. Refer to the checkpoint directory guide for the local format and provenance fields. [Checkpoint guide](checkpoints/README.md)

## 24. References

- XLM-RoBERTa Base pinned model assets. [Hugging Face](https://huggingface.co/FacebookAI/xlm-roberta-base/tree/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089)
- Extractive QA tokenization and span-label conventions. [Transformers documentation](https://huggingface.co/docs/transformers/main/tasks/question_answering)

## 25. Contributors

Saunak owns Phase 3. Dhiraj supplies the Phase 1 English checkpoint and uses the final comparisons in Phase 7. Aditya Pise uses the 3B configuration and frozen evaluation artifacts for Phase 5.
