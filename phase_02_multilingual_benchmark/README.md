# Phase 2 — AgXQA-Indic Benchmark Construction

**Owner:** Saunak. Phase 2 uses existing Gemini 3.5 Flash-Lite translations into Hindi, Marathi, Bengali and Tamil. The local `gemini-v2` bundle was rebuilt from the three byte-for-byte `gemini-v1` exports and the recovered validation evidence. The final benchmark awaits human review and English source adjudication. [Draft dataset card](../data/agxqa_indic/dataset_card.md) · [Measured data audit](results/gemini_v2_audit.json)

## Export audit

| Export | Rows | IDs common to languages present | Rows with at least one invalid offset |
|---|---:|---:|---:|
| `train_flat_multilingual_clean.jsonl` | 6,520 | 1,304 | 0 |
| `fixed_tval_flat.jsonl` | 1,399 | 347 | 135 |
| `test_flat_multilingual.jsonl` | 1,635 | 327 | 132 |

All exported IDs match the pinned English source split. The audit found no duplicate row IDs. Train and test include English and all four translated languages. Validation contains 349 Bengali rows and 350 each for Hindi, Marathi and Tamil. Its 347 shared IDs can be paired with English rows from the pinned source for a parallel comparison.

Checking the exact stored passages found invalid offsets in 135 validation and 132 test rows. Normalizing both passage and answer to NFC makes 131 of those validation rows and 122 of those test rows match at their recorded offsets. Audit the remaining rows individually and version each correction with its source ID and reason.

The original test exports reproduce the study's all-reference substring percentages to one decimal place.

| Language | Raw text | After NFC | Valid stored offsets before repair |
|---|---:|---:|---:|
| English | 98.5% | 98.5% | 322/327 |
| Hindi | 99.7% | 100.0% | 324/327 |
| Marathi | 99.7% | 99.7% | 325/327 |
| Bengali | 87.2% | 100.0% | 206/327 |
| Tamil | 100.0% | 100.0% | 326/327 |

The test exports carry 433 English references. Each Indic language carries only 327 references, one per question. That is 106 fewer than the English source across 58 multi-answer questions per language. The count comparison leaves semantic equivalence for human review.

## Provenance and exclusions

All 353 parsed Gemini validation responses were recovered with the original aligned and repair outputs. The [recovered validation script settings and prompt](configs/gemini_provenance.json) record the requested model alias, JSON response type, four-second successful-request delay, and ten-second exception delay. The script continues after exceptions and declares no retry or model fallback. Server-resolved model versions, sampling defaults, execution logs and original train/test response files remain unavailable. The draft's train/test settings are historical claims with incomplete run evidence.

The reduced benchmark keeps 1,304 train, 347 validation and 327 test groups for comparability with the reported counts. The exclusion ledger records 199 train, 6 validation and 3 test IDs. Train and test exclusions are absent from the exports, so their individual original rejection reasons remain unknown. All six omitted validation IDs have all four languages in the recovered raw responses. All official source rows are labeled Irrigation, so a labeled subtopic comparison is unavailable. Among 39 training citation groups with at least ten questions, exclusion rates range from 0% to 33.3%. The five pending English test groups are tracked within the 327 retained groups, separately from the 208 source exclusions.

## Prepared dataset

The importer kept the 347 validation IDs shared by all four translated languages and logged 11 unmatched validation export rows. It sources every English row directly from the pinned release, including the 139 train records whose supplied answers were trimmed in the Gemini flat export. It normalizes translated passages and answers to NFC, maps verified positions after trimming boundary whitespace, and accepts unique exact or whitespace-equivalent spans. One Marathi answer was copied with its original joiner intact. Every emitted answer is an exact substring at its stored offset. Translated rows keep their exported reference counts.

| Span-checked subset | Question IDs | Rows |
|---|---:|---:|
| Train | 1,304 | 6,520 |
| Validation | 347 | 1,735 |
| Test | 322 | 1,610 |

The five English test records `1170477`, `1174212`, `1183012`, `1181597`, and `481` need source adjudication. Their nine invalid references are inherited from the official English test data. All five languages for those IDs remain in the candidate set and are held out of the span-checked test subset. The final build targets all 327 retained test groups and 9,890 total rows. The provisional test subset should be reported as 322 matched groups if used.

The official English train–test overlap is 309 of 330 exact passages by test row. For the retained export IDs, it is 300 of 327, and for the span-checked test it is 295 of 322. In the span-checked test, 318 of 322 groups share a source document group with train. Use the unchanged official split membership for baseline comparability. Claims about unseen passages need the separate document-disjoint comparison.

The document-disjoint subset keeps the same provisional test IDs, removes conflicting official validation and train groups, and assigns 36 questions from surviving train document components to validation using seed 42. It contains 306 train, 36 validation and 322 test groups with no shared normalized-passage or cited-document keys across splits. Performance on this subset remains unmeasured. Phase 3A needs an English checkpoint trained on this disjoint subset to support that comparison.

## Rebuild the local bundle

From the AgXQA-Indic checkout, use Python 3.10 or later and a fresh output directory.

```sh
phase=phase_02_multilingual_benchmark
bundle="$phase/outputs/gemini-v2"
python3 "$phase/scripts/import_gemini_exports.py" \
  --source-dir ../../data --export-dir "$phase/outputs/gemini-v1/inputs" \
  --validation-evidence-dir "$HOME/json_script" --output-dir "$bundle"
python3 -m shared.validate "$bundle/span_checked/shared/train.jsonl"
python3 -m shared.validate "$bundle/span_checked/shared/validation.jsonl"
python3 -m shared.validate "$bundle/span_checked/shared/test.jsonl"
```

The bundle contains original exports in `inputs/`, recovered responses and provenance in `evidence/`, flat and parallel splits under `span_checked/` and `document_disjoint/`, and retained IDs, corrections, exclusions, reports and hashes at the bundle root. Reviewers use the packets in `review_material/` and the matching CSVs in `human_review/`. To rebuild from a copy of the bundle, use that copy's `inputs/` and `evidence/` directories and a fresh output path.

The Hugging Face source card displays an MIT dataset tag. The source documentation reviewed lacks passage-by-passage permission records for the original extension material and its translations. Google's output terms leave responsibility for use with the user. Redistribution clearance for the complete bundle remains unverified. The text-bearing artifacts and evidence are stored locally. [AgXQA source card](https://huggingface.co/datasets/msu-ceco/agxqa_v1) · [Gemini API terms](https://ai.google.dev/gemini-api/terms)

After review, run the finalizer with the same source files and review policy. It reads the bundle's worksheets by default and accepts `--review-dir` for a separate set of completed worksheets. It writes `finalization_report.json` and creates `final/` only when all required reviews and all spans pass.

```sh
python3 "$phase/scripts/finalize_gemini_exports.py" \
  --source-dir ../../data --bundle-dir "$bundle"
```

## Source audit

From the AgXQA-Indic checkout, point `--source-dir` at the directory containing the original English JSONL files. The following uses the downloaded files in the Agroberta workspace and writes into a fresh directory.

```sh
phase=phase_02_multilingual_benchmark
audit_dir=$(mktemp -d)
python3 "$phase/scripts/prepare_source.py" \
  --source-dir ../../data \
  --output-dir "$audit_dir/source" \
  --results-dir "$audit_dir"
```

Use `--download` with a designated source directory if the pinned English files are unavailable. Source preparation verifies the pinned hashes and reports split counts, invalid source offsets and cross-split overlap. Python 3.10+ is sufficient for the audit. Optional language identification, YAML and plots use [the Phase 2 dependencies](requirements.txt).

## Human review

Review the source and translated context, question and answer together. Check terminology, numerical values, semantic equivalence and answer extractability. The local worksheets contain 601 Hindi cases for Aditya Saxena, 564 Marathi cases for Aditya Pise, 568 Bengali cases for Saunak, 742 Tamil cases for Dhiraj, and 5 English source cases. The counts include all retained Indic test rows, the seeded train and validation samples, and automated flags. No completed judgments have been supplied. [Annotation guidelines](human_review/annotation_guidelines.md)
