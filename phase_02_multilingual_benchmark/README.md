# Phase 2 — AgXQA-Indic Benchmark Construction

**Owner:** Saunak. Phase 2 uses the existing Gemini 3.5 Flash-Lite translations into Hindi, Marathi, Bengali and Tamil. The local `gemini-v1` bundle contains the preserved inputs, span-checked splits, and human-review worksheets. The final benchmark is awaiting human review and English source adjudication. [Draft dataset card](../data/agxqa_indic/dataset_card.md) · [Pinned source audit](results/source_preflight.json)

## Export audit

| Export | Rows | IDs common to languages present | Rows with at least one invalid offset |
|---|---:|---:|---:|
| `train_flat_multilingual_clean.jsonl` | 6,520 | 1,304 | 0 |
| `fixed_tval_flat.jsonl` | 1,399 | 347 | 135 |
| `test_flat_multilingual.jsonl` | 1,635 | 327 | 132 |

All exported IDs match the pinned English source split. The audit found no duplicate row IDs. Train and test include English and all four translated languages. Validation contains 349 Bengali rows and 350 each for Hindi, Marathi and Tamil. Its 347 shared IDs can be paired with English rows from the pinned source for a parallel comparison.

Checking the exact stored passages found invalid offsets in 135 validation and 132 test rows. Normalizing both passage and answer to NFC makes 131 of those validation rows and 122 of those test rows match at their recorded offsets. Audit the remaining rows individually and version each correction with its source ID and reason.

## Prepared dataset

The importer kept the 347 validation IDs shared by all four translated languages and added their English rows. It logged 11 unmatched validation export rows. It normalized translated passages and answers to NFC, mapped confirmed offsets, and used unique exact matches for shifted spans. One Marathi answer was matched ignoring joiners and then copied from the passage with its joiner intact. The correction log records the original and resulting spans. English annotations continue to use their supplied reference counts and translated rows keep their exported reference counts.

| Span-checked subset | Question IDs | Rows |
|---|---:|---:|
| Train | 1,304 | 6,520 |
| Validation | 347 | 1,735 |
| Test | 322 | 1,610 |

The five English test records `1170477`, `1174212`, `1183012`, `1181597`, and `481` need source adjudication. Their nine invalid references are inherited from the official English test data. All five languages for those IDs remain in the candidate set and are held out of the span-checked test subset. The final build targets all 327 retained test groups and 9,890 total rows. The provisional test subset should be reported as 322 matched groups if used.

The manifest records zero cross-split question IDs, 39 cross-split reference-document groups, and 205 exact-context groups across the span-checked exports. The original split membership is preserved, so these splits measure performance with inherited context overlap. Match review corrections to the candidate and source hashes. Preserve the original English annotations used for Phase 1.

## Rebuild the local bundle

From the AgXQA-Indic checkout, use Python 3.10 or later and a fresh output directory.

```sh
phase=phase_02_multilingual_benchmark
bundle="$phase/outputs/gemini-v1"
python3 "$phase/scripts/import_gemini_exports.py" \
  --source-dir ../../data --export-dir "$HOME/Downloads" --output-dir "$bundle"
python3 -m shared.validate "$bundle/span_checked/shared/train.jsonl"
python3 -m shared.validate "$bundle/span_checked/shared/validation.jsonl"
python3 -m shared.validate "$bundle/span_checked/shared/test.jsonl"
```

The bundle contains byte-for-byte originals under `inputs/`, flat and parallel data under `span_checked/`, `candidates.jsonl`, `corrections.jsonl`, `validation_exclusions.jsonl`, and file hashes in `manifest.json`. Reviewers use the language-specific packets under `review_material/` and return the matching CSVs from `human_review/`.

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
