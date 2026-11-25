# Phase 4 — Indic Agricultural Extension Corpus

Owner: Aditya Pise. Status: reviewed four-language pilot release and submission evidence verified for `3f3481d6662a2b69`.
Main repository: https://github.com/adipise5/AgXQA_Indic
Baseline: e8754e957bcb8da6ddd9fd916e7820d8a334ecb8.

## Research submission

Indic-AEC `3f3481d6662a2b69` contains 12 reviewed native-language excerpts from eight publication families and four Indian public-institution publishers. The train and validation sets each contain six excerpts and cover Hindi, Marathi, Bengali, and Tamil. Training contains 9,120 XLM-R tokens and validation contains 18,911. The [dataset card](../data/indic_aec/dataset_card.md) gives the full breakdown, intended use and known limitations. The [submission audit](results/submission-audit.json) contains the per-source ledger, runtime versions, and cryptographic hashes.

The release verifier checks the corpus schema, exact source captures, source registry and reservations, review hashes, token counts, and split membership. The word 5-gram and whitespace-insensitive character 10-gram audits each compared 12 excerpts with 1,960 Phase 6 references. Both returned zero flags at pairwise or distributed containment thresholds of 0.30, 0.40 and 0.50. All 2,351 Phase 6 QA row hashes were verified against 827 unique context IDs. Full-page comparison used 1,133 windows from reconstructed pages, four of which differ from the original HTML hashes. The character detector flagged all four exact copies and all eight spacing and character-deletion controls at 0.50. These controls demonstrate behavior on those constructed inputs. The audit measures literal and character-level overlap. A preliminary check against 217 official English AgXQA held-out contexts also yielded zero word 5-gram flags. The retained multilingual Phase 2 evaluation passages and translated-copy checks remain pending for the final Phase 5 comparison.

The `submission` directory under `outputs/phase_04_indic_aec_corpus/` contains a metadata archive for the research appendix and a local reproduction archive with the corpus, saved captures, and held-out references. Reuse evidence applies individually. The two Hindi Samay extracts have unresolved public dataset redistribution terms. Use the metadata archive when sharing submission evidence without the corpus text. The verified pilot does not imply adaptation gains or comprehensive agricultural coverage.

To reproduce the report, extract the local archive and run from its root. Its paths match the original checkout. The lock records the execution environment on Python 3.14.7 and macOS arm64.

```sh
python -m pip install -r phase_04_indic_aec_corpus/requirements-submission-lock.txt
phase4=outputs/phase_04_indic_aec_corpus/india_book_acquisition/step_07_technical_training_candidates
HF_HUB_OFFLINE=1 python -m phase_04_indic_aec_corpus.audit_submission \
  --config "$phase4/config.json" \
  --release "$phase4/releases/3f3481d6662a2b69" \
  --english-source outputs/phase_04_indic_aec_corpus/submission/official_english \
  --rebuild --output verified-submission-audit.json
```

The audit re-extracts the saved captures and reproduces `corpus.jsonl`, `train.jsonl`, `validation.jsonl`, and `statistics.json` byte for byte in a temporary directory. Reconstructing captures from live pages is a separate operation. The `official_english` files are benchmark inputs and belong outside the corpus and MLM training inputs.

## Run the first milestone

From the repository root (Python 3.10+):

```sh
python -m pip install -r phase_04_indic_aec_corpus/requirements.txt
python phase_04_indic_aec_corpus/download_documents.py --dry-run
python -m unittest discover -s tests -v
```

The dry run writes an acquisition report under `outputs/phase_04_indic_aec_corpus/reports/` without making document requests. Ten ICAR Kheti articles have document-specific noncommercial reuse terms. Seven other source leads remain pending. See [the measured Hindi pilot](results/pilot.json) and [the source expansion](results/source-expansion.md) for extraction counts, input hashes and the Phase 6 audit.

## Source curation

`sources.csv` records document-level URLs. For each source:

- Record the applicable license and evidence in `permission_reference`; set `permission_status=approved` only after verifying permitted use.
- Verify native-language provenance, then set `language_origin=native_verified`.
- Use a unique `AEC_HI_*`, `AEC_MR_*`, `AEC_BN_*` or `AEC_TA_*` ID.
- Assign `document_family_id` across editions, translations and syndicated copies.
- Avoid all 84 Phase 6 reservations (including unused candidates), their mirrors, and all Vikaspedia sources in v1.

Marathi source catalog: https://mpkv.ac.in/Extension/ExtensionPublication — investigate Shri Sugi 2025/2026 issues and separate technical folders. Three MPKV document URLs are in the manifest pending reuse evidence. Krishidarshani 2025, the 2025 Marathi recommendations, and Shri Sugi Rabi 2024 are reserved. New editions require content-overlap review.

Pilot target: ten accepted documents per language, with multiple topics and publishers. Targets are not achieved counts.

## Acquisition and extraction

```sh
python phase_04_indic_aec_corpus/download_documents.py
python phase_04_indic_aec_corpus/extract_text.py
# For scanned PDFs, additionally install pytesseract/Pillow and Tesseract language packs:
python phase_04_indic_aec_corpus/extract_text.py --ocr
```

The downloader checks source URLs, redirects, raw hashes and similar reserved titles. It replaces the run manifest with successful acquisitions. Inspect failures before continuing. It limits each document to 150 MiB. Work products stay under git-ignored `outputs/`; none are published automatically.

Extraction reuses Phase 6 HTML/PDF/OCR and paragraph helpers without running Phase 6 CLI functions or writing its files. NFC and Indic vowel signs are preserved. Post-OCR language/garbling checks run again. Source metadata, paragraphs, page numbers and extraction methods survive in `documents.jsonl`.

Review each record against the source. Record its document ID, cleaned text SHA-256, decision, reviewer and notes in `quality_reviews.csv`. Check language, corruption, tables, units, numbers and completeness. A review is valid only for the exact text it reviewed.

## Held-out reference

```sh
python phase_04_indic_aec_corpus/rebuild_phase6_reference.py
python phase_04_indic_aec_corpus/pipeline.py audit
```

The reference builder reconstructs public Phase 6 sources under `outputs/phase6_audit/` and verifies each released passage hash. It compares the full downloaded HTML hashes with the original manifest and reports differences. Full-document comparison uses the reconstructed public pages. Sections outside the QA passages lack original text hashes, so the report cannot certify byte equality for those pages when their raw hashes differ.

The audit checks candidates with usable text against both the verified QA passages and overlapping windows from the full public pages. It reports candidates that could not be scanned. The build runs this audit again before writing any corpus split.

## Build and handoff

```sh
python phase_04_indic_aec_corpus/build_corpus.py
```

The builder excludes Phase 4 candidates, never modifies the benchmark. It uses normalized word 5-gram containment, including overlap distributed across multiple candidate documents. It removes exact/fuzzy document duplicates, assigns global document families to train/validation with seed 42, validates the repository corpus schema, and requires every language in both splits.

A valid release is written to `outputs/phase_04_indic_aec_corpus/releases/<content-version>/`:

- `corpus.jsonl`, `train.jsonl`, `validation.jsonl`
- `statistics.json`

Every record contains the required `document_id`, `language`, `text`, `source_url`, `license`, `split`, plus provenance. Failed builds produce no new release. Older version directories remain historical artifacts, not evidence that the latest run passed.

Phase 5 should tokenize these document-disjoint splits using a pinned XLM-R tokenizer and retain document/family provenance. MLM preparation and training remain Phase 5 work. Do not feed Phase 6 contexts to training.

## Further corpus expansion

1. Confirm reuse terms and native origin for the pending sources and expand collection in Marathi, Bengali and Tamil.
2. Obtain the four original HTML snapshots listed in the reference report if exact full-document coverage is required.
3. Review additional extracts and record decisions against their text hashes.
4. Evaluate overlap detection on naturally occurring OCR errors, known negatives, and translated copies.
5. Scale the pairwise deduplication step before substantially increasing the corpus size.
6. Verify edition-specific redistribution terms before publishing the underlying text.

The current tests use synthetic text and verify pipeline safeguards; they are not measured corpus results.

## Source expansion after the Hindi pilot

The registry now includes Marathi practical training material from Vigyan Ashram and
Bangladesh Open University agriculture units on irrigation, fibre crops and soil.
The school texts cover Bangladeshi conditions. Keep the region and source type
when reporting coverage and avoid treating it as West Bengal field guidance.

The Marathi agricultural glossary is a supplementary terminology candidate. Assess
its suitability before counting it as extension prose. The CAG Tamil seed
booklet and Tamil Wikibooks mushroom chapter have published reuse terms, but
their original-language provenance needs review before acquisition into the
core corpus. The Phase 6 reservations and document-family rules still apply.

To process newly acquired documents without rerunning OCR on unchanged inputs,
pass `--document-id` once per requested ID:

```sh
python phase_04_indic_aec_corpus/extract_text.py --ocr \
    --document-id AEC_MR_VA_001 --document-id AEC_BN_BOU_2301_06
python phase_04_indic_aec_corpus/pipeline.py audit
```

Existing extracted records carry forward only when the latest acquisition
manifest has the same raw hash. A spot check helps identify problems, but a
full review is needed before marking a document accepted. In particular,
compare tables and diagram text with the source page and check numeric
values, units, and reading order.

For Phase 4 Bengali PDFs, the extraction config selects `ben+eng` when OCR is
needed. Install both Tesseract language packs before running `--ocr`. Phase 6
retains its original OCR defaults for benchmark reconstruction. Review
multi-column pages and mixed-language tables after extraction, since the OCR
language setting does not control reading order.

## Indian government books scope

The release configuration accepts books and book chapters published by Indian
government bodies or Indian public institutions. The registry records the
publisher country, document format and publisher evidence. Acquisition
checks this scope, and the build checks existing extracts against the current
registry so older rows cannot carry foreign or non-book material into a release.

The prior Bangladesh Open University extracts and the ICAR Kheti journal
pilot remain research artifacts outside the books-only release. Free access
to a PDF does not supply reuse terms. Indian textbook candidates stay
pending until the exact edition has reuse and original-language evidence.

### Prepare the candidate corpus before human review

After acquisition and OCR, assemble the review packet with:

```sh
python phase_04_indic_aec_corpus/pipeline.py prepare-review
```

This applies the release's source, rights, provenance and overlap checks. It
writes candidate extracts, their exact hashes, and the per-language coverage
report under `outputs/phase_04_indic_aec_corpus/pre_review/`. Human review is not
required to generate the packet. Successful human reviews are recorded afterward
in `quality_reviews.csv`, then the normal build generates the train and
validation splits. If no eligible text is available the packet reports zero;
this is not a completed corpus or a passing overlap audit.
