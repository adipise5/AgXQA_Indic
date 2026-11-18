# Phase 4 — Indic Agricultural Extension Corpus

Owner: Aditya Pise. Status: initial pipeline implemented; no research corpus released.
Main repository: https://github.com/adipise5/AgXQA_Indic
Baseline: e8754e957bcb8da6ddd9fd916e7820d8a334ecb8.

## Run the first milestone

From the repository root (Python 3.10+):

```sh
python -m pip install -r phase_04_indic_aec_corpus/requirements.txt
python phase_04_indic_aec_corpus/download_documents.py --dry-run
python -m unittest discover -s tests -v
```

The dry run writes an acquisition report under `outputs/phase_04_indic_aec_corpus/reports/` without making document requests. Ten ICAR Kheti articles have document-specific noncommercial reuse terms. Seven other source leads remain pending. See [the measured Hindi pilot](results/pilot.json) for extraction counts, input hashes and the Phase 6 audit.

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

## Remaining work before a research release

1. Confirm reuse terms and native origin for the pending sources and expand collection in Marathi, Bengali and Tamil.
2. Obtain the four original HTML snapshots listed in the reference report if exact full-document coverage is required.
3. Complete extraction quality review and record reviewer coverage.
4. Calibrate containment thresholds, add OCR-tolerant character matching and translated-copy review; current word shingles do not establish semantic non-overlap.
5. Add scalable deduplication and paragraph-level duplicate-family analysis before large-scale collection. The initial pairwise implementation favors small pilots.
6. Generate corpus-wide language, split, domain and publisher statistics with the pinned tokenizer.
7. Review permissions before publishing any full text.

The current tests use synthetic text and verify pipeline safeguards; they are not measured corpus results.
