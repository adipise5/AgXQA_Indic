# Phase 6: IndicAgriQA-Native

A held-out extractive QA benchmark built from **native** Hindi, Marathi, Bengali and Tamil agricultural extension documents. Unlike Phase 2 (AgXQA-Indic, translated from English), every context, question and answer comes from native-language source text. Nothing is written in English first and then translated.

| | Phase 2: AgXQA-Indic | Phase 6: IndicAgriQA-Native |
|---|---|---|
| Source content | English AgXQA, translated | Native Indic documents |
| Questions | Translated | Generated directly in the target language |
| Gold answers | Translated and aligned | Exact spans of native contexts |
| Parallel across languages | Yes | No (IDs are not aligned) |
| Purpose | Cross-lingual evaluation | Native agricultural QA evaluation (test only) |

Owner: Aditya Saxena. Reviewers: Aditya Saxena (hi), Aditya Pise (mr), Saunak (bn), Dhiraj (ta).
Targets: 250 QA pairs per language (1,000 total). The minimum viable target is 100 per language. Quality comes before quantity.

## Status (2026-11-18)

- 70 native source documents downloaded and extracted (Hindi 21, Marathi 20, Bengali 18, Tamil 11), mostly Vikaspedia articles credited to state agriculture departments, TNAU and KVKs. See `sources/SOURCES_NOTES.md`.
- 827 contexts. 2,506 QA candidates were generated with `gemini-3.5-flash-lite` (prompt v2, temperature 0.2).
- **2,351 QA pairs** passed automatic validation and dedup: Hindi 1,107, Marathi 581, Bengali 420, Tamil 243.
- **Not yet human-reviewed.** Every record has `human_validated: false`.
- **Passages are not in this repo.** Vikaspedia reuse needs permission (vikaspedia@cdac.in), and 9 articles are third-party newspaper content. Raw documents, contexts and QA files are git-ignored. Rebuild them with the scripts, or get them from the Phase 6 owner.
- Phase 4 has no corpus yet, so the leakage check has not run. `sources/phase4_exclusion_list.csv` lists the documents Indic-AEC must not ingest.

## Layout

```
phase_06_native_indic_qa/
├── configs/generation_config.yaml   # every threshold and the LLM settings
├── prompts/qa_generation_prompt.txt # QA prompt ($LANGUAGE, $CONTEXT); bump prompt_version on edit
├── sources/<language>_sources.csv   # hand-curated document lists (input)
├── sources/source_manifest.csv      # generated: hashes, paths, provenance
├── scripts/                         # pipeline, one script per step (below)
├── human_review/<language>_review.csv
└── results/                         # reports, statistics, plots

data/native_indic_qa/                # repo root; raw_documents/ is git-ignored
├── raw_documents/<language>/
├── processed_contexts/{extracted,cleaned}/ and <lang>_contexts.jsonl
├── generated_qa/<lang>_raw.jsonl    # raw LLM output + model/prompt/settings
├── validated_qa/<lang>_validated.jsonl, <lang>_dedup.jsonl
├── final/{hindi,marathi,bengali,tamil,test}.jsonl
├── leakage_exclusions.json, metadata.csv, dataset_card.md
```

## Setup

Python 3.10+. Phase 6 has its own dependencies (the shared utilities stay standard-library only):

```bash
pip install -r phase_06_native_indic_qa/requirements.txt
```

QA generation reads `GEMINI_API_KEY` (and optionally `GEMINI_API_KEY_2`, used when the first key's daily budget is spent) from a `.env` file at the repo root. `.env` is git-ignored; never commit keys.

Optional extras:

- OCR for scanned or legacy-font PDFs: `pip install pytesseract`, plus the Tesseract binary with the `hin`, `mar`, `ben` and `tam` language packs (on Windows, use the UB Mannheim build).
- Plots: `pip install matplotlib`.

All commands below run from the repo root. `--languages` takes `hi,mr,bn,ta` or `all`.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1–2 | `download_documents.py` | Downloads `source_url` or copies `local_path` for each CSV row, computes SHA-256 and writes `source_manifest.csv` |
| 3 | `extract_text.py [--ocr]` | Extracts PDF text page by page with PyMuPDF and HTML text with trafilatura (BeautifulSoup as fallback). Flags pages that need OCR |
| 4 | `clean_text.py` | Applies NFC, removes repeated headers, footers and page numbers, rebuilds paragraphs, and runs language, script, corruption and near-duplicate filters |
| 5 | `export_phase4_exclusions.py`, `check_data_leakage.py` | Writes `sources/phase4_exclusion_list.csv` for Phase 4 to skip. Then checks hash, URL, title and passage overlap with the Indic-AEC corpus once that corpus exists |
| 6 | `chunk_documents.py` | Builds 100–200 word contexts within one section. Records page range, a domain hint and the XLM-R token count |
| 7 | `generate_qa.py` | Sends one Gemini call per context with temperature 0.2 and JSON output. Resumable, with a daily call budget |
| 8 | `validate_answer_spans.py`, `deduplicate.py` | Computes answer offsets in code and runs the quality checks and duplicate removal |
| 9 | `export_review_sheets.py` | Writes or updates the reviewer CSVs without overwriting existing reviews |
| 10 | `build_dataset.py` | Writes the final SQuAD-format JSONL files with `NATIVE_XX_000001` IDs |
| 10b | `dataset_statistics.py` | Writes statistics, the per-step funnel, irrigation vs other topics, IAA and plots |

```bash
S=phase_06_native_indic_qa/scripts
python $S/download_documents.py --dry-run
python $S/download_documents.py
python $S/extract_text.py
python $S/clean_text.py
python $S/chunk_documents.py
python $S/check_data_leakage.py --phase4-manifest <p4_manifest.csv> --phase4-corpus <p4_corpus_dir>
python $S/generate_qa.py --dry-run
python $S/generate_qa.py --languages hi --limit 3     # sanity check first
python $S/generate_qa.py --languages all --limit 130  # ~130 contexts x 2-3 Qs ≈ 350 candidates/lang
python $S/validate_answer_spans.py
python $S/deduplicate.py
python $S/export_review_sheets.py --second-sample 50
#   ... reviewers fill human_review/*.csv ...
python $S/check_data_leakage.py --phase4-manifest ... --phase4-corpus ... --strict
python $S/build_dataset.py
python $S/dataset_statistics.py
```

`generate_qa.py --domain irrigation` restricts a run to contexts whose keyword hint is irrigation. Use it to reach the target of 300 irrigation QA pairs. Every script rewrites its own outputs, so you can re-run any step after a fix. Generation is the only step that spends API calls, and it skips contexts that are already done.

To smoke-test without touching `data/`, set `PHASE6_SANDBOX=<dir>`. This redirects data, sources, review and results into `<dir>`.

## Step 1: collecting documents

Fill one row per document in `sources/<language>_sources.csv`:

`document_id` (e.g. `HI_DOC_001`), `document_title`, `language`, `organization`, `source_url`, `publication_year`, `region`, `domain`, `license`, `language_origin` (`native` / `translated` / `unknown`), `local_path` (for files saved by hand, e.g. behind a form), `include` (`yes`/`no`), `notes`.

- **Sources:** use government, ICAR, KVK and agricultural-university sources only. Starting points:
  - Hindi: ICAR Hindi publications, IARI, ICAR-CIAH farmer advisories, Raj Kisan, ICAR Kheti.
  - Marathi: MPKV, VNMKV farmer leaflets, Maharashtra Agriculture Department.
  - Bengali: Matir Katha, UBKV, BCKV, Cooch Behar KVK.
  - Tamil: TNAU Agritech (`agritech.tnau.ac.in/ta/`), Tamil Nadu Agriculture Department.

  Avoid blogs, Wikipedia, AI-generated pages and translated QA datasets.
- **Native text only:** set `language_origin = native` only when the document was written in that language. Translated government circulars do not qualify.
- **Licensing:** do not assume a public government PDF may be redistributed. Record what you know in `license`. Anything else stays `unverified` and is not published (see Release below).
- **Format:** prefer digital PDFs and HTML. `extract_text.py` flags scanned pages and **broken-font PDFs**, which are common with Indic fonts. Broken-font PDFs extract as text like `ठɘबक ɧ͆सचाई`. A garbled-text detector catches these. Re-run them with `--ocr`, or replace them with a cleaner source.
- **Phase 4 separation:** coordinate with Aditya Pise before collecting. Phase 6 documents must not be in the Phase 4–5 Indic-AEC corpus.

## Automatic validation (step 8)

The LLM's offsets are never used. Each check below is either a hard rejection or a warning that is passed to reviewers.

| Check | Outcome |
|---|---|
| Answer is an exact contiguous substring of the NFC context (only boundary punctuation and whitespace may be trimmed) | reject if not |
| Answer appears more than once in the context | the evidence sentence picks the occurrence. If it can't, a warning is added |
| Question is in the target script, and Hindi vs Marathi is detected with function-word votes | reject if wrong |
| Answer is 2+ characters, at most 25 words and at most 50% of the context | reject if not |
| Question is 3–40 words, does not contain the answer, and is not equal to the answer | reject if not |
| No question word (possible yes/no question), no `?`, copied context wording, evidence sentence not found, invalid enum value | warning |
| Exact or fuzzy (≥90) duplicate question, same answer span twice in one context, more than 3 questions per context | removed in dedup |

ZWJ/ZWNJ and vowel signs are never stripped. Bengali and Marathi spelling depends on them.

## Human review (step 9)

Each reviewer opens their `human_review/<language>_review.csv` in Excel or Sheets and keeps it in UTF-8. They fill in:

- `fluency`, `question_correctness`, `answer_correctness`, `terminology`: scored 1–5, where 5 is perfect.
- `status`: `Accept`, `Needs correction` or `Reject`.
- `corrected_question` / `corrected_answer`: optional. Fill these with `Needs correction`. A corrected answer must still be copied exactly from the context, or the build drops the item.
- `comments`

Check the context, the question's fluency, whether the answer is correct and is an exact span, the terminology, any ambiguity, and that the item can be traced to its source (`source_url`, `page_start`). The `auto_warnings` column shows what the automatic checks flagged.

**Agreement:** `--second-sample 50` creates `<language>_review_second.csv` for a second qualified reviewer. `dataset_statistics.py` then reports percent agreement and Cohen's κ on status. Where no second reviewer exists, the statistics state that a single reviewer validated the language. Native-speaker review is **not** agricultural-expert validation, and the paper should not describe it as such.

## Release

- `final/*.jsonl` uses the SQuAD-style schema (`id`, `title`, `language`, `context`, `question`, `answers.text`, `answers.answer_start`, `metadata`). `test.jsonl` contains all four languages.
- `final/test.shared.jsonl` holds the same records in the team repo's shared contract (`schemas/qa.schema.json` in adipise5/AgXQA-Indic). Each record has `split: "test"` and `source_document_id`, and `answers` is a list of `{text, answer_start}`. It passes `shared/validate.py`, so Phase 7 can use `shared.evaluate` on it directly.
- Phase 6 is a **test-only** benchmark. Do not train on it.
- Publish full passages only from documents whose `license` permits redistribution. For the rest, publish the manifest (URLs and hashes) and these scripts so the data can be rebuilt. `raw_documents/` and `processed_contexts/extracted/` are git-ignored.

## Hand-off to Phase 7 (Dhiraj)

- Report EM and F1 per language and the macro-average. Report irrigation-domain results separately from the other domains. `metadata.domain` holds the domain.
- Do **not** compute cross-lingual consistency on this dataset, because the questions are not parallel.
- Use `max_seq_length` with doc stride. Indic contexts tokenize into many more XLM-R tokens than English contexts, and `chunk_documents.py` flags contexts over 384 tokens.
