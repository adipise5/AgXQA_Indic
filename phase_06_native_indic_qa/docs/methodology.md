# Phase 6 methodology (technical detail)

The README covers what was done and the results. This document records **how** each step decides, with the exact rules and thresholds, so the pipeline can be audited and reproduced. All thresholds are in `configs/generation_config.yaml`, and shared logic is in `scripts/common.py`.

## 1. Source selection

A candidate document is included only if all of these hold:

- it is from a government, university, ICAR or KVK source, or Vikaspedia republishing one;
- its text is extractable Unicode in the target language;
- it is not a translation of an English page, as far as can be determined.

| Check | Rule |
|---|---|
| Script ratio | Share of letters (incl. combining marks) in the target script's Unicode block: Devanagari U+0900–097F and U+A8E0–A8FF; Bengali U+0980–09FF; Tamil U+0B80–0BFF |
| Garbled ratio | Maximum of three signals (below) |
| Native origin | Excluded when the page mirrors an English page file for file (TNAU Agritech); otherwise recorded as `unknown`, with the article credit kept for reviewers |

Garbled-ratio signals (a page or paragraph is rejected when the value exceeds 0.01):

1. **Foreign letters.** The share of letters that are neither the target script nor ASCII. This catches broken ToUnicode maps such as `ठɘबक ɧ͆सचाई`.
2. **Bad word starts.** The share of words that begin with a dependent vowel sign or virama, which is impossible in valid text. This catches glyph remapping such as `िाधनिंपत्ती`.
3. **Doubled signs.** Doubled dependent vowel signs (`कीी`, `दोोनोों`) per word, which accompany dropped half-letters.

Legacy-font PDFs (Kruti Dev, Shree-Dev, Bamini) fail the script ratio instead; their text is Latin gibberish.

## 2. Extraction

- **HTML.** Prefer the longest `content` or `articleBody` HTML string embedded in `__NEXT_DATA__` or JSON-LD. Fall back to trafilatura, then BeautifulSoup. The embedded HTML is converted to lines as follows:
  - source whitespace (including newlines) is collapsed first;
  - every `tr` becomes one line, with cells joined by ` | `;
  - every block element (`p`, `li`, `h1`–`h6`, `div`, …) becomes its own line.
- **PDF.** PyMuPDF, page by page, `sort=True`. A page is flagged `needs_ocr` if it has fewer than 40 characters, a script ratio below 0.35, or a garbled ratio above 0.01. With `--ocr`, flagged pages are rendered at 300 dpi and read by Tesseract (`hin`, `mar`, `ben`, `tam`).

## 3. Cleaning

Steps, in order:

1. **Normalise.** NFC. ZWJ and ZWNJ are kept. Python's `re` `\w` misses Indic vowel signs, so the `regex` module with `\p{L}\p{M}\p{N}` is used for matching keys.
2. **Headers and footers.** A short line (≤ 12 words) appearing on ≥ 40% of pages (minimum 2 pages, documents with ≥ 3 pages) is removed. Page-number lines are removed by regex, including Indic digits and words like पृष्ठ, पान, পৃষ্ঠা, பக்கம்.
3. **Credit lines.** Lines beginning with स्त्रोत, स्रोत, माहिती संदर्भ, लेखक, ஆதாரம், সূত্র, সুত্র, তথ্য সংকলন, … are removed from the text and recorded in the source lists.
4. **Paragraphs.** For HTML, one line is one paragraph. For PDF, wrapped lines are joined, and a sentence-final line shorter than 75% of the median line length ends the paragraph. A short line without a sentence terminator, followed by text, becomes a heading.
5. **Filters.** Paragraphs need a script ratio ≥ 0.6 (headings ≥ 0.3, to allow English glosses) and a garbled ratio ≤ 0.01. A paragraph under 8 words that ends with a terminator is merged into the preceding paragraph; otherwise it is dropped.
6. **Hindi vs Marathi.** Function-word votes:
   - Hindi markers include है, हैं, और, के, की, में, को, से, लिए, चाहिए, यह.
   - Marathi markers include आहे, आहेत, आणि, मध्ये, साठी, नाही, असते, किंवा, त्यामुळे.
   - Words shared by both languages (या, ही, होती, जाती, जाते, कर) are excluded.
   - Abbreviation-prone markers followed by a full stop (हे., को., से., व.) are ignored.
   - A language wins only with ≥ 2 votes and more than twice the other's.
7. **Deduplication.** Exact (normalised key) and near-duplicate (rapidfuzz `ratio` ≥ 92) paragraphs are removed across all documents of a language.

## 4. Chunking

Paragraphs are packed in order, and a chunk never crosses a heading:

- A chunk is emitted once it reaches 100 words and the next paragraph would exceed 200.
- A chunk is also emitted when the hard maximum of 260 would be exceeded.
- A paragraph longer than 260 words is split at sentence ends (`।`, `॥`, `.`, `?`, `!`).
- A tail under 60 words joins the previous chunk of the same section if it fits under 260. Otherwise it is kept if it is at least 40 words, and dropped if not.
- Each chunk records `context_id` (`<document>_C###`), title (the section heading), XLM-R token count (`xlm-roberta-base` tokenizer), and a keyword `domain_hint`.

The `domain_hint` is used only to prioritise irrigation contexts during generation. Final domains come from the generator.

## 5. Generation

- One request per context, using `prompts/qa_generation_prompt.txt` with `$LANGUAGE` (e.g. `Tamil (தமிழ்)`) and `$CONTEXT` substituted.
- `temperature=0.2`, `response_mime_type=application/json`.
- **Parsing.** Accepts a JSON array, an object wrapping an array (`qa_pairs`, `questions`, …) or a single object. On a decode error, stray backslashes are escaped and parsing is retried.
- **Retries.** Up to 3 attempts per context, with back-off of 8 s × attempt. A quota error (429 / RESOURCE_EXHAUSTED) does not use up an attempt; two in a row retire the current key.
- **Budget.** A per-key daily counter (Pacific day) is incremented **before** each request, so a crash can never under-count.
- **Order.** Irrigation-hinted contexts are processed first. Other contexts follow round-robin across documents and interleaved across languages, so a budget stop leaves every language with some coverage.

## 6. Answer-span validation

For each candidate:

1. NFC-normalise the question, answer and evidence sentence.
2. **Locate the answer.** Try exact `find` on the raw answer, then on the answer with boundary punctuation and whitespace trimmed, then a whitespace-flexible regex. There is **no fuzzy matching**: a paraphrased answer is rejected.
3. **Pick the occurrence.** Locate the evidence sentence (exact, else rapidfuzz `partial_ratio_alignment` ≥ 85). Choose the answer occurrence inside the evidence span; otherwise the occurrence nearest to it, with a warning. Then assert `context[start:start+len] == answer`.
4. **Hard rejects:**
   - answer under 2 characters, over 20 words, or more than 50% of the context;
   - question under 3 or over 40 words;
   - question not in the target language (script ratio ≥ 0.6, garbled ≤ 0.01, Hindi vs Marathi);
   - answer (≥ 4 characters) appears inside the question, or the question equals the answer.
5. **Warnings** (kept, shown to reviewers):
   - answer over 10 words;
   - full-sentence answer;
   - no question word from the per-language list;
   - question copies context wording (partial ratio ≥ 95, ≥ 6 words);
   - answer outside the evidence sentence;
   - multiple occurrences;
   - invalid labels. Labels are normalised through alias maps, e.g. `water management` → `irrigation`.

## 7. Deduplication

Candidates are sorted by context order, then by number of warnings, so better candidates are kept first. A candidate is removed if:

- its normalised question exactly matches a kept question; or
- its question is a fuzzy match (`token_sort_ratio` ≥ 90) of a kept question in the same language; or
- its answer span duplicates another one in the same context; or
- its context already has 3 kept questions.

Removed candidates and their reasons are written to `validated_qa/<lang>_dedup_removed.jsonl`.

## 8. Dataset build and IDs

`build_dataset.py` applies reviews:

| Review status | Effect |
|---|---|
| `Accept` | kept |
| `Needs correction` with a valid fix | corrected answer must be an exact substring; its offset is recomputed at the occurrence nearest the original |
| `Needs correction` without a fix | left out |
| `Reject` | dropped |
| Unreviewed | kept only with `--include-unreviewed` (used for the current release), marked `human_validated: false` |

Documents and contexts listed in `leakage_exclusions.json` are always dropped. IDs (`NATIVE_HI_000001` …) are assigned in source order at build time, so they are contiguous; the candidate ID is kept in `metadata.candidate_id`.

The release (`final/`) replaces the passage with `context_id`, `context_sha256` (SHA-256 of the UTF-8 NFC context) and `context_chars`.

## 9. Hydration

`hydrate_dataset.py` maps each release record to a context by `context_id`. If the hash differs, it searches all contexts for one with that hash, in case numbering shifted after a source edit. A record is emitted only if the SHA-256 matches and `context[start:start+len] == answer`. On 2026-11-18 the rebuild was byte-identical to the build output for all 2,351 records.

## 10. Leakage checks

| Check | Rule |
|---|---|
| Phase 4 manifest | Identical `document_hash`; identical `normalize_url` key; title `token_sort_ratio` ≥ 90 (flagged only) |
| Normalised URL | Lowercase host with `www.` removed, and every `*.vikaspedia.in` host collapsed to `vikaspedia.in`. In the path: percent-decoded, NFC, repeated `/` collapsed, `/viewcontent` dropped. Legacy hex slugs (`907924930` → `इतर`) decoded |
| Passage overlap | Word 5-grams of normalised text. A context is contaminated if ≥ 50% of its 5-grams occur in the corpus. A document is excluded if ≥ 20% of its contexts are contaminated. The corpus is streamed, keeping only shingles that also occur in Phase 6, so memory stays small |
| Internal | Contexts whose 5-grams are ≥ 50% present in a different Phase 6 document |

## 11. Statistics definitions

| Metric | Definition |
|---|---|
| Documents collected | Rows assessed in the source lists (`include` yes and no) |
| Unique documents | Distinct file SHA-256 among downloaded documents |
| Rejected QA pairs | Generated candidates − final records (validation failures + dedup removals) |
| Answer-span validity | 1 − (candidates failing `answer_not_in_context`) / candidates |
| Duplicate rate | (exact + fuzzy duplicate questions + duplicate spans) / candidates passing validation |
| Question / answer correctness | Share of reviewed items scored ≥ 4 |
| Human acceptance rate | (Accept + Needs correction) / reviewed |
| Words | Whitespace tokens |
| Tokens | `xlm-roberta-base` input IDs, including special tokens |
