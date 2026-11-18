# Phase 6 annotation guidelines (human validation)

Each reviewer checks the QA pairs in their own language:

| Language | Reviewer |
|---|---|
| Hindi | Aditya Saxena |
| Marathi | Aditya Pise |
| Bengali | Saunak |
| Tamil | Dhiraj |

## Your sheet

Open `human_review/private/<language>_review.csv` in Excel, LibreOffice or Google Sheets, and keep it UTF-8. The Phase 6 owner shares this sheet privately because it contains the source passage (`context`). The committed `human_review/<language>_review.csv` holds the same rows without the passage.

Each row has:

- **What you review:** `context`, `question`, `answer_text`, `evidence_sentence`.
- **Labels to sanity-check:** `question_type`, `domain`, `difficulty` (set by the generator).
- **Automatic checks:** `auto_warnings`, e.g. `answer_occurs_multiple_times`, `long_answer`, `no_question_word_possible_yes_no`.
- **Traceability:** `source_url`, to open the original article.

## What to fill

| Column | Value |
|---|---|
| `reviewer` | your name (pre-filled) |
| `fluency` | 1–5: is the question natural, grammatical text a native speaker would write? |
| `question_correctness` | 1–5: is it a meaningful agricultural question, answerable from the passage alone? |
| `answer_correctness` | 1–5: does `answer_text` correctly and completely answer it, as the shortest sufficient span? |
| `terminology` | 1–5: are agricultural terms, units and names correct and natural in your language? |
| `status` | `Accept`, `Needs correction` or `Reject` |
| `corrected_question` | only with `Needs correction`: the fixed question |
| `corrected_answer` | only with `Needs correction`: the fixed answer, **copied exactly from the passage** |
| `comments` | free text: why you rejected or changed it |

Scores: **5** perfect · **4** minor issue, acceptable · **3** noticeable issue · **2** serious issue · **1** unusable.

## Decision rules

**Accept** when every score is 4 or 5 and no fix is needed.

**Needs correction** when the pair is worth keeping but one of these needs fixing:

- The question has a grammar, spelling or mixed-script error, e.g. `फुदka` instead of `फुदका`.
- The question depends on the passage, as in "this pest" (`इस कीट`, `এই পোকা`, `இந்த …`). Rewrite it so it stands alone, e.g. "Which pest attacks paddy at the tillering stage?".
- The answer is too long or too short. Pick the shortest exact span from the passage.
- The answer string appears more than once. Check that the intended occurrence is the one answering the question; if not, tighten the answer so it is unique.

**Reject** when:

- The answer is wrong, or the question can't be answered from the passage.
- The question is a yes/no question, trivial, or a near-duplicate of another question on the same passage.
- The passage itself is not usable agricultural information, e.g. only a subsidy table, or garbled text.
- The text looks machine-translated rather than native. Say so in `comments`; the source document will be checked.

## Rules for corrections

- A corrected answer must be an **exact copy** of a passage substring. Copy and paste it; don't retype it. The build drops corrections that aren't found in the passage, and recomputes the offset.
- Never edit `context`, `id` or the passage itself.
- Keep corrections in your language; don't introduce English unless the passage uses that term.

## After review

Send the sheet back, or save it in place. Then the Phase 6 owner runs:

```bash
python phase_06_native_indic_qa/scripts/export_review_sheets.py   # copies reviews into the public sheet
python phase_06_native_indic_qa/scripts/build_dataset.py          # applies Accept / corrections / Reject
python phase_06_native_indic_qa/scripts/dataset_statistics.py     # fills Table 3 and the review-outcome chart
```

**Agreement.** If a second qualified reviewer is available for a language, run `export_review_sheets.py --languages <code> --second-sample 50`. They review the 50 items in `<language>_review_second.csv` independently, and Cohen's κ on `status` is reported.

**Honesty.** Report actual coverage, e.g. "Hindi: 400/1,107 reviewed". Never describe a sample as full review. Native-speaker review is not agricultural-expert verification.
