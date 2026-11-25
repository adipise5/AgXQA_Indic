# AgXQA-Indic annotation guidelines

These instructions define a review protocol. Blank worksheet rows and assigned reviewer names are not completed reviews. No human ratings have been collected in the current preparation snapshot.

| Language | Assigned reviewer | Target |
|---|---|---|
| Hindi | Aditya Saxena | All retained test questions, seeded train/validation sample, every flagged question |
| Marathi | Aditya Pise | Same |
| Bengali | Saunak | Same |
| Tamil | Dhiraj | Same |

The sample fractions are 5% of train and 10% of validation, rounded upward. For the Gemini cohort, this gives 66 train, 35 validation, and 327 test questions per Indic language before adding flagged questions. Seed 42 determines membership independently by language and split. All five English questions with invalid source gold offsets also require source adjudication.

## Review materials and workflow

1. For the Gemini bundle, use `review_material/<language>.jsonl`, which pairs the pinned English source with the cleaned candidate. The original exports are in `inputs/` and span diagnostics are in `corrections.jsonl`. Match `question_id` to `original_id` and preserve its official split. Review context, question and **every** answer reference together. Verify agricultural terms, factual coverage, negation, numbers, measurements and answer extractability.
2. Work in the bundle's versioned `human_review/*_review.csv` worksheets. Preserve `raw_record_sha256` and `candidate_sha256`, which bind the judgment to the original input and exact candidate version. Record proposed changes in the review CSV so the original exports stay available for comparison.
3. Enter your actual name in `reviewer`, a timezone-bearing ISO timestamp in `reviewed_at` (for example `2026-11-18T15:00:00+05:30`), all four ratings and an `Accept`, `Correct` or `Reject` decision. The example timestamp is a format example, not evidence of a review. Quote CSV fields containing commas, newlines or JSON.
4. Correct text only in `corrected_context`, `corrected_question` and `corrected_answers_json`. Recompute answer offsets against the **entire exact corrected context** using Python Unicode character positions, not UTF-8 byte counts. Use NFC for Indic text, preserving meaningful joiners. Corrected answers use `{"text":[...],"answer_start":[...]}` and must preserve the original number/order of references. Explain the change in `comments`.
5. Rerun the build/validation commands. A human decision cannot override an invalid/nonempty answer span, corrupt Unicode, missing references or changed split. For numerical/unit/script or other heuristic warnings you have checked, list each code in `override_issues`, separated by semicolons, and give the linguistic/technical reason in `comments`. New numerical/unit/script problems introduced by corrections are checked again. The original automated flags remain in the accepted record as history.
6. Preserve original judgments. If reviewers disagree, add one separate named adjudicator judgment with `adjudicator` set to that adjudicator's name. More than one active judgment by the same reviewer is rejected; archive superseded worksheets outside the active directory. Never silently pick the first conflicting decision.

## Rating rubric

Rate the **original cleaned machine output before correction**, so corrected text does not inflate translation-quality measurements. All criteria use integers 1–5.

| Score | Meaning |
|---|---|
| 5 | Correct and natural; no substantive issue |
| 4 | Good; negligible issue that preserves meaning and answer correctness |
| 3 | Minor correction needed |
| 2 | Major correction needed |
| 1 | Incorrect or unusable |

| Column | What to inspect |
|---|---|
| `semantic_equivalence` | Context and question retain the English meaning, scope, negation and information |
| `fluency` | Natural, grammatical target-language text |
| `terminology` | Agriculture/irrigation terms retain technical meaning |
| `answer_correctness` | All gold answers express the source answers and can be located at valid offsets |

`Accept` means no text/answer correction. A low rating should be accompanied by an explanation even when an annotation is retained. `Correct` requires at least one actual edit plus rationale. `Reject` blocks the full benchmark until the problem is resolved through a traceable new version/adjudication; the builder does not silently drop it or reduce target counts.

Suggested `error_categories` (semicolon separated): `terminology`, `answer_mismatch`, `grammar`, `meaning_change`, `missing_content`, `numerical_accuracy`, `other`. These are real human judgments; do not infer them from automated flags. Document additional categories consistently.

## Answer and source adjudication

Independent answer translation may inflect or paraphrase differently from the translated paragraph. A missing substring or repeated occurrence must be resolved by reading both languages and recording the correct **existing semantic answer span**. Do not append answer strings to contexts to manufacture extractability. Correct a context only when the context translation itself is wrong; preserve its meaning.

The official English test records `1170477`, `1174212`, `1183012`, `1181597` and `481` contain nine invalid answer offsets. Record `481` also has a placeholder-like gold string. Confirm actual gold text from the source rather than guessing. Preserve the Phase 1 official source bytes and original scoring experiment. Any Phase 2 English correction creates a documented benchmark annotation version; later evaluation must identify that version. A changed reference count needs a separately approved annotation protocol and is currently blocked.

## Agreement and coverage

Only mark `independent_review=true` if a second reviewer scored the same language/example without consulting the first review. Different people assigned to different languages do not establish agreement. Cohen's kappa is reported on paired categorical decisions for each reviewer pair/language with at least two paired examples; constant expected agreement makes it undefined (`null`). An adjudication is not an independent judgment.

Quality means and decision rates count submitted judgments. Coverage counts unique original question IDs by language and split. Report both denominators and the sampling bias from oversampling flagged records. Before/after span plots use the same accepted, review-resolved IDs and disclose exclusions. For the Gemini evaluation, report review coverage against the 327 retained test IDs per Indic language and disclose that the official source originally had 330.
