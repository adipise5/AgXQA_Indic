# Human review sheets

`scripts/export_review_sheets.py` writes one sheet per language here: `hindi_review.csv`, `marathi_review.csv`, `bengali_review.csv`, `tamil_review.csv`. Each row has the context, question, answer and the automatic warnings for one QA pair.

The sheets contain source passages, so they are git-ignored. This repo is public, and redistribution of the passages is not yet permitted (see `../sources/SOURCES_NOTES.md`). They are shared with reviewers privately.

| Reviewer | Language | Sheet |
|---|---|---|
| Aditya Saxena | Hindi | `hindi_review.csv` (1,107 rows) |
| Aditya Pise | Marathi | `marathi_review.csv` (581 rows) |
| Saunak | Bengali | `bengali_review.csv` (420 rows) |
| Dhiraj | Tamil | `tamil_review.csv` (243 rows) |

Rubric: score `fluency`, `question_correctness`, `answer_correctness` and `terminology` from 1 to 5. Set `status` to one of:

- `Accept`
- `Needs correction` (fill `corrected_question` and/or `corrected_answer`; a corrected answer must be copied exactly from the context)
- `Reject`

Re-running the export keeps every review already entered. `build_dataset.py` applies the reviews when it builds the dataset.
