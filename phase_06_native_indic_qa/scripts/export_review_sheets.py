"""Step 9 (prep): write human-review sheets for native-language reviewers.

Creates human_review/<language>_review.csv from validated_qa/<lang>_dedup.jsonl.
Re-running keeps everything reviewers already filled in and only appends new
candidates. Sheets are UTF-8 with BOM so Excel shows Indic text correctly.

Reviewer columns (see README for the rubric):
    reviewer, fluency, question_correctness, answer_correctness, terminology (1-5 each),
    status (Accept / Needs correction / Reject), corrected_question, corrected_answer, comments

--second-sample N also writes <language>_review_second.csv with N random items
for a second qualified reviewer, used to compute inter-annotator agreement.

Usage:
    python phase_06_native_indic_qa/scripts/export_review_sheets.py --languages all
    python phase_06_native_indic_qa/scripts/export_review_sheets.py --languages hi --second-sample 50
"""

from __future__ import annotations

import argparse
import random

from common import (LANGUAGES, REVIEW_DIR, VALIDATED_DIR, read_csv, read_jsonl, rel, selected_languages,
                    write_csv)

REVIEW_FIELDS = [
    "id", "language", "context", "question", "answer_text", "auto_warnings",
    "reviewer", "fluency", "question_correctness", "answer_correctness", "terminology",
    "status", "corrected_question", "corrected_answer", "comments",
    "evidence_sentence", "question_type", "domain", "difficulty", "context_id", "document_id",
    "source_url", "page_start", "answer_start",
]
FILLED_BY_REVIEWER = ["reviewer", "fluency", "question_correctness", "answer_correctness", "terminology",
                      "status", "corrected_question", "corrected_answer", "comments"]
DEFAULT_REVIEWERS = {"hi": "Aditya Saxena", "mr": "Aditya Pise", "bn": "Saunak", "ta": "Dhiraj"}


def to_sheet_row(row: dict, reviewer: str) -> dict:
    return {
        "id": row["candidate_id"], "language": row["language"], "context": row["context"],
        "question": row["question"], "answer_text": row["answers"]["text"][0],
        "auto_warnings": ";".join(row["warnings"]), "reviewer": reviewer,
        "evidence_sentence": row.get("evidence_sentence", ""), "question_type": row.get("question_type") or "",
        "domain": row.get("domain") or row.get("domain_hint") or "", "difficulty": row.get("difficulty") or "",
        "context_id": row["context_id"], "document_id": row["document_id"],
        "source_url": row.get("source_url") or "", "page_start": row.get("page_start") or "",
        "answer_start": row["answers"]["answer_start"][0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", default="all")
    parser.add_argument("--second-sample", type=int, default=0, help="Items for a second reviewer (IAA)")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    for lang in selected_languages(args.languages):
        folder = LANGUAGES[lang]["folder"]
        rows = read_jsonl(VALIDATED_DIR / f"{lang}_dedup.jsonl")
        path = REVIEW_DIR / f"{folder}_review.csv"
        existing = {r["id"]: r for r in read_csv(path)}
        sheet = []
        for row in rows:
            fresh = to_sheet_row(row, DEFAULT_REVIEWERS[lang])
            old = existing.pop(row["candidate_id"], None)
            if old:
                fresh.update({k: old.get(k, "") for k in FILLED_BY_REVIEWER})
            sheet.append(fresh)
        # Keep reviewed rows whose candidate disappeared (e.g. after re-running dedup),
        # so no reviewer work is ever lost; build_dataset.py ignores them.
        sheet.extend(existing.values())
        write_csv(path, sheet, REVIEW_FIELDS)
        reviewed = sum(bool((r.get("status") or "").strip()) for r in sheet)
        print(f"[{lang}] {len(sheet)} rows ({reviewed} already reviewed) -> {rel(path)}")

        if args.second_sample and rows:
            second_path = REVIEW_DIR / f"{folder}_review_second.csv"
            if second_path.exists():
                print(f"  {rel(second_path)} exists - not overwritten")
                continue
            picked = random.Random(args.seed).sample(rows, min(args.second_sample, len(rows)))
            write_csv(second_path, [to_sheet_row(r, "") for r in picked], REVIEW_FIELDS)
            print(f"  second-reviewer sample of {len(picked)} -> {rel(second_path)}")


if __name__ == "__main__":
    main()
