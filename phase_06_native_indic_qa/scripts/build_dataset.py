"""Step 10: build the final SQuAD-compatible IndicAgriQA-Native files.

Inputs: validated_qa/<lang>_dedup.jsonl, human_review/<language>_review.csv and
data/native_indic_qa/leakage_exclusions.json.

Rules:
- Accept                    -> included (human_validated = true)
- Needs correction + fix    -> corrected question/answer applied; the corrected
                               answer must still be an exact substring of the context
- Needs correction, no fix  -> left out (still pending)
- Reject                    -> left out
- not yet reviewed          -> left out, unless --include-unreviewed (human_validated = false)
- documents / contexts listed in leakage_exclusions.json -> always left out

IDs (NATIVE_HI_000001, ...) are assigned here, in source order, so they are
contiguous in the released files. The candidate id is kept in metadata.

Usage:
    python phase_06_native_indic_qa/scripts/build_dataset.py
    python phase_06_native_indic_qa/scripts/build_dataset.py --include-unreviewed
"""

from __future__ import annotations

import argparse
from collections import Counter

from common import (DATA_DIR, FINAL_DIR, LANGUAGES, RESULTS_DIR, REVIEW_DIR, VALIDATED_DIR, nfc, read_csv,
                    read_json, read_jsonl, rel, selected_languages, write_csv, write_json, write_jsonl)

METADATA_FIELDS = ["id", "language", "candidate_id", "context_id", "document_id", "domain", "question_type",
                   "difficulty", "source_url", "page_number", "region", "human_validated", "review_status",
                   "reviewer", "generation_model", "prompt_version"]


def to_shared_schema(record: dict) -> dict:
    answers = record["answers"]
    return {
        "id": record["id"],
        "language": record["language"],
        "split": "test",  # Phase 6 is a held-out, test-only benchmark
        "source_document_id": record["metadata"]["document_id"],
        "title": record["title"],
        "context": record["context"],
        "question": record["question"],
        "answers": [{"text": t, "answer_start": s} for t, s in zip(answers["text"], answers["answer_start"])],
        "metadata": record["metadata"],
    }


def norm_status(value: str) -> str:
    v = (value or "").strip().lower()
    if v.startswith("accept"):
        return "accept"
    if v.startswith("needs") or "correct" in v:
        return "needs_correction"
    if v.startswith("reject"):
        return "reject"
    return "unreviewed"


def apply_correction(row: dict, review: dict) -> tuple[dict | None, str]:
    question = nfc((review.get("corrected_question") or "").strip()) or row["question"]
    answer = nfc((review.get("corrected_answer") or "").strip()) or row["answers"]["text"][0]
    if question == row["question"] and answer == row["answers"]["text"][0]:
        return None, "needs_correction_without_fix"
    context = row["context"]
    starts = []
    i = context.find(answer)
    while i != -1:
        starts.append(i)
        i = context.find(answer, i + 1)
    if not starts:
        return None, "corrected_answer_not_in_context"
    old = row["answers"]["answer_start"][0]
    start = min(starts, key=lambda s: abs(s - old))  # occurrence nearest the original answer
    fixed = dict(row, question=question, answers={"text": [answer], "answer_start": [start]})
    return fixed, "corrected"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", default="all")
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="Also include candidates no reviewer has marked yet (flagged human_validated=false)")
    args = parser.parse_args()

    exclusions = read_json(DATA_DIR / "leakage_exclusions.json")
    if not exclusions or not exclusions.get("checks_run"):
        print("WARNING: no leakage check has been run against the Phase 4 corpus "
              "(run check_data_leakage.py). Building anyway; this is recorded in the build report.")
        exclusions = exclusions or {"checks_run": [], "document_ids": [], "context_ids": []}
    bad_docs, bad_ctx = set(exclusions["document_ids"]), set(exclusions["context_ids"])

    all_rows, metadata, report = [], [], {"leakage_checks_run": exclusions["checks_run"], "languages": {}}
    for lang in selected_languages(args.languages):
        cfg = LANGUAGES[lang]
        reviews = {r["id"]: r for r in read_csv(REVIEW_DIR / f"{cfg['folder']}_review.csv")}
        outcome = Counter()
        final = []
        for row in read_jsonl(VALIDATED_DIR / f"{lang}_dedup.jsonl"):
            if row["document_id"] in bad_docs or row["context_id"] in bad_ctx:
                outcome["excluded_leakage"] += 1
                continue
            review = reviews.get(row["candidate_id"], {})
            status = norm_status(review.get("status", ""))
            if status == "reject":
                outcome["rejected"] += 1
                continue
            if status == "needs_correction":
                row, how = apply_correction(row, review)
                outcome[how] += 1
                if row is None:
                    continue
            elif status == "unreviewed":
                if not args.include_unreviewed:
                    outcome["unreviewed_left_out"] += 1
                    continue
                outcome["unreviewed_included"] += 1
            else:
                outcome["accepted"] += 1
            final.append((row, status, review))

        records = []
        for n, (row, status, review) in enumerate(final, start=1):
            start, text = row["answers"]["answer_start"][0], row["answers"]["text"][0]
            assert row["context"][start:start + len(text)] == text, row["candidate_id"]
            record = {
                "id": f"{cfg['prefix']}_{n:06d}",
                "title": row.get("title") or "",
                "language": lang,
                "context": row["context"],
                "question": row["question"],
                "answers": {"text": [text], "answer_start": [start]},
                "metadata": {
                    "document_id": row["document_id"],
                    "context_id": row["context_id"],
                    "candidate_id": row["candidate_id"],
                    "source_url": row.get("source_url"),
                    "page_number": row.get("page_start"),
                    "page_end": row.get("page_end"),
                    "domain": row.get("domain") or row.get("domain_hint"),
                    "region": row.get("region"),
                    "difficulty": row.get("difficulty"),
                    "question_type": row.get("question_type"),
                    "generation_method": "llm",
                    "generation_model": row.get("model"),
                    "prompt_version": row.get("prompt_version"),
                    "human_validated": status in ("accept", "needs_correction"),
                    "review_status": status,
                    # Sheets are pre-filled with the language owner, so only credit a
                    # reviewer once they have actually set a status.
                    "reviewer": ((review.get("reviewer") or "").strip() or None) if status != "unreviewed"
                    else None,
                },
            }
            records.append(record)
        out = FINAL_DIR / f"{cfg['folder']}.jsonl"
        write_jsonl(out, records)
        report["languages"][lang] = {"final_records": len(records), **dict(outcome)}
        print(f"[{lang}] {len(records)} records -> {rel(out)}  {dict(outcome)}")

    # test.jsonl and metadata.csv always cover every language file present, so a
    # --languages run for one language does not drop the others.
    for lang, cfg in LANGUAGES.items():
        for record in read_jsonl(FINAL_DIR / f"{cfg['folder']}.jsonl"):
            all_rows.append(record)
            m = record["metadata"]
            metadata.append({"id": record["id"], "language": lang,
                             **{k: m.get(k) for k in METADATA_FIELDS if k in m}})
    previous = read_json(RESULTS_DIR / "build_report.json", {}) or {}
    report["languages"] = {**previous.get("languages", {}), **report["languages"]}
    write_jsonl(FINAL_DIR / "test.jsonl", all_rows)
    # Same records in the team repo's shared QA contract (schemas/qa.schema.json in
    # adipise5/AgXQA-Indic), which shared.validate / shared.evaluate consume in Phase 7:
    # split + source_document_id required, answers as a list of {text, answer_start}.
    write_jsonl(FINAL_DIR / "test.shared.jsonl", (to_shared_schema(r) for r in all_rows))
    write_csv(DATA_DIR / "metadata.csv", metadata, METADATA_FIELDS)
    report["total_records"] = len(all_rows)
    report["include_unreviewed"] = args.include_unreviewed
    write_json(RESULTS_DIR / "build_report.json", report)
    print(f"Total {len(all_rows)} records -> {rel(FINAL_DIR / 'test.jsonl')}")


if __name__ == "__main__":
    main()
