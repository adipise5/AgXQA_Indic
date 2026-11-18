"""Step 8b: remove duplicate and surplus questions among validated candidates.

Within each language (candidates with fewer warnings are preferred):
- exact duplicate questions (after normalisation)
- fuzzy duplicate questions (rapidfuzz token_sort_ratio >= threshold)
- two questions on the same context with the same answer span
- more than max_questions_per_context questions on one context

Writes validated_qa/<lang>_dedup.jsonl (kept candidates only) and
results/deduplication_report.json.

Usage:
    python phase_06_native_indic_qa/scripts/deduplicate.py --languages all
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from rapidfuzz import fuzz, process

from common import (RESULTS_DIR, VALIDATED_DIR, load_config, norm_for_match, read_json, read_jsonl,
                    selected_languages, write_json, write_jsonl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", default="all")
    args = parser.parse_args()
    cfg = load_config()["deduplication"]

    report = read_json(RESULTS_DIR / "deduplication_report.json", {}) or {}
    for lang in selected_languages(args.languages):
        rows = [r for r in read_jsonl(VALIDATED_DIR / f"{lang}_validated.jsonl") if r["status"] == "passed"]
        order = {cid: i for i, cid in enumerate(dict.fromkeys(r["context_id"] for r in rows))}
        rows.sort(key=lambda r: (order[r["context_id"]], len(r["warnings"]), r["candidate_id"]))

        kept, kept_keys, removed = [], [], Counter()
        removed_rows: list[dict] = []
        seen_exact: set[str] = set()
        spans_by_context: dict[str, set] = defaultdict(set)
        per_context: Counter = Counter()

        def drop(row: dict, reason: str) -> None:
            removed[reason] += 1
            removed_rows.append({"candidate_id": row["candidate_id"], "language": lang, "reason": reason,
                                 "question": row["question"], "answer": row["answers"]["text"][0],
                                 "domain": row.get("domain"), "question_type": row.get("question_type")})

        for row in rows:
            key = norm_for_match(row["question"])
            span = (row["answers"]["answer_start"][0], row["answers"]["text"][0])
            if key in seen_exact:
                drop(row, "exact_duplicate_question")
                continue
            if kept_keys and process.extractOne(key, kept_keys, scorer=fuzz.token_sort_ratio,
                                                score_cutoff=cfg["fuzzy_question_threshold"]):
                drop(row, "fuzzy_duplicate_question")
                continue
            if span in spans_by_context[row["context_id"]]:
                drop(row, "same_answer_same_context")
                continue
            if per_context[row["context_id"]] >= cfg["max_questions_per_context"]:
                drop(row, "over_per_context_cap")
                continue
            seen_exact.add(key)
            kept_keys.append(key)
            spans_by_context[row["context_id"]].add(span)
            per_context[row["context_id"]] += 1
            kept.append(row)

        write_jsonl(VALIDATED_DIR / f"{lang}_dedup.jsonl", kept)
        write_jsonl(VALIDATED_DIR / f"{lang}_dedup_removed.jsonl", removed_rows)
        report[lang] = {"passed_validation": len(rows), "kept": len(kept), "removed": dict(removed)}
        print(f"[{lang}] kept {len(kept)} of {len(rows)} validated candidates; removed {dict(removed)}")
    write_json(RESULTS_DIR / "deduplication_report.json", report)


if __name__ == "__main__":
    main()
