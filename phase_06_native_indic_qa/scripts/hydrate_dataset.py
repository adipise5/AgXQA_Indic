"""Rebuild the full dataset (with passage text) from the public release.

The committed release (data/native_indic_qa/final/*.jsonl) contains questions,
answers, offsets and a SHA-256 of each context, but no passage text: Vikaspedia
content may not be redistributed without permission. Anyone can regenerate the
passages from the public source URLs and re-attach them:

    python phase_06_native_indic_qa/scripts/download_documents.py
    python phase_06_native_indic_qa/scripts/extract_text.py
    python phase_06_native_indic_qa/scripts/clean_text.py
    python phase_06_native_indic_qa/scripts/chunk_documents.py
    python phase_06_native_indic_qa/scripts/hydrate_dataset.py

Every context is checked byte-for-byte against its context_sha256, and every
answer span is re-checked against the rebuilt context. If a source page has
changed since 2026-11-18, the affected records are reported and left out (never
silently altered).

Output: data/native_indic_qa/full/<language>.jsonl, full/test.jsonl and
full/test.shared.jsonl (the team repo's shared QA schema).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from common import (CONTEXTS_DIR, FINAL_DIR, FULL_DIR, LANGUAGES, RESULTS_DIR, read_jsonl, rel,
                    selected_languages, sha256_text, write_json, write_jsonl)
from build_dataset import to_shared_schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", default="all")
    args = parser.parse_args()

    report, all_rows = {}, []
    for lang in selected_languages(args.languages):
        folder = LANGUAGES[lang]["folder"]
        release = read_jsonl(FINAL_DIR / f"{folder}.jsonl")
        contexts = read_jsonl(CONTEXTS_DIR / f"{lang}_contexts.jsonl")
        if release and not contexts:
            raise SystemExit(f"No contexts for {lang}: run download/extract/clean/chunk first (see --help).")
        by_id = {c["context_id"]: c for c in contexts}
        by_hash = defaultdict(list)  # fallback if context numbering shifted
        for c in contexts:
            by_hash[sha256_text(c["context"])].append(c)

        outcome, rows = Counter(), []
        for rec in release:
            ctx = by_id.get(rec["context_id"])
            if ctx is None or sha256_text(ctx["context"]) != rec["context_sha256"]:
                matches = by_hash.get(rec["context_sha256"], [])
                ctx = matches[0] if matches else None
                outcome["matched_by_hash" if ctx else "context_changed_or_missing"] += 1
                if ctx is None:
                    continue
            answer = rec["answers"][0]
            text, start = answer["text"], answer["answer_start"]
            if ctx["context"][start:start + len(text)] != text:
                outcome["answer_span_mismatch"] += 1
                continue
            outcome["hydrated"] += 1
            rows.append({"id": rec["id"], "title": ctx.get("title") or "", "language": lang,
                         "context": ctx["context"], "question": rec["question"],
                         "answers": {"text": [text], "answer_start": [start]}, "metadata": rec["metadata"]})
        write_jsonl(FULL_DIR / f"{folder}.jsonl", rows)
        all_rows.extend(rows)
        report[lang] = {"release_records": len(release), **dict(outcome)}
        print(f"[{lang}] {outcome['hydrated']}/{len(release)} records hydrated  {dict(outcome)}")

    write_jsonl(FULL_DIR / "test.jsonl", all_rows)
    write_jsonl(FULL_DIR / "test.shared.jsonl", (to_shared_schema(r) for r in all_rows))
    write_json(RESULTS_DIR / "hydration_report.json", report)
    print(f"{len(all_rows)} records -> {rel(FULL_DIR / 'test.jsonl')} and test.shared.jsonl")


if __name__ == "__main__":
    main()
