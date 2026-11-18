"""Step 5 / 10: data-leakage checks for the held-out native benchmark.

Phase 6 is test-only, so its documents and passages must not appear in any
training or adaptation data. Checks, in order of strength:

1. Phase 4 manifest (--phase4-manifest): identical file SHA-256, identical
   normalised source URL, near-identical titles (flagged for a human decision).
2. Passage overlap with any training corpus (--phase4-corpus, --compare NAME=PATH):
   each Phase 6 context is split into word 5-grams and we measure the share of
   them found anywhere in that corpus. Use it for the Indic-AEC corpus (Phase 4-5)
   and for the AgXQA-Indic translated data (Phase 2-3 fine-tuning).
3. Internal: near-duplicate contexts across different Phase 6 documents.

Writes results/data_leakage_report.json and data/native_indic_qa/leakage_exclusions.json
(used by build_dataset.py to drop contaminated documents/contexts).

Usage:
    python phase_06_native_indic_qa/scripts/check_data_leakage.py \
        --phase4-manifest phase_04_indic_aec_corpus/sources.csv \
        --phase4-corpus path/to/indic_aec_corpus --text-field text \
        --compare agxqa_indic=path/to/multilingual_train.jsonl:context
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rapidfuzz import fuzz

from common import (CONTEXTS_DIR, DATA_DIR, LANGUAGES, RESULTS_DIR, load_config, load_manifest,
                    norm_for_match, normalize_url, read_csv, write_json)


def pick(row: dict, *names: str) -> str:
    for name in names:
        if row.get(name):
            return row[name].strip()
    return ""


def shingles(text: str, n: int) -> set[int]:
    words = norm_for_match(text).split()
    return {hash(tuple(words[i:i + n])) for i in range(max(0, len(words) - n + 1))}


def iter_corpus_texts(path: Path, text_field: str):
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        suffix = file.suffix.lower()
        try:
            if suffix == ".jsonl":
                with file.open("r", encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        if line.strip():
                            obj = json.loads(line)
                            yield str(obj.get(text_field, "")) if isinstance(obj, dict) else str(obj)
            elif suffix == ".json":
                obj = json.loads(file.read_text(encoding="utf-8", errors="replace"))
                for item in obj if isinstance(obj, list) else [obj]:
                    yield str(item.get(text_field, "")) if isinstance(item, dict) else str(item)
            elif suffix in (".txt", ".md", ".csv", ".tsv"):
                with file.open("r", encoding="utf-8", errors="replace") as stream:
                    buffer = []
                    for line in stream:  # stream in blocks to bound memory
                        buffer.append(line)
                        if len(buffer) >= 20000:
                            yield "".join(buffer)
                            buffer = []
                    if buffer:
                        yield "".join(buffer)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            print(f"  skipped {file}: {error}")


def load_contexts() -> list[dict]:
    rows = []
    for lang in LANGUAGES:
        path = CONTEXTS_DIR / f"{lang}_contexts.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def passage_overlap(contexts: list[dict], ctx_shingles: dict, corpus: Path, field: str, cfg: dict) -> dict:
    """Stream a (possibly large) corpus once, keeping only shingles that also occur
    in Phase 6 contexts, so memory stays small."""
    wanted = set().union(*ctx_shingles.values()) if ctx_shingles else set()
    found: set[int] = set()
    texts = 0
    for text in iter_corpus_texts(corpus, field):
        texts += 1
        words = norm_for_match(text).split()
        for i in range(max(0, len(words) - cfg["shingle_size"] + 1)):
            h = hash(tuple(words[i:i + cfg["shingle_size"]]))
            if h in wanted:
                found.add(h)
    hits, shares = {}, []
    for c in contexts:
        sh = ctx_shingles[c["context_id"]]
        share = len(sh & found) / len(sh) if sh else 0.0
        shares.append(share)
        if share >= cfg["containment_threshold"]:
            hits[c["context_id"]] = round(share, 3)
    return {"corpus": str(corpus), "text_field": field, "text_blocks_scanned": texts,
            "contexts_checked": len(contexts), "contexts_over_threshold": len(hits),
            "max_containment": round(max(shares, default=0.0), 3),
            "contexts_with_any_shared_5gram": sum(s > 0 for s in shares), "contaminated_contexts": hits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase4-manifest", type=Path, help="Phase 4 sources CSV (hash / URL / title)")
    parser.add_argument("--phase4-corpus", type=Path, help="Directory or file of Phase 4 corpus text")
    parser.add_argument("--text-field", default="text", help="Text field in Phase 4 .jsonl/.json files")
    parser.add_argument("--compare", action="append", default=[], metavar="NAME=PATH[:FIELD]",
                        help="Extra training corpus to check passage overlap against (repeatable)")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if any leakage is found")
    args = parser.parse_args()
    cfg = load_config()["leakage"]
    n = cfg["shingle_size"]

    phase6 = load_manifest()
    if not phase6:
        raise SystemExit("No Phase 6 manifest yet - run download_documents.py first.")
    contexts = load_contexts()
    ctx_shingles = {c["context_id"]: shingles(c["context"], n) for c in contexts}
    excluded_docs: set[str] = set()
    contaminated: dict[str, float] = {}
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "method": {"shingle": f"word {n}-grams on normalised text",
                         "containment_threshold": cfg["containment_threshold"],
                         "title_similarity_threshold": cfg["title_similarity_threshold"]},
              "phase6": {"documents": len(phase6), "contexts": len(contexts)}}

    # 1. Phase 4 manifest
    if args.phase4_manifest:
        p4 = read_csv(args.phase4_manifest)
        findings = []
        p4_hashes = {pick(r, "document_hash", "sha256", "hash"): pick(r, "document_id", "id") for r in p4}
        p4_urls = {normalize_url(pick(r, "source_url", "url")): pick(r, "document_id", "id") for r in p4}
        p4_titles = [(norm_for_match(pick(r, "document_title", "title")), pick(r, "document_id", "id"))
                     for r in p4 if pick(r, "document_title", "title")]
        p4_hashes.pop("", None)
        p4_urls.pop("", None)
        for row in phase6:
            doc_id = row["document_id"]
            if row.get("document_hash") in p4_hashes:
                findings.append({"document_id": doc_id, "check": "identical_file_hash",
                                 "phase4_document": p4_hashes[row["document_hash"]]})
                excluded_docs.add(doc_id)
            url = normalize_url(row.get("source_url", ""))
            if url and url in p4_urls:
                findings.append({"document_id": doc_id, "check": "identical_source_url",
                                 "phase4_document": p4_urls[url]})
                excluded_docs.add(doc_id)
            title = norm_for_match(row.get("document_title", ""))
            for p4_title, p4_id in p4_titles if title else []:
                score = fuzz.token_sort_ratio(title, p4_title)
                if score >= cfg["title_similarity_threshold"]:
                    # Generic titles repeat, so this is flagged for a human, not excluded.
                    findings.append({"document_id": doc_id, "check": "similar_title_review_manually",
                                     "phase4_document": p4_id, "score": round(score, 1)})
        report["phase4_manifest"] = {"path": str(args.phase4_manifest), "phase4_documents": len(p4),
                                     "findings": findings}
        print(f"Phase 4 manifest: {len(p4)} documents, {len(findings)} findings.")

    # 2. Passage overlap with training corpora
    corpora = []
    if args.phase4_corpus:
        corpora.append(("indic_aec_phase4", args.phase4_corpus, args.text_field))
    for spec in args.compare:
        name, _, rest = spec.partition("=")
        path, _, field = rest.rpartition(":") if ":" in rest[2:] else (rest, "", "text")
        corpora.append((name, Path(path), field or "text"))
    report["passage_overlap"] = {}
    for name, path, field in corpora:
        result = passage_overlap(contexts, ctx_shingles, path, field, cfg)
        report["passage_overlap"][name] = result
        contaminated.update(result["contaminated_contexts"])
        per_doc = defaultdict(list)
        for c in contexts:
            per_doc[c["document_id"]].append(c["context_id"] in result["contaminated_contexts"])
        for doc_id, flags in per_doc.items():
            if sum(flags) and sum(flags) >= 0.2 * len(flags):  # a fifth of a document -> drop the document
                excluded_docs.add(doc_id)
        print(f"Overlap vs {name}: {result['text_blocks_scanned']} text blocks, "
              f"{result['contexts_over_threshold']} contexts over threshold, max containment "
              f"{result['max_containment']:.0%}, {result['contexts_with_any_shared_5gram']} share any 5-gram.")

    # 3. Internal near-duplicates across different Phase 6 documents
    index = defaultdict(set)
    for c in contexts:
        for h in ctx_shingles[c["context_id"]]:
            index[h].add(c["document_id"])
    cross = []
    for c in contexts:
        sh = ctx_shingles[c["context_id"]]
        shared = sum(1 for h in sh if len(index[h] - {c["document_id"]}) > 0)
        share = shared / len(sh) if sh else 0.0
        if share >= cfg["containment_threshold"]:
            cross.append({"context_id": c["context_id"], "containment_in_other_documents": round(share, 3)})
    report["internal_cross_document_duplicates"] = {"contexts_over_threshold": len(cross), "contexts": cross}
    print(f"Internal: {len(cross)} contexts largely repeated in another Phase 6 document.")

    checks_run = (["phase4_manifest"] if args.phase4_manifest else []) + [name for name, _, _ in corpora]
    report["checks_run"] = checks_run
    report["excluded_documents"] = sorted(excluded_docs)
    report["contaminated_contexts"] = contaminated
    report["leakage_count"] = len(contaminated) + len(excluded_docs)
    write_json(RESULTS_DIR / "data_leakage_report.json", report)
    write_json(DATA_DIR / "leakage_exclusions.json",
               {"checks_run": checks_run, "document_ids": sorted(excluded_docs),
                "context_ids": sorted(contaminated)})
    if not checks_run:
        print("No external corpus given - only the internal check ran.")
    print(f"Excluded documents: {len(excluded_docs)} | contaminated contexts: {len(contaminated)} "
          f"-> results/data_leakage_report.json")
    if args.strict and (excluded_docs or contaminated):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
