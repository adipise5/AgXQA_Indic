"""Step 5 / 10: make sure no Phase 6 benchmark document leaks into the Phase 4-5
Indic-AEC domain-adaptation corpus.

Checks, in order of strength:
1. identical file SHA-256 between the two manifests
2. identical normalised source URL
3. near-identical document titles (alternative editions / re-uploads)
4. passage overlap: each Phase 6 context is split into word 5-grams and we measure
   what fraction of them also occur anywhere in the Phase 4 corpus text

Get the Phase 4 manifest and corpus from Aditya Pise. Any of the inputs can be
omitted, but run it with the corpus text before Phase 7 evaluation.

Writes results/leakage_report.json and data/native_indic_qa/leakage_exclusions.json
(used by build_dataset.py to drop contaminated documents/contexts).

Usage:
    python phase_06_native_indic_qa/scripts/check_data_leakage.py \
        --phase4-manifest path/to/indic_aec_manifest.csv \
        --phase4-corpus path/to/indic_aec_corpus_dir_or_file --text-field text
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz

from common import (CONTEXTS_DIR, DATA_DIR, LANGUAGES, RESULTS_DIR, load_config, load_manifest,
                    norm_for_match, normalize_url, read_csv, write_json)


norm_url = normalize_url


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
                items = obj if isinstance(obj, list) else [obj]
                for item in items:
                    yield str(item.get(text_field, "")) if isinstance(item, dict) else str(item)
            elif suffix in (".txt", ".md", ".csv", ".tsv"):
                with file.open("r", encoding="utf-8", errors="replace") as stream:
                    buffer = []
                    for line in stream:  # stream in ~1 MB blocks to bound memory
                        buffer.append(line)
                        if len(buffer) >= 20000:
                            yield "".join(buffer)
                            buffer = []
                    if buffer:
                        yield "".join(buffer)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            print(f"  skipped {file}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase4-manifest", type=Path, help="CSV with document hashes / URLs / titles")
    parser.add_argument("--phase4-corpus", type=Path, help="Directory or file of Phase 4 corpus text")
    parser.add_argument("--text-field", default="text", help="Field holding text in .jsonl/.json corpus files")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if any leakage is found")
    args = parser.parse_args()
    cfg = load_config()["leakage"]
    n = cfg["shingle_size"]

    phase6 = load_manifest()
    if not phase6:
        raise SystemExit("No Phase 6 manifest yet - run download_documents.py first.")
    findings = []
    excluded_docs: set[str] = set()

    if args.phase4_manifest:
        p4 = read_csv(args.phase4_manifest)
        p4_hashes = {pick(r, "document_hash", "sha256", "hash"): pick(r, "document_id", "id") for r in p4}
        p4_urls = {norm_url(pick(r, "source_url", "url")): pick(r, "document_id", "id") for r in p4}
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
            url = norm_url(row.get("source_url", ""))
            if url and url in p4_urls:
                findings.append({"document_id": doc_id, "check": "identical_source_url",
                                 "phase4_document": p4_urls[url]})
                excluded_docs.add(doc_id)
            title = norm_for_match(row.get("document_title", ""))
            for p4_title, p4_id in p4_titles if title else []:
                score = fuzz.token_sort_ratio(title, p4_title)
                if score >= cfg["title_similarity_threshold"]:
                    # Titles alone are weak evidence (generic titles repeat), so this is
                    # flagged for a human decision, not excluded automatically.
                    findings.append({"document_id": doc_id, "check": "similar_title_review_manually",
                                     "phase4_document": p4_id, "score": round(score, 1)})
        print(f"Manifest checks against {len(p4)} Phase 4 documents done.")

    contaminated_contexts: dict[str, float] = {}
    if args.phase4_corpus:
        contexts = []
        for lang in LANGUAGES:
            path = CONTEXTS_DIR / f"{lang}_contexts.jsonl"
            if path.exists():
                with path.open(encoding="utf-8") as stream:
                    contexts.extend(json.loads(line) for line in stream if line.strip())
        ctx_shingles = {c["context_id"]: shingles(c["context"], n) for c in contexts}
        wanted = set().union(*ctx_shingles.values()) if ctx_shingles else set()
        found: set[int] = set()
        texts = 0
        # Stream the (large) Phase 4 corpus once; only shingles that also occur in
        # Phase 6 contexts are kept, so memory stays small.
        for text in iter_corpus_texts(args.phase4_corpus, args.text_field):
            texts += 1
            words = norm_for_match(text).split()
            for i in range(max(0, len(words) - n + 1)):
                h = hash(tuple(words[i:i + n]))
                if h in wanted:
                    found.add(h)
        per_doc = defaultdict(list)
        for c in contexts:
            sh = ctx_shingles[c["context_id"]]
            share = len(sh & found) / len(sh) if sh else 0.0
            per_doc[c["document_id"]].append(share)
            if share >= cfg["containment_threshold"]:
                contaminated_contexts[c["context_id"]] = round(share, 3)
                findings.append({"document_id": c["document_id"], "context_id": c["context_id"],
                                 "check": "passage_overlap", "containment": round(share, 3)})
        for doc_id, shares in per_doc.items():
            hit = sum(s >= cfg["containment_threshold"] for s in shares)
            if hit and hit >= 0.2 * len(shares):  # a fifth of a document's passages -> drop the document
                excluded_docs.add(doc_id)
        print(f"Passage check: {len(contexts)} Phase 6 contexts vs {texts} Phase 4 text blocks; "
              f"{len(contaminated_contexts)} contexts over {cfg['containment_threshold']:.0%} {n}-gram containment.")

    checks_run = [name for name, flag in (("manifest", args.phase4_manifest), ("corpus", args.phase4_corpus)) if flag]
    report = {"checks_run": checks_run, "phase6_documents": len(phase6),
              "excluded_documents": sorted(excluded_docs),
              "contaminated_contexts": contaminated_contexts, "findings": findings}
    write_json(RESULTS_DIR / "leakage_report.json", report)
    write_json(DATA_DIR / "leakage_exclusions.json",
               {"checks_run": checks_run, "document_ids": sorted(excluded_docs),
                "context_ids": sorted(contaminated_contexts)})
    if not checks_run:
        print("No Phase 4 inputs given - nothing was checked. Pass --phase4-manifest and/or --phase4-corpus.")
    print(f"Excluded documents: {len(excluded_docs)} | contaminated contexts: {len(contaminated_contexts)} | "
          f"findings: {len(findings)} -> results/leakage_report.json")
    if args.strict and (excluded_docs or contaminated_contexts):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
