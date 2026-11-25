"""Verify Phase 5 QA inputs and audit adaptation data against held-out sets."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.qa_release import ROOT, VARIANTS, load_qa_release, sha256
from shared.validate import read_jsonl


def audit_corpus(corpus, phase6_contexts, validation_and_test):
    from phase_04_indic_aec_corpus.pipeline import Exclusions, audit_documents, load_heldout, shingles

    documents = read_jsonl(corpus)
    if not documents or len(documents) != len({row["document_id"] for row in documents}):
        raise ValueError("Adaptation corpus must contain unique document IDs")
    if any(not shingles(row["text"]) for row in documents):
        raise ValueError("Adaptation corpus contains an unscannable document")
    reserve = Exclusions()
    reserved_hits = [
        {"document_id": row["document_id"], "reason": reason}
        for row in documents
        if (reason := reserve.check(row["source_url"], row.get("sha256"))
            or reserve.check(row.get("resolved_url", row["source_url"])))
    ]
    phase2 = [{"context_id": f"phase2:{row['split']}:{row['id']}", "context": row["context"]}
              for row in validation_and_test]
    phase6 = load_heldout(phase6_contexts)
    phase2_hits = audit_documents(documents, phase2)
    phase6_hits = audit_documents(documents, phase6)
    return {
        "status": "overlap_found" if reserved_hits or phase2_hits or phase6_hits else "no_overlap_at_threshold",
        "method": "Normalized word 5-gram containment, including distributed corpus matches",
        "threshold": 0.5, "corpus_sha256": sha256(corpus), "document_count": len(documents),
        "phase_2_reference_rows": len(phase2), "phase_6_reference_windows": len(phase6),
        "phase_2_findings": phase2_hits, "phase_6_findings": phase6_hits,
        "phase_6_reservation_findings": reserved_hits,
    }


def heldout_rows(bundle, evaluation):
    # Reserve every recovered official held-out group, including pending English spans.
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    candidates = bundle / "candidates.jsonl"
    if sha256(candidates) != manifest["candidate_sha256"]:
        raise ValueError("Phase 2 held-out candidate checksum mismatch")
    official = {row["id"]: row for row in read_jsonl(candidates) if row["split"] != "train"}
    updated = []
    for row in evaluation:
        original = official.get(row["id"])
        if original is None or original["context"] != row["context"]:
            updated.append({**row, "id": row["id"] + ":evaluation"})
    return list(official.values()) + updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--corpus", type=Path, help="Exact Phase 4 document JSONL used for MLM")
    parser.add_argument("--phase6-contexts", type=Path, help="Verified Phase 6 reconstruction directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        splits, identity = load_qa_release(config, args.bundle_dir, args.variant)
        report = {"dataset": identity, "experiment_executed": False}
        missing = [name for name, path, check in (
            ("actual_adaptation_corpus", args.corpus, Path.is_file),
            ("verified_phase_6_contexts", args.phase6_contexts, Path.is_dir),
        ) if path is None or not check(path)]
        if missing:
            report["adaptation_overlap"] = {"status": "blocked_missing_inputs", "missing": missing}
        else:
            pin = json.loads((ROOT / config["dataset_contract"]).read_text(encoding="utf-8"))
            bundle = args.bundle_dir or ROOT / pin["bundle_dir"]
            heldout = heldout_rows(bundle, splits["validation"] + splits["test"])
            report["adaptation_overlap"] = audit_corpus(args.corpus, args.phase6_contexts, heldout)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
        print(json.dumps(report, indent=2))
        if report["adaptation_overlap"]["status"] != "no_overlap_at_threshold":
            parser.exit(2, "Phase 5 adaptation audit is blocked or detected overlap.\n")
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f"Phase 5 preflight: {error}\n")


if __name__ == "__main__":
    main()
