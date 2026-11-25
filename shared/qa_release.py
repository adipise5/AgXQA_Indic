"""Load the pinned Phase 2 splits used by both Phase 3 experiments and Phase 5."""

import hashlib
import json
from pathlib import Path

from shared.validate import LANGUAGES, read_jsonl, validate


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
VARIANTS = ("span_checked", "document_disjoint", "final")


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_qa_release(experiment_config, bundle_dir=None, variant=None):
    contract = json.loads((ROOT / experiment_config["dataset_contract"]).read_text(encoding="utf-8"))
    bundle = Path(bundle_dir) if bundle_dir is not None else ROOT / contract["bundle_dir"]
    variant = variant or contract["primary_variant"]
    if variant not in VARIANTS:
        raise ValueError(f"Unknown QA dataset variant: {variant}")
    manifest_path = bundle / "manifest.json"
    manifest_hash = sha256(manifest_path)
    if manifest_hash != contract["bundle_manifest_sha256"]:
        raise ValueError("Phase 2 bundle manifest differs from the pinned evaluation contract")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset_version"] != contract["dataset_version"]:
        raise ValueError("Phase 2 bundle version mismatch")

    if variant == contract["published_evaluation_variant"]:
        final = bundle / variant / "build_manifest.json"
        if not final.is_file():
            raise ValueError("Final Phase 2 data require completed human review and source adjudication")
        final_manifest = json.loads(final.read_text(encoding="utf-8"))
        if (final_manifest["status"] != "ready" or final_manifest["dataset_version"] != manifest["dataset_version"]
                or final_manifest["parent_bundle_manifest_sha256"] != manifest_hash):
            raise ValueError("Final data do not match the reviewed Phase 2 bundle")
        expected = {split: final_manifest["file_sha256"][f"{split}.jsonl"] for split in SPLITS}
    else:
        expected = contract["variant_split_sha256"][variant]

    counts, data, observed = {}, {}, {}
    splits_by_id = {}
    for split in SPLITS:
        path = bundle / variant / f"{split}.jsonl"
        observed[split] = sha256(path)
        if observed[split] != expected[split]:
            raise ValueError(f"Pinned {split} checksum mismatch")
        rows = read_jsonl(path)
        groups = {}
        converted = []
        for row in rows:
            original_id, language = row["original_id"], row["language"]
            if (row["split"] != split or row["id"] != f"{original_id}_{language}"
                    or splits_by_id.setdefault(original_id, split) != split):
                raise ValueError(f"Question has an invalid identity or split: {row['id']}")
            groups.setdefault(original_id, set()).add(language)
            answer = row["answers"]
            if len(answer["text"]) != len(answer["answer_start"]):
                raise ValueError(f"Mismatched answer arrays: {row['id']}")
            converted.append({**row, "id": original_id, "answers": [
                {"text": text, "answer_start": start} for text, start in zip(answer["text"], answer["answer_start"])
            ]})
        if any(languages != LANGUAGES for languages in groups.values()):
            raise ValueError(f"Incomplete five-language groups in {split}")
        counts[split] = validate(converted)
        data[split] = rows

    return data, {"dataset_version": contract["dataset_version"], "variant": variant,
                  "bundle_manifest_sha256": manifest_hash, "split_sha256": observed, "coverage": counts,
                  "reviewed_release": variant == contract["published_evaluation_variant"]}
