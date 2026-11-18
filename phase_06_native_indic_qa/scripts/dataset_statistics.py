"""Step 10b: dataset statistics, pipeline funnel and inter-annotator agreement.

Writes results/dataset_statistics.json and, if matplotlib is installed,
bar charts in results/plots/.

Inter-annotator agreement (Cohen's kappa on Accept / Needs correction / Reject)
is computed for a language only when human_review/<language>_review_second.csv
exists with reviewed rows; otherwise the report says single-reviewer validation.

Usage:
    python phase_06_native_indic_qa/scripts/dataset_statistics.py
"""

from __future__ import annotations

from collections import Counter

from common import (CONTEXTS_DIR, FINAL_DIR, GENERATED_DIR, LANGUAGES, RESULTS_DIR, REVIEW_DIR, VALIDATED_DIR,
                    load_config, load_manifest, read_csv, read_jsonl, word_count, write_json)
from build_dataset import norm_status


def mean(values) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 1) if values else None


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if not a:
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in labels) / (n * n)
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else 1.0


def agreement(lang: str) -> dict:
    folder = LANGUAGES[lang]["folder"]
    second = read_csv(REVIEW_DIR / f"{folder}_review_second.csv")
    if not second:
        return {"method": "single_reviewer"}
    primary = {r["id"]: norm_status(r.get("status", "")) for r in read_csv(REVIEW_DIR / f"{folder}_review.csv")}
    pairs = [(primary[r["id"]], norm_status(r.get("status", ""))) for r in second
             if r["id"] in primary and primary[r["id"]] != "unreviewed"
             and norm_status(r.get("status", "")) != "unreviewed"]
    if not pairs:
        return {"method": "single_reviewer", "note": "second-reviewer sheet exists but has no overlapping reviews"}
    a, b = zip(*pairs)
    return {"method": "two_reviewers_on_subset", "items": len(pairs),
            "percent_agreement": round(sum(x == y for x, y in pairs) / len(pairs), 3),
            "cohen_kappa": cohen_kappa(list(a), list(b))}


def plot(stats: dict) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out_dir = RESULTS_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    langs = [l for l in LANGUAGES if l in stats["languages"]]
    for key, title in (("domains", "QA pairs by domain"), ("question_types", "QA pairs by question type")):
        categories = sorted({c for l in langs for c in stats["languages"][l][key]})
        if not categories:
            continue
        fig, ax = plt.subplots(figsize=(10, 4.5))
        width = 0.8 / max(len(langs), 1)
        for i, lang in enumerate(langs):
            values = [stats["languages"][lang][key].get(c, 0) for c in categories]
            ax.bar([x + i * width for x in range(len(categories))], values, width, label=LANGUAGES[lang]["name"])
        ax.set_xticks([x + width * (len(langs) - 1) / 2 for x in range(len(categories))])
        ax.set_xticklabels(categories, rotation=30, ha="right")
        ax.set_title(title)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = out_dir / f"{key}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)
    return written


def main() -> None:
    targets = load_config()["targets"]
    manifest = load_manifest()
    stats = {"dataset": "IndicAgriQA-Native", "targets_per_language": targets, "languages": {}}
    total = 0
    for lang, cfg in LANGUAGES.items():
        rows = read_jsonl(FINAL_DIR / f"{cfg['folder']}.jsonl")
        docs = [m for m in manifest if m["language"] == lang]
        raw = read_jsonl(GENERATED_DIR / f"{lang}_raw.jsonl")
        validated = read_jsonl(VALIDATED_DIR / f"{lang}_validated.jsonl")
        if not (rows or docs or raw):
            continue
        meta = [r["metadata"] for r in rows]
        domains = Counter(m.get("domain") or "unknown" for m in meta)
        stats["languages"][lang] = {
            "qa_pairs": len(rows),
            "progress_vs_target": f"{len(rows)}/{targets['qa_pairs']}",
            "meets_minimum_viable": len(rows) >= targets["minimum_viable"],
            "documents_in_manifest": len(docs),
            "documents_in_final": len({m["document_id"] for m in meta}),
            "organizations": dict(Counter(d.get("organization") or "unknown" for d in docs)),
            "license_status": dict(Counter(d.get("license") or "unverified" for d in docs)),
            "unique_contexts": len({m["context_id"] for m in meta}),
            "human_validated": sum(bool(m.get("human_validated")) for m in meta),
            "irrigation_vs_other": {"irrigation": domains.get("irrigation", 0),
                                    "other": len(rows) - domains.get("irrigation", 0)},
            "domains": dict(domains.most_common()),
            "question_types": dict(Counter(m.get("question_type") or "unknown" for m in meta).most_common()),
            "difficulty": dict(Counter(m.get("difficulty") or "unknown" for m in meta).most_common()),
            "mean_context_words": mean(word_count(r["context"]) for r in rows),
            "mean_context_chars": mean(len(r["context"]) for r in rows),
            "mean_question_words": mean(word_count(r["question"]) for r in rows),
            "mean_answer_words": mean(word_count(r["answers"]["text"][0]) for r in rows),
            "funnel": {
                "contexts": len(read_jsonl(CONTEXTS_DIR / f"{lang}_contexts.jsonl")),
                "contexts_sent_to_llm": len({r["context_id"] for r in raw}),
                "candidates_generated": len(validated),
                "passed_automatic_validation": sum(v["status"] == "passed" for v in validated),
                "after_deduplication": len(read_jsonl(VALIDATED_DIR / f"{lang}_dedup.jsonl")),
                "final": len(rows),
            },
            "review_agreement": agreement(lang),
        }
        total += len(rows)
    stats["total_qa_pairs"] = total
    stats["plots"] = plot(stats)
    write_json(RESULTS_DIR / "dataset_statistics.json", stats)
    for lang, s in stats["languages"].items():
        print(f"[{lang}] {s['progress_vs_target']} QA | irrigation {s['irrigation_vs_other']['irrigation']} | "
              f"validated {s['human_validated']} | funnel {s['funnel']}")
    print(f"Total {total} QA pairs -> results/dataset_statistics.json")


if __name__ == "__main__":
    main()
