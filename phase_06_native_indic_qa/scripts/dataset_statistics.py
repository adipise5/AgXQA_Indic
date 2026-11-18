"""Step 10b: dataset statistics, quality tables, error analysis and plots.

Writes:
- results/dataset_statistics.json  Table 1 (language-wise statistics), Table 2 (domain
  distribution), Table 3 (quality assessment), pipeline funnel, source breakdown
  and inter-annotator agreement
- results/error_analysis.csv        every rejected / removed / flagged candidate with its reason
- results/plots/*.png               figures computed from the actual dataset

Human-review metrics stay null until reviewers fill the sheets: nothing is estimated.
Inter-annotator agreement (Cohen's kappa) is computed only when a second reviewer's
sheet (human_review/private/<language>_review_second.csv) has overlapping reviews.

Usage:
    python phase_06_native_indic_qa/scripts/dataset_statistics.py
"""

from __future__ import annotations

from collections import Counter
from statistics import mean

from common import (CONTEXTS_DIR, DOMAINS, FULL_DIR, LANGUAGES, QUESTION_TYPES, RESULTS_DIR, REVIEW_DIR,
                    REVIEW_PRIVATE_DIR, SOURCES_DIR, VALIDATED_DIR, load_config, load_manifest, read_csv,
                    read_json, read_jsonl, word_count, write_csv, write_json)
from build_dataset import norm_status

LANG_ORDER = ["hi", "mr", "bn", "ta"]
# Categorical slots 1-4 of the validated reference palette (adjacent-pair CVD-safe),
# assigned to languages in a fixed order so a language keeps its colour in every plot.
LANG_COLORS = {"hi": "#2a78d6", "mr": "#eb6834", "bn": "#1baf7a", "ta": "#eda100"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
DUPLICATE_REASONS = {"exact_duplicate_question", "fuzzy_duplicate_question", "same_answer_same_context"}

# Article-credit keywords -> publisher type (credits are recorded in the source CSVs' notes).
SOURCE_TYPES = [
    ("Press / magazine", ["अग्रोवन", "अॅग्रोवन", "தினமணி", "பசுமை விகடன்", "পত্র পত্রিকা"]),
    ("University / ICAR / KVK", ["பல்கலைக்கழக", "அறிவியல் நிலைய", "ஆராய்ச்சி நிலைய", "அனुसंधान", "अनुसंधान",
                                 "विद्यापीठ", "ICAR", "आई सी ए आर", "আই সি এ আর", "গবেষণা", "வளரும் வேளாண்மை"]),
    ("Government department", ["विभाग", "सरकार", "शासन", "पोर्टल,भारत", "किसान पोर्टल", "समेति", "महान्युज",
                               "দফতর", "সরকার"]),
    ("Vikaspedia content team", ["বিকাশপিডিয়া", "পোর্টাল কন", "পোর্টাল কন্টেন্ট", "पोर्टल विषय"]),
]


def avg(values) -> float | None:
    values = list(values)
    return round(mean(values), 1) if values else None


def pct(part: float, whole: float) -> float | None:
    return round(100 * part / whole, 1) if whole else None


def source_type(note: str) -> str:
    credit = note.split("Article credit:")[-1] if "Article credit:" in note else ""
    for label, keys in SOURCE_TYPES:
        if any(k in credit for k in keys):
            return label
    return "NGO / company / individual" if credit.strip() and "none found" not in credit else "No credit line"


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else 1.0


def review_sheet(folder: str, suffix: str = "") -> list[dict]:
    private = REVIEW_PRIVATE_DIR / f"{folder}_review{suffix}.csv"
    return read_csv(private if private.exists() else REVIEW_DIR / f"{folder}_review{suffix}.csv")


def agreement(folder: str, primary: dict[str, str]) -> dict:
    second = review_sheet(folder, "_second")
    pairs = [(primary[r["id"]], norm_status(r.get("status", ""))) for r in second
             if primary.get(r["id"], "unreviewed") != "unreviewed" and norm_status(r.get("status", "")) != "unreviewed"]
    if not pairs:
        return {"method": "single_reviewer", "items": 0, "cohen_kappa": None}
    a, b = zip(*pairs)
    return {"method": "two_reviewers_on_subset", "items": len(pairs),
            "percent_agreement": pct(sum(x == y for x, y in pairs), len(pairs)), "cohen_kappa": cohen_kappa(a, b)}


def language_stats(lang: str, manifest: list[dict], sources: list[dict]) -> tuple[dict, list[dict]]:
    folder = LANGUAGES[lang]["folder"]
    contexts = read_jsonl(CONTEXTS_DIR / f"{lang}_contexts.jsonl")
    validated = read_jsonl(VALIDATED_DIR / f"{lang}_validated.jsonl")
    dedup = read_jsonl(VALIDATED_DIR / f"{lang}_dedup.jsonl")
    removed = read_jsonl(VALIDATED_DIR / f"{lang}_dedup_removed.jsonl")
    final = read_jsonl(FULL_DIR / f"{folder}.jsonl")
    docs = [m for m in manifest if m["language"] == lang]
    meta = [r["metadata"] for r in final]
    reviews = {r["id"]: r for r in review_sheet(folder)}
    status = {cid: norm_status(r.get("status", "")) for cid, r in reviews.items()}
    reviewed = [r for cid, r in reviews.items() if status.get(cid) != "unreviewed"]
    passed = [v for v in validated if v["status"] == "passed"]
    span_fail = sum("answer_not_in_context" in v["fail_reasons"] for v in validated)
    duplicates = sum(r["reason"] in DUPLICATE_REASONS for r in removed)

    def score_share(field: str) -> float | None:
        scores = [int(r[field]) for r in reviewed if (r.get(field) or "").strip().isdigit()]
        return pct(sum(s >= 4 for s in scores), len(scores)) if scores else None

    table1 = {
        "documents_collected": len(sources),
        "unique_documents": len({d["document_hash"] for d in docs}),
        "documents_used": len(docs),
        "context_passages": len(contexts),
        "generated_qa_pairs": len(validated),
        "rejected_qa_pairs": len(validated) - len(final),
        "human_reviewed_qa_pairs": len(reviewed),
        "final_qa_pairs": len(final),
        "avg_context_words": avg(word_count(r["context"]) for r in final),
        "avg_context_xlmr_tokens": avg(c["xlmr_tokens"] for c in contexts if c.get("xlmr_tokens")),
        "avg_question_words": avg(word_count(r["question"]) for r in final),
        "avg_answer_words": avg(word_count(r["answers"]["text"][0]) for r in final),
    }
    table3 = {
        "answer_span_validity_pct": pct(len(validated) - span_fail, len(validated)),
        "automatic_validation_pass_pct": pct(len(passed), len(validated)),
        "human_review_coverage_pct": pct(len(reviewed), len(final)),
        "question_correctness_pct": score_share("question_correctness"),
        "answer_correctness_pct": score_share("answer_correctness"),
        "human_acceptance_rate_pct": pct(sum(status[r["id"]] in ("accept", "needs_correction") for r in reviewed),
                                         len(reviewed)) if reviewed else None,
        "duplicate_rate_pct": pct(duplicates, len(passed)),
        "data_leakage_count": None,  # filled from the leakage report
    }
    errors = []
    for v in validated:
        for reason in v["fail_reasons"]:
            errors.append({"candidate_id": v["candidate_id"], "language": lang, "stage": "automatic_validation",
                           "reason": reason, "question": v["question"], "answer": v["answer_raw"],
                           "question_type": v.get("question_type"), "domain": v.get("domain")})
    for r in removed:
        errors.append({**r, "stage": "deduplication"})
    by_cid = {v["candidate_id"]: v for v in validated}
    for r in final:
        v = by_cid.get(r["metadata"]["candidate_id"], {})
        for warning in v.get("warnings", []):
            errors.append({"candidate_id": v["candidate_id"], "language": lang, "stage": "review_flag",
                           "reason": warning, "question": r["question"], "answer": r["answers"]["text"][0],
                           "question_type": v.get("question_type"), "domain": v.get("domain")})
    stats = {
        "table1": table1,
        "table2_domains": {d: sum(m.get("domain") == d for m in meta) for d in DOMAINS},
        "table3": table3,
        "question_types": dict(Counter(m.get("question_type") or "unlabelled" for m in meta).most_common()),
        "difficulty": dict(Counter(m.get("difficulty") or "unlabelled" for m in meta).most_common()),
        "review_outcomes": dict(Counter(status[r["id"]] for r in reviewed)),
        "source_types": dict(Counter(source_type(s.get("notes", "")) for s in sources
                                     if s.get("include") == "yes").most_common()),
        "context_tokens": [c["xlmr_tokens"] for c in contexts if c.get("xlmr_tokens")],
        "answer_words": [word_count(r["answers"]["text"][0]) for r in final],
        "funnel": {"contexts": len(contexts), "generated": len(validated), "passed_validation": len(passed),
                   "after_dedup": len(dedup), "final": len(final)},
        "review_agreement": agreement(folder, status),
    }
    return stats, errors


# ---------------------------------------------------------------- plots
def plots(langs: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "text.color": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})
    out = RESULTS_DIR / "plots"
    out.mkdir(parents=True, exist_ok=True)
    order = [l for l in LANG_ORDER if l in langs]
    names = {l: LANGUAGES[l]["name"] for l in order}
    written = []

    def save(fig, name: str, note: str | None = None):
        if note:
            fig.text(0.01, 0.01, note, fontsize=8, color=INK_2, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
        written.append(name)

    def grouped(ax, categories: list[str], values: dict, labels: list[str] | None = None):
        width = 0.8 / len(order)
        for i, lang in enumerate(order):
            xs = [x + (i - (len(order) - 1) / 2) * width for x in range(len(categories))]
            ax.bar(xs, [values[lang].get(c, 0) for c in categories], width * 0.92,
                   color=LANG_COLORS[lang], label=names[lang])
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(labels or categories, rotation=25, ha="right")
        ax.grid(axis="x", visible=False)
        top = max((values[l].get(c, 0) for l in order for c in categories), default=0)
        ax.set_ylim(0, top * 1.18 if top else 1)  # headroom so the legend never sits on a bar
        ax.legend(frameon=False, ncol=len(order), loc="upper right")

    # 1 language distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = [langs[l]["table1"]["final_qa_pairs"] for l in order]
    bars = ax.bar([names[l] for l in order], counts, 0.6, color=[LANG_COLORS[l] for l in order])
    ax.bar_label(bars, labels=[f"{c:,}" for c in counts], padding=3, color=INK)
    ax.axhline(250, color=INK_2, linewidth=1, linestyle="--")
    ax.text(-0.45, 262, "target: 250 per language", color=INK_2, ha="left", va="bottom", fontsize=9)
    ax.set_ylim(0, max(counts) * 1.1)
    ax.set_title("Final QA pairs by language")
    ax.set_ylabel("QA pairs")
    ax.grid(axis="x", visible=False)
    save(fig, "language_distribution.png")

    # 2 domain distribution
    fig, ax = plt.subplots(figsize=(10, 4.6))
    grouped(ax, DOMAINS, {l: langs[l]["table2_domains"] for l in order},
            [d.replace("_", " ").replace("fertilizers nutrition", "fertilizers") for d in DOMAINS])
    ax.set_title("QA pairs by agricultural domain (generator labels, not yet reviewed)")
    ax.set_ylabel("QA pairs")
    save(fig, "domain_distribution.png")

    # 3 question types
    fig, ax = plt.subplots(figsize=(10, 4.6))
    grouped(ax, QUESTION_TYPES, {l: langs[l]["question_types"] for l in order})
    ax.set_title("QA pairs by question type")
    ax.set_ylabel("QA pairs")
    save(fig, "question_type_distribution.png")

    # 4 difficulty (share within language, so languages of different size compare)
    fig, ax = plt.subplots(figsize=(7, 4))
    levels = ["easy", "medium", "hard"]
    shades = {"easy": "#a9c9f0", "medium": "#2a78d6", "hard": "#0d3e7a"}  # one hue, light -> dark
    left = [0] * len(order)
    for level in levels:
        vals = [pct(langs[l]["difficulty"].get(level, 0), langs[l]["table1"]["final_qa_pairs"]) for l in order]
        ax.barh([names[l] for l in order], vals, left=left, color=shades[level], label=level,
                edgecolor=SURFACE, linewidth=2)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of QA pairs in language")
    ax.set_title("Difficulty (generator label)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    save(fig, "difficulty_distribution.png")

    # 5 context length (XLM-R tokens)
    edges = [(0, 100), (101, 200), (201, 300), (301, 384), (385, 512), (513, 10**6)]
    bucket_labels = ["0–100", "101–200", "201–300", "301–384", "385–512", "513+"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    grouped(ax, bucket_labels, {l: {lab: sum(lo <= t <= hi for t in langs[l]["context_tokens"])
                                     for lab, (lo, hi) in zip(bucket_labels, edges)} for l in order})
    ax.set_xticklabels(bucket_labels, rotation=0, ha="center")
    ax.set_title("Context length in XLM-R tokens (max_seq_length 384)")
    ax.set_ylabel("contexts")
    save(fig, "context_length_distribution.png")

    # 6 answer length (words)
    a_edges = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 10**6)]
    a_labels = ["1", "2", "3", "4–5", "6–8", "9+"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    grouped(ax, a_labels, {l: {lab: sum(lo <= w <= hi for w in langs[l]["answer_words"])
                               for lab, (lo, hi) in zip(a_labels, a_edges)} for l in order})
    ax.set_xticklabels(a_labels, rotation=0, ha="center")
    ax.set_title("Answer length in words")
    ax.set_xlabel("words in gold answer span")
    ax.set_ylabel("QA pairs")
    save(fig, "answer_length_distribution.png")

    # 7 validation funnel
    stages = ["generated", "passed_validation", "final"]
    stage_labels = ["generated", "passed automatic validation", "final (after dedup)"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    grouped(ax, stages, {l: langs[l]["funnel"] for l in order}, stage_labels)
    ax.set_xticklabels(stage_labels, rotation=0, ha="center")
    ax.set_title("QA candidates through automatic validation")
    ax.set_ylabel("QA pairs")
    save(fig, "validation_results.png", "Human review not yet performed: final counts are pre-review.")

    # 8 source distribution
    types = [label for label, _ in SOURCE_TYPES] + ["NGO / company / individual", "No credit line"]
    present = [t for t in types if any(langs[l]["source_types"].get(t) for l in order)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    grouped(ax, present, {l: langs[l]["source_types"] for l in order})
    ax.set_title("Source documents by original publisher (from article credit lines)")
    ax.set_ylabel("documents")
    save(fig, "source_distribution.png")

    # 9 human review outcomes - only once reviews exist
    if any(langs[l]["review_outcomes"] for l in order):
        outcomes = ["accept", "needs_correction", "reject"]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        grouped(ax, outcomes, {l: langs[l]["review_outcomes"] for l in order},
                ["accepted", "corrected", "rejected"])
        ax.set_xticklabels(["accepted", "corrected", "rejected"], rotation=0, ha="center")
        ax.set_title("Human validation outcomes")
        ax.set_ylabel("QA pairs")
        save(fig, "human_validation_outcomes.png")
    return written


def main() -> None:
    cfg = load_config()
    manifest = load_manifest()
    langs, errors = {}, []
    for lang in LANG_ORDER:
        sources = read_csv(SOURCES_DIR / f"{LANGUAGES[lang]['folder']}_sources.csv")
        stats, errs = language_stats(lang, manifest, sources)
        langs[lang] = stats
        errors.extend(errs)

    leakage = read_json(RESULTS_DIR / "data_leakage_report.json", {}) or {}
    for lang, stats in langs.items():
        prefix = LANGUAGES[lang]["folder"][:2].upper()
        ctx = [c for c in leakage.get("contaminated_contexts", {}) if c.startswith(prefix)]
        docs = [d for d in leakage.get("excluded_documents", []) if d.startswith(prefix)]
        stats["table3"]["data_leakage_count"] = len(ctx) + len(docs) if leakage.get("checks_run") else None

    totals = {k: sum(langs[l]["table1"][k] for l in langs)
              for k in ("documents_collected", "unique_documents", "documents_used", "context_passages",
                        "generated_qa_pairs", "rejected_qa_pairs", "human_reviewed_qa_pairs", "final_qa_pairs")}
    all_final = sum(langs[l]["table1"]["final_qa_pairs"] for l in langs)
    summary = {
        "dataset": "IndicAgriQA-Native",
        "targets_per_language": cfg["targets"],
        "table1_language_statistics": {**{l: s["table1"] for l, s in langs.items()}, "total": totals},
        "table2_domain_distribution": {l: s["table2_domains"] for l, s in langs.items()},
        "table3_quality_assessment": {l: s["table3"] for l, s in langs.items()},
        "question_types": {l: s["question_types"] for l, s in langs.items()},
        "difficulty": {l: s["difficulty"] for l, s in langs.items()},
        "source_publisher_types": {l: s["source_types"] for l, s in langs.items()},
        "funnel": {l: s["funnel"] for l, s in langs.items()},
        "review_agreement": {l: s["review_agreement"] for l, s in langs.items()},
        "irrigation_share_pct": {l: pct(s["table2_domains"]["irrigation"], s["table1"]["final_qa_pairs"])
                                 for l, s in langs.items()},
        "irrigation_share_overall_pct": pct(sum(s["table2_domains"]["irrigation"] for s in langs.values()), all_final),
        "error_analysis": {
            stage: dict(Counter(e["reason"] for e in errors if e["stage"] == stage).most_common())
            for stage in ("automatic_validation", "deduplication", "review_flag")},
        "notes": "Human-review metrics are null until reviewers fill the sheets; no value is estimated.",
    }
    summary["plots"] = plots(langs)
    write_json(RESULTS_DIR / "dataset_statistics.json", summary)
    write_csv(RESULTS_DIR / "error_analysis.csv", errors,
              ["candidate_id", "language", "stage", "reason", "question", "answer", "question_type", "domain"])
    for l, s in langs.items():
        t = s["table1"]
        print(f"[{l}] final {t['final_qa_pairs']} | generated {t['generated_qa_pairs']} | rejected "
              f"{t['rejected_qa_pairs']} | reviewed {t['human_reviewed_qa_pairs']} | span validity "
              f"{s['table3']['answer_span_validity_pct']}% | duplicate rate {s['table3']['duplicate_rate_pct']}%")
    print(f"Total {all_final} QA pairs | {len(errors)} error-analysis rows | plots: {len(summary['plots'])}")


if __name__ == "__main__":
    main()
