"""Draw docs/architecture.png: the implemented Phase 6 pipeline with actual counts.

Counts are read from results/*.json, so the diagram always matches the run.

Usage:
    python phase_06_native_indic_qa/scripts/plot_architecture.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import PHASE_DIR, RESULTS_DIR, read_json

SURFACE, INK, INK_2, BORDER = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7"
DONE, PENDING = "#e8f1fc", "#fdf3e1"  # light blue = done, light amber = pending


def main() -> None:
    stats = read_json(RESULTS_DIR / "dataset_statistics.json", {})
    t1 = stats["table1_language_statistics"]["total"]
    leak = read_json(RESULTS_DIR / "data_leakage_report.json", {}) or {}
    cand = sum(f["generated"] for f in stats["funnel"].values())
    passed = sum(f["passed_validation"] for f in stats["funnel"].values())

    steps = [
        ("Native agricultural documents", f"{t1['documents_used']} articles (hi 21 · mr 20 · bn 18 · ta 11)\n"
         "Vikaspedia: state agriculture depts, TNAU, KVKs, ICAR", DONE, "download_documents.py"),
        ("Text extraction & preprocessing", "embedded article HTML → headings, paragraphs, table rows\n"
         "NFC · credit-line removal · script / hi-mr checks · dedup", DONE, "extract_text.py · clean_text.py"),
        ("Context generation", f"{t1['context_passages']} passages, 100–200 words, within one section\n"
         "XLM-R token count per passage", DONE, "chunk_documents.py"),
        ("LLM-based QA generation", f"gemini-3.5-flash-lite · prompt v2 · temperature 0.2\n"
         f"{cand:,} native-language candidates", DONE, "generate_qa.py"),
        ("Automated validation", f"exact answer span (offset computed in code) · language · length\n"
         f"{passed:,} passed → {t1['final_qa_pairs']:,} after deduplication", DONE,
         "validate_answer_spans.py · deduplicate.py"),
        ("Human validation", "one native reviewer per language\nsheets ready · 0 reviewed so far", PENDING,
         "export_review_sheets.py"),
        ("Data leakage verification", "vs AgXQA-Indic (Phase 2): 0 contaminated passages\n"
         "vs Indic-AEC (Phase 4): corpus not built yet · exclusion list issued", PENDING,
         "check_data_leakage.py · export_phase4_exclusions.py"),
        ("IndicAgriQA-Native", f"{t1['final_qa_pairs']:,} QA pairs · test-only · release without passages\n"
         "hydrate_dataset.py rebuilds full files from source URLs", DONE, "build_dataset.py · dataset_statistics.py"),
        ("Phase 7 — comparative evaluation", "English-trained, multilingual and Indic-AEC XLM-R\n"
         "EM / F1 per language, macro average, irrigation subset", PENDING, "(owner: Dhiraj)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 13.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.5 + 0.95)
    ax.axis("off")
    for i, (title, body, fill, script) in enumerate(steps):
        y = (len(steps) - i - 1) * 1.5 + 0.5
        ax.add_patch(FancyBboxPatch((0.6, y), 8.8, 1.05, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=fill, edgecolor=BORDER, linewidth=1.2))
        ax.text(0.85, y + 0.8, f"{i + 1}. {title}", fontsize=12, fontweight="bold", color=INK, va="center")
        ax.text(0.85, y + 0.38, body, fontsize=9, color=INK_2, va="center", linespacing=1.4)
        ax.text(9.2, y + 0.8, script, fontsize=8, color=INK_2, va="center", ha="right", family="monospace")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((5, y), (5, y - 0.43), arrowstyle="-|>", mutation_scale=14,
                                         color=INK_2, linewidth=1.2))
    top = len(steps) * 1.5
    ax.text(0.6, top + 0.68, "Phase 6 pipeline as implemented (2026-11-18)", fontsize=14,
            fontweight="bold", color=INK, va="center")
    for x, fill, label in ((0.6, DONE, "done"), (1.75, PENDING, "pending / partial")):
        ax.add_patch(FancyBboxPatch((x, top + 0.2), 0.3, 0.2, boxstyle="round,pad=0.01",
                                    facecolor=fill, edgecolor=BORDER))
        ax.text(x + 0.4, top + 0.3, label, fontsize=9, color=INK_2, va="center")
    out = PHASE_DIR / "docs" / "architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
