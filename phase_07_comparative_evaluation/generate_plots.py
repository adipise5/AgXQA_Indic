"""Phase 7 result plots: translated F1, Native F1 and Native paired differences with intervals.

Inputs are the text-free files written by `statistical_analysis.py report`. Model colors stay fixed across panels.
"""

import argparse
import json
from pathlib import Path

NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi", "bn": "Bengali", "ta": "Tamil", "macro": "Indic macro"}
NATIVE = ("hi", "mr", "bn", "ta")
TRANSLATED = ("en", "hi", "mr", "bn", "ta")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"phase_3a": "#2a78d6", "phase_3b": "#eb6834", "phase_5_alpha_0": "#1baf7a", "phase_5_alpha_1": "#6250d6"}
LABELS = {"phase_3a": "3A English-only", "phase_3b": "3B multilingual (selected)",
          "phase_5_alpha_0": "Phase 5 α 0 (diagnostic control)", "phase_5_alpha_1": "Phase 5 α 1 adapted (diagnostic)"}


def style(plt):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def grouped_bars(plt, groups, series, title, path):
    """groups: [(label, n)], series: [(model, [values])]. Scores on a fixed 0-100 axis."""
    figure, axis = plt.subplots(figsize=(1.6 * len(groups) + 2.5, 4.2))
    width = 0.8 / len(series)
    for index, (model, values) in enumerate(series):
        positions = [g + (index - (len(series) - 1) / 2) * width for g in range(len(groups))]
        bars = axis.bar(positions, values, width=width - 0.04, color=COLORS[model], label=LABELS[model])
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3),
                          textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    axis.set_xticks(range(len(groups)), [f"{name}\n(n={count:,})" for name, count in groups])
    axis.set_ylim(0, 100)
    axis.set_ylabel("F1 (%)")
    axis.grid(axis="x", visible=False)
    axis.set_title(title, loc="left", fontsize=11, color=INK)
    axis.legend(frameon=False, loc="upper right", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def delta_plot(plt, result, title, path):
    rows = {(row["language"], row["metric"]): row for row in result["rows"]}
    names = [*NATIVE, "macro"]
    figure, axis = plt.subplots(figsize=(7.5, 3.6))
    for index, name in enumerate(names):
        row = rows[(name, "f1")]
        y = len(names) - 1 - index
        axis.plot([row["lower_pp"], row["upper_pp"]], [y, y], color=INK_2, linewidth=2, solid_capstyle="round")
        axis.plot(row["delta_pp"], y, "o", markersize=8, color=INK if name == "macro" else INK_2,
                  markeredgecolor=SURFACE, markeredgewidth=2)
        # Interval labels sit in a column right of the plot area so they never collide with the marks.
        axis.annotate(f"{row['delta_pp']:+.2f} [{row['lower_pp']:+.2f}, {row['upper_pp']:+.2f}]",
                      (1.02, y), xycoords=("axes fraction", "data"), va="center", ha="left", fontsize=8, color=INK_2)
    axis.axvline(0, color=INK_2, linewidth=1)
    axis.set_yticks(range(len(names)), [f"{NAMES[n]} (docs={rows[(n, 'f1')]['documents']}, "
                                         f"n={rows[(n, 'f1')]['questions']:,})" for n in reversed(names)])
    low = min(0.0, *(rows[(n, "f1")]["lower_pp"] for n in names))
    high = max(0.0, *(rows[(n, "f1")]["upper_pp"] for n in names))
    pad = 0.08 * (high - low)
    axis.set_xlim(low - pad, high + pad)
    axis.set_xlabel("F1 difference, percentage points (dot = paired estimate, line = 95% interval)")
    axis.grid(axis="y", visible=False)
    axis.set_title(title, loc="left", fontsize=11, color=INK)
    figure.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Directory written by statistical_analysis.py report")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    read = lambda name: json.loads((args.results / name).read_text(encoding="utf-8"))
    translated, native = read("translated.json"), read("native_scores.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = [(NAMES[l], translated["question_groups"]) for l in TRANSLATED]
    grouped_bars(plt, groups, [(m, [translated["scores"][m][l]["f1"] for l in TRANSLATED]) for m in ("phase_3a", "phase_3b")],
                 "Translated AgXQA-Indic F1 (provisional, seed 42)", args.output_dir / "translated_f1.png")
    counts = native["phase_3b"]["by_language"]
    groups = [(NAMES[l], counts[l]["count"]) for l in NATIVE]
    grouped_bars(plt, groups, [(m, [native[m]["by_language"][l]["f1"] for l in NATIVE]) for m in ("phase_3a", "phase_3b")],
                 "Native IndicAgriQA F1 (provisional, not human-reviewed)", args.output_dir / "native_f1.png")
    delta_plot(plt, read("native_phase_3b_vs_phase_3a.json"), "Native: 3B − 3A, paired by question",
               args.output_dir / "native_delta_3b_vs_3a.png")
    if (args.results / "native_phase_5_alpha_1_vs_alpha_0.json").exists():
        delta_plot(plt, read("native_phase_5_alpha_1_vs_alpha_0.json"),
                   "Native diagnostic: Phase 5 α 1 − α 0, paired by question",
                   args.output_dir / "native_delta_alpha1_vs_alpha0.png")
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
