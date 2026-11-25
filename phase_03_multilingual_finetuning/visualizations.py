"""Write figures for the metrics and logs supplied by completed runs."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_plots(directory, results, gaps, joint, history, categories):
    directory.mkdir(parents=True, exist_ok=False)
    written = []

    def save(fig, ax, name, title, ylabel):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(directory / name, dpi=160)
        plt.close(fig)
        written.append(name)

    def comparison(name, title, ylabel, x, a, b, cap=None):
        fig, ax = plt.subplots(figsize=(7.2, 4.1))
        ax.bar([i - .19 for i in range(len(x))], a, width=.38, label="3A English only", color="#416e9a")
        ax.bar([i + .19 for i in range(len(x))], b, width=.38, label="3B Multilingual", color="#19826c")
        ax.set_xticks(range(len(x)), x)
        if cap:
            ax.set_ylim(0, cap)
        ax.legend()
        save(fig, ax, name, title, ylabel)

    by_language = results[:5]
    codes = [row["language"].upper() for row in by_language]
    for metric, label in (("em", "Exact match"), ("f1", "Token F1")):
        comparison(f"{metric}_comparison.png", label + " by language", label + " (%)", codes,
                   [row[metric + "_3a"] for row in by_language], [row[metric + "_3b"] for row in by_language], 100)
    comparison("language_gap.png", "English minus Indic F1", "Gap (percentage points)",
               [row["language"].upper() for row in gaps], [row["gap_3a_pp"] for row in gaps],
               [row["gap_3b_pp"] for row in gaps])
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.bar(codes, [row["delta_f1_pp"] for row in by_language], color="#19826c")
    ax.axhline(0, color="#47606c", linewidth=0.8)
    save(fig, ax, "multilingual_improvement.png", "Change in F1 after multilingual fine-tuning",
         "3B minus 3A (percentage points)")
    comparison("joint_correctness.png", "Number of languages correct for the same question", "Question groups (%)",
               [f"{i}/5" for i in range(6)],
               [row["percent"] for row in joint["phase_3a"]["correct_in_n_languages"]],
               [row["percent"] for row in joint["phase_3b"]["correct_in_n_languages"]], 100)
    comparison("macro_f1.png", "Macro-averaged F1", "F1 (%)", ["Indic", "All languages"],
               [row["f1_3a"] for row in results[5:]], [row["f1_3b"] for row in results[5:]], 100)

    for key, title, filename in (("loss", "Training loss", "training_loss.png"),
                                 ("eval_loss", "Validation loss", "validation_loss.png")):
        events = [row for row in history if key in row]
        if events:
            fig, ax = plt.subplots(figsize=(7.2, 4.1))
            ax.plot([row["epoch"] for row in events], [row[key] for row in events], marker=".", color="#416e9a")
            ax.set_xlabel("Epoch")
            save(fig, ax, filename, title, "Span loss")

    validation = [row for row in history if "eval_macro_f1" in row]
    if validation:
        fig, ax = plt.subplots(figsize=(7.2, 4.1))
        for code in codes:
            ax.plot([row["epoch"] for row in validation], [row[f"eval_{code.lower()}_f1"] for row in validation],
                    marker="o", label=code)
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 100)
        save(fig, ax, "validation_f1.png", "Validation F1 by language", "F1 (%)")
    if categories:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        names = sorted(categories, key=categories.get)
        ax.barh(names, [categories[name] for name in names], color="#416e9a")
        ax.set_xlabel("Human-coded cases")
        save(fig, ax, "error_distribution.png", "Human-coded error categories", "")
    return written
