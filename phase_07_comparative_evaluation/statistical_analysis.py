"""Phase 7 paired statistics, error sample and result tables.

Subcommands:
  native      Paired Native comparison. Source documents are resampled within each language, with the same draws
              for both models.
  translated  Reuse the pinned Phase 3 translated predictions (hash-verified) and the Phase 5 diagnostic.
  errors      Reproducible paired error sample per Native language. The CSV holds passage text, so keep it private.
  report      Text-free tables and manifest from a completed run directory.
"""

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from statistics import mean
import unicodedata

from phase_03_multilingual_finetuning.common import LANGUAGES as TRANSLATED_LANGUAGES, read_json, read_settings, write_json
from phase_03_multilingual_finetuning.scoring import aggregate, bootstrap, compare_inputs, consistency, percentile
from shared.evaluate import evaluate
from shared.metrics import best_scores
from shared.qa_release import sha256
from shared.validate import read_jsonl

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parent
PHASE3 = ROOT / "phase_03_multilingual_finetuning" / "results"
NATIVE_LANGUAGES = ("hi", "mr", "bn", "ta")
NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi", "bn": "Bengali", "ta": "Tamil"}
METRICS = ("em", "f1")
ERROR_FIELDS = ("id", "language", "case", "source_document_id", "question", "gold_answers", "control_prediction",
                "candidate_prediction", "control_f1", "candidate_f1", "gold_has_numeral", "partial_overlap",
                "prediction_longer_than_gold", "prediction_inside_gold", "gold_exceeds_decode_limit",
                "multi_window_passage", "context", "error_category", "reviewer", "comments")


def bootstrap_settings():
    settings = read_settings("evaluation_config.yaml")["bootstrap"]
    return settings["iterations"], settings["confidence"], settings["seed"]


# ---------------------------------------------------------------- Native paired bootstrap

def native_paired(gold, control, candidate, iterations, confidence, seed):
    """Candidate − control deltas per language and for the equal-language macro, in percentage points."""
    a, b = evaluate(gold, control), evaluate(gold, candidate)
    documents = {(row["id"], row["language"]): row["source_document_id"] for row in gold}
    first = {(row["id"], row["language"]): row for row in a["per_example"]}
    second = {(row["id"], row["language"]): row for row in b["per_example"]}
    if set(first) != set(second) or set(first) != set(documents):
        raise ValueError("Both prediction sets must cover the same (id, language) rows as the gold file")
    languages = [language for language in NATIVE_LANGUAGES if any(key[1] == language for key in first)]
    clusters = {}
    for language in languages:
        sums = defaultdict(lambda: [0, 0.0, 0.0])
        for key in sorted(first):
            if key[1] == language:
                cluster = sums[documents[key]]
                cluster[0] += 1
                for index, metric in enumerate(METRICS, 1):
                    cluster[index] += second[key][metric] - first[key][metric]
        clusters[language] = [tuple(values) for _, values in sorted(sums.items())]

    def deltas(chosen):
        result = {}
        for language in languages:
            count = sum(cluster[0] for cluster in chosen[language])
            result[language] = [100 * sum(cluster[i] for cluster in chosen[language]) / count for i in (1, 2)]
        result["macro"] = [mean(result[language][i] for language in languages) for i in (0, 1)]
        return result

    point = deltas(clusters)
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(iterations):
        drawn = deltas({language: rng.choices(clusters[language], k=len(clusters[language])) for language in languages})
        for name, values in drawn.items():
            for metric, value in zip(METRICS, values):
                samples[(name, metric)].append(value)
    tail = (1 - confidence) / 2
    rows = []
    for name in (*languages, "macro"):
        for index, metric in enumerate(METRICS):
            ordered = sorted(samples[(name, metric)])
            scope = languages if name == "macro" else [name]
            control_score = (mean(a["by_language"][lang][metric] for lang in scope))
            candidate_score = (mean(b["by_language"][lang][metric] for lang in scope))
            rows.append({"language": name, "metric": metric, "control": control_score, "candidate": candidate_score,
                         "delta_pp": point[name][index], "lower_pp": percentile(ordered, tail),
                         "upper_pp": percentile(ordered, 1 - tail),
                         "documents": sum(len(clusters[lang]) for lang in scope),
                         "questions": sum(cluster[0] for lang in scope for cluster in clusters[lang])})
    return rows


def cmd_native(args):
    iterations, confidence, seed = bootstrap_settings()
    iterations, seed = args.iterations or iterations, args.seed if args.seed is not None else seed
    gold = read_jsonl(args.gold)
    rows = native_paired(gold, read_jsonl(args.control), read_jsonl(args.candidate), iterations, confidence, seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {
        "comparison": f"{args.candidate_label} minus {args.control_label}", "control": args.control_label,
        "candidate": args.candidate_label, "scorer": "shared.evaluate (agxqa-indic-v1), scores 0-100",
        "macro": "equal-weight mean of the Native language scores",
        "resampling": "source documents with replacement within each language; identical draws for both models",
        "iterations": iterations, "confidence": confidence, "seed": seed,
        "input_sha256": {"gold": sha256(args.gold), "control": sha256(args.control), "candidate": sha256(args.candidate)},
        "rows": rows,
    })
    with args.output.with_suffix(".csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


# ---------------------------------------------------------------- translated (Phase 3 reuse)

def pinned_predictions(experiment):
    directory = PHASE3 / experiment
    manifest = read_json(directory / "run_manifest.json")
    digest = sha256(directory / "predictions.jsonl")
    if digest != manifest["output_file_sha256"]["predictions.jsonl"]:
        raise ValueError(f"{experiment} predictions differ from their run manifest")
    return read_jsonl(directory / "predictions.jsonl"), digest


def cmd_translated(args):
    iterations, confidence, seed = bootstrap_settings()
    (a, a_hash), (b, b_hash) = pinned_predictions("phase_3a"), pinned_predictions("phase_3b")
    groups = compare_inputs(a, b)
    scores = {"phase_3a": aggregate(a), "phase_3b": aggregate(b)}
    joint = {"phase_3a": consistency(a), "phase_3b": consistency(b)}
    gap = {name: {lang: score["en"]["f1"] - score[lang]["f1"] for lang in TRANSLATED_LANGUAGES[1:]}
           for name, score in scores.items()}
    phase5 = read_json(ROOT / "phase_05_domain_adaptation" / "results" / "alpha_sweep_summary.json")
    write_json(args.output, {
        "dataset": read_json(PHASE3 / "phase_3b" / "run_manifest.json")["dataset"],
        "scorer": "agxqa-indic-v1 (Phase 3 scoring)", "question_groups": len(groups[0]),
        "source_documents": len({group["en"]["source_document_id"] for group in groups[0].values()}),
        "predictions_sha256": {"phase_3a": a_hash, "phase_3b": b_hash},
        "scores": scores, "english_minus_language_f1_gap": gap,
        "all_five_languages_exact": {name: next(row for row in result["correct_in_n_languages"]
                                                if row["correct_languages"] == 5) for name, result in joint.items()},
        "phase_3b_minus_phase_3a_document_intervals": bootstrap(*groups, iterations, confidence, seed, "document"),
        "phase_5_diagnostic": {
            "label": "Phase 5 alpha 1 minus its alpha-0 rerun (translated test, diagnostic only; the sweep selected alpha 0)",
            "alpha_0": phase5["test"]["alpha_0_control"]["scores"],
            "alpha_1": phase5["test"]["alpha_1_context_seen_before_sweep"]["scores"],
            "intervals": phase5["paired_test_f1_and_em_delta_pp_source_document_95ci"]["alpha_1_minus_alpha_0_control_context_only"],
            "source": "phase_05_domain_adaptation/results/alpha_sweep_summary.json",
            "source_sha256": sha256(ROOT / "phase_05_domain_adaptation" / "results" / "alpha_sweep_summary.json")},
    })
    print(f"Wrote {args.output}")


# ---------------------------------------------------------------- error sample

def has_numeral(text):
    return any(unicodedata.category(character) == "Nd" for character in text)


def run_records(directory):
    predictions = {row["id"]: row for row in read_jsonl(directory / "predictions.jsonl")}
    details = {row["id"]: row for row in read_jsonl(directory / "predictions_detail.jsonl")}
    if set(predictions) != set(details):
        raise ValueError(f"{directory}: predictions and diagnostics cover different IDs")
    return predictions, details


def error_sample(gold, control_run, candidate_run, per_case, seed):
    (a, a_detail), (b, b_detail) = run_records(control_run), run_records(candidate_run)
    if set(a) != set(b) or set(a) != {row["id"] for row in gold}:
        raise ValueError("Both runs must cover every gold ID")
    counts, sample = {}, []
    for language in NATIVE_LANGUAGES:
        cases = defaultdict(list)
        for row in sorted((row for row in gold if row["language"] == language), key=lambda row: row["id"]):
            references = [answer["text"] for answer in row["answers"]]
            first, second = (best_scores(run[row["id"]]["prediction_text"], references) for run in (a, b))
            case = {(1, 1): "both_correct", (0, 0): "both_wrong", (0, 1): "candidate_recovered",
                    (1, 0): "candidate_newly_missed"}[(int(first["em"]), int(second["em"]))]
            cases[case].append((row, references, first, second))
        counts[language] = {"questions": sum(len(items) for items in cases.values()),
                            **{case: len(cases[case]) for case in ("both_correct", "both_wrong", "candidate_recovered",
                                                                   "candidate_newly_missed")}, "sampled": {}, "flags": {}}
        for case in ("candidate_recovered", "candidate_newly_missed", "both_wrong"):
            chosen = random.Random(f"{seed}:{language}:{case}").sample(cases[case], min(per_case, len(cases[case])))
            counts[language]["sampled"][case] = len(chosen)
            flags = Counter()
            for row, references, first, second in sorted(chosen, key=lambda item: item[0]["id"]):
                missed = a if case == "candidate_recovered" else b
                missed_score = first if case == "candidate_recovered" else second
                prediction = missed[row["id"]]["prediction_text"]
                record = {
                    "id": row["id"], "language": language, "case": case, "source_document_id": row["source_document_id"],
                    "question": row["question"], "gold_answers": " | ".join(references),
                    "control_prediction": a[row["id"]]["prediction_text"],
                    "candidate_prediction": b[row["id"]]["prediction_text"],
                    "control_f1": round(first["f1"], 4), "candidate_f1": round(second["f1"], 4),
                    "gold_has_numeral": any(has_numeral(text) for text in references),
                    "partial_overlap": 0 < missed_score["f1"] < 1,
                    "prediction_longer_than_gold": any(text in prediction and text != prediction for text in references),
                    "prediction_inside_gold": any(prediction in text and text != prediction for text in references),
                    "gold_exceeds_decode_limit": a_detail[row["id"]]["gold_exceeds_decode_limit"],
                    "multi_window_passage": a_detail[row["id"]]["windows"] > 1,
                    "context": row["context"], "error_category": "", "reviewer": "", "comments": "",
                }
                flags.update(flag for flag in ERROR_FIELDS[10:16] if record[flag] is True)
                sample.append(record)
            counts[language]["flags"][case] = {flag: flags[flag] for flag in ERROR_FIELDS[10:16]}
    return counts, sample


def cmd_errors(args):
    gold = read_jsonl(args.gold)
    counts, sample = error_sample(gold, args.control_run, args.candidate_run, args.per_case, args.seed)
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    with args.private_output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ERROR_FIELDS)
        writer.writeheader()
        writer.writerows(sample)
    write_json(args.counts_output, {
        "control": args.control_run.name, "candidate": args.candidate_run.name, "per_case_sample": args.per_case,
        "seed": args.seed, "definition": "Cases use exact match. Automatic flags describe the model that missed; "
        "alternative phrasing and unit errors need human review in the private CSV.", "by_language": counts})
    print(f"Wrote {args.private_output} (private) and {args.counts_output}")


# ---------------------------------------------------------------- report (text-free)

MODELS = ("phase_3a", "phase_3b", "phase_5_alpha_0", "phase_5_alpha_1")
LABELS = {"phase_3a": "XLM-R English-only (3A, checkpoint-500, seed 42)",
          "phase_3b": "XLM-R multilingual (3B, checkpoint-1792, seed 42; selected)",
          "phase_5_alpha_0": "Phase 5 α 0 rerun of the 3B recipe (checkpoint-896, seed 42; diagnostic control)",
          "phase_5_alpha_1": "Phase 5 α 1 Indic-AEC adapted (checkpoint-896, seed 42; diagnostic)"}
PAIRS = {"native_phase_3b_vs_phase_3a.json": ("phase_3a", "phase_3b", "3B − 3A (primary)"),
         "native_phase_5_alpha_1_vs_alpha_0.json": ("phase_5_alpha_0", "phase_5_alpha_1", "α 1 − α 0 (diagnostic)"),
         "native_phase_5_alpha_0_vs_phase_3b.json": ("phase_3b", "phase_5_alpha_0", "α 0 rerun − recorded 3B (rerun variation)")}
PLOTS = (("translated_f1.png", "Translated F1, 3A vs 3B"), ("native_f1.png", "Native F1, 3A vs 3B"),
         ("native_delta_3b_vs_3a.png", "Native 3B − 3A paired differences"),
         ("native_delta_alpha1_vs_alpha0.png", "Native diagnostic α 1 − α 0 paired differences"))
CODE = ("phase_07_comparative_evaluation/evaluate_frozen.py", "phase_07_comparative_evaluation/statistical_analysis.py",
        "phase_07_comparative_evaluation/generate_plots.py", "phase_07_comparative_evaluation/hpc/run_phase7.sh",
        "phase_07_comparative_evaluation/configs/checkpoints.json")
FLAGS = ERROR_FIELDS[10:16]
CASES = ("candidate_recovered", "candidate_newly_missed", "both_wrong")
CASE_NAMES = {"candidate_recovered": "3B recovered", "candidate_newly_missed": "3B newly missed", "both_wrong": "Both wrong"}


def fmt(value):
    return f"{value:.2f}"


def interval(row):
    return f"{row['delta_pp']:+.2f} [{row['lower_pp']:+.2f}, {row['upper_pp']:+.2f}]"


def native_scores(run, model):
    metrics = read_json(run / model / "metrics.json")
    manifest = read_json(run / model / "run_manifest.json")
    if manifest["smoke"] or metrics["count"] != manifest["dataset"]["records"]:
        raise ValueError(f"{model}: metrics do not cover the full Native file")
    if {lang: metrics["by_language"][lang]["count"] for lang in NATIVE_LANGUAGES} != manifest["dataset"]["languages"]:
        raise ValueError(f"{model}: language denominators differ from the frozen file")
    if (sha256(run / model / "predictions.jsonl") != manifest["predictions_sha256"]
            or metrics["input_sha256"] != {"gold": manifest["dataset"]["sha256"], "predictions": manifest["predictions_sha256"]}):
        raise ValueError(f"{model}: metrics were not computed from the verified files")
    return metrics, manifest


def git(*arguments):
    import subprocess

    try:
        return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def revisions(run_source):
    """Run revision (code used for inference) and report-generation revision, with per-file git blob IDs."""
    report = {"base_commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
              "uncommitted_changes": bool(git("status", "--porcelain")),
              "file_git_blob": {path: git("hash-object", path) for path in CODE}}
    tree = (run_source or {}).get("tree")
    run_blobs = {path: git("rev-parse", f"{tree}:{path}") for path in CODE} if tree else {}
    report["changed_since_run"] = sorted(path for path in CODE if run_blobs.get(path) is not None
                                         and run_blobs[path] != report["file_git_blob"][path])
    return {"run": {**(run_source or {}), "file_git_blob": run_blobs}, "report_generation": report}


def error_review(run):
    """Counts only: how many sampled error cases carry a reviewer decision in the private CSV."""
    path = run / "analysis" / "native_error_sample.private.csv"
    if not path.exists():
        return {"sampled_rows": None, "reviewed_rows": None, "status": "private sample unavailable"}
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    reviewed = sum(1 for row in rows if row["error_category"].strip() and row["reviewer"].strip())
    return {"sampled_rows": len(rows), "reviewed_rows": reviewed,
            "status": "automated flags only; not reviewer-confirmed" if reviewed < len(rows) else "reviewed"}


def report_markdown(phase1, translated, dataset, native, pairs, errors, review):
    human = dataset["human_reviewed_records"]
    t_scores, t_int = translated["scores"], {(r["language"], r["metric"]): r
                                             for r in translated["phase_3b_minus_phase_3a_document_intervals"]}
    groups = translated["question_groups"]
    documents = {lang: pairs["native_phase_3b_vs_phase_3a.json"][(lang, "f1")]["documents"] for lang in (*NATIVE_LANGUAGES, "macro")}
    lines = [
        "# Phase 7 comparative results", "",
        "Scores are percentages. All models are single runs with seed 42. **Phase 3B is the selected model.** "
        "Phase 5's sweep found no qualifying adaptation.", "",
        f"- **Native review coverage:** {human} of {dataset['records']:,} questions human-reviewed. Native results are provisional.",
        f"- **Error-sample review:** {review['reviewed_rows']} of {review['sampled_rows']} sampled cases have a reviewer decision. "
        "The error categories below are automated and provisional.",
        "- **Translated release:** the Gemini v2 `span_checked` release is unreviewed (provisional).", "",
        "## 1. Original English AgXQA", "",
        f"{phase1['test_count']} official test items, scored with `{phase1['evaluation_protocol']}` "
        "(SQuAD 1.1 normalization, EM and token-overlap F1).", "",
        "| System | EM | F1 |", "|---|---|---|",
        f"| Original AgRoBERTa (published) | {fmt(phase1['published_reference']['exact_match'])} | {fmt(phase1['published_reference']['f1'])} |",
        f"| Phase 1 reimplementation | {fmt(phase1['agroberta_reimplementation']['exact_match'])} | {fmt(phase1['agroberta_reimplementation']['f1'])} |", "",
        "## 2. Translated AgXQA-Indic", "",
        f"{groups} aligned question groups in each of 5 languages ({groups * 5:,} rows), from {translated['source_documents']} source documents. "
        "Scored with `agxqa-indic-v1` (EM and token-overlap F1).", "",
        "| System | " + " | ".join(f"{NAMES[l]} EM / F1 (n={groups})" for l in TRANSLATED_LANGUAGES) + " | Indic macro EM / F1 |",
        "|---|" + "---|" * (len(TRANSLATED_LANGUAGES) + 1)]
    for name in ("phase_3a", "phase_3b"):
        score = t_scores[name]
        lines.append(f"| {LABELS[name]} | " + " | ".join(f"{fmt(score[l]['exact_match'])} / {fmt(score[l]['f1'])}" for l in TRANSLATED_LANGUAGES)
                     + f" | {fmt(score['indic_macro']['exact_match'])} / {fmt(score['indic_macro']['f1'])} |")
    lines += ["", f"Paired 3B − 3A (95% intervals, {translated['source_documents']} source documents resampled, 10,000 draws, seed 42):", "",
              "| Metric | " + " | ".join(NAMES[l] for l in TRANSLATED_LANGUAGES) + " | Indic macro |",
              "|---|" + "---|" * (len(TRANSLATED_LANGUAGES) + 1)]
    for metric, key in (("F1", "f1"), ("EM", "exact_match")):
        lines.append(f"| {metric} | " + " | ".join(interval(t_int[(l, key)]) for l in (*TRANSLATED_LANGUAGES, "indic_macro")) + " |")
    gap, joint = translated["english_minus_language_f1_gap"], translated["all_five_languages_exact"]
    lines += ["", f"English − language F1 gap on the same {groups} aligned groups, and joint correctness "
              f"(a group counts if all five languages are exact):", "",
              "| System | " + " | ".join(f"EN − {NAMES[l]}" for l in TRANSLATED_LANGUAGES[1:]) + " | All five exact |",
              "|---|" + "---|" * len(TRANSLATED_LANGUAGES)]
    for name in ("phase_3a", "phase_3b"):
        lines.append(f"| {LABELS[name]} | " + " | ".join(fmt(gap[name][l]) for l in TRANSLATED_LANGUAGES[1:])
                     + f" | {joint[name]['count']} of {groups} ({fmt(joint[name]['percent'])}%) |")
    diagnostic = translated["phase_5_diagnostic"]["intervals"]
    lines += ["", "Translated diagnostic, Phase 5 α 1 − α 0 (both checkpoint-896, seed 42; the α 1 test scores were seen "
              "before the Phase 5 sweep): F1 " + "; ".join(
                  f"{NAMES.get(k.split('/')[0], 'Indic macro')} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"
                  for k, v in diagnostic.items() if k.endswith("/f1") and k.split("/")[0] in ("en", "indic_macro")) + ".", "",
              "## 3. Native IndicAgriQA (provisional)", "",
              f"{dataset['records']:,} questions from {dataset['source_documents']} source documents and {dataset['passages']} passages, "
              "with identical frozen rows for every model. Scored with `agxqa-indic-v1`. "
              "Indic macro is the equal-weight mean of the four languages.", "",
              "| System | " + " | ".join(f"{NAMES[l]} EM / F1 (n={dataset['languages'][l]:,}, docs={documents[l]})" for l in NATIVE_LANGUAGES)
              + " | Indic macro EM / F1 |", "|---|" + "---|" * (len(NATIVE_LANGUAGES) + 1)]
    for model, (metrics, _) in native.items():
        by = metrics["by_language"]
        lines.append(f"| {LABELS[model]} | " + " | ".join(f"{fmt(by[l]['em'])} / {fmt(by[l]['f1'])}" for l in NATIVE_LANGUAGES)
                     + f" | {fmt(mean(by[l]['em'] for l in NATIVE_LANGUAGES))} / {fmt(mean(by[l]['f1'] for l in NATIVE_LANGUAGES))} |")
    lines += ["", "## 4. Native paired differences", "",
              "Pairs are matched by `(id, language)`. Source documents are resampled within each language with the same draws "
              "for both models (10,000 draws, 95% percentile interval, seed 42). The intervals describe uncertainty over source "
              "documents for these fixed checkpoints, not variation between training runs. The diagnostic rows are secondary.", "",
              "| Comparison | Metric | " + " | ".join(NAMES[l] for l in NATIVE_LANGUAGES) + " | Indic macro |",
              "|---|---|" + "---|" * (len(NATIVE_LANGUAGES) + 1)]
    for name, rows in pairs.items():
        for metric in ("f1", "em"):
            lines.append(f"| {PAIRS[name][2]} | {metric.upper()} | "
                         + " | ".join(interval(rows[(l, metric)]) for l in (*NATIVE_LANGUAGES, "macro")) + " |")
    lines += ["", "## 5. Figures", ""] + [f"- [{caption}](plots/{file})" for file, caption in PLOTS] + [
        "", "## 6. Native error summary (3A vs 3B, exact match)", "", "Paired outcomes over all questions:", "",
        "| Language | Questions | Both correct | Both wrong | 3B recovered | 3B newly missed |", "|---|---|---|---|---|---|"]
    for language, item in errors["by_language"].items():
        lines.append(f"| {NAMES[language]} | {item['questions']} | {item['both_correct']} | {item['both_wrong']} | "
                     f"{item['candidate_recovered']} | {item['candidate_newly_missed']} |")
    lines += ["", f"Sampling coverage: up to {errors['per_case_sample']} cases per outcome and language, drawn with seed "
              f"{errors['seed']}. Automated flags (**provisional**) describe the model that missed; for both-wrong cases, 3B. "
              f"Review status: {review['reviewed_rows']} of {review['sampled_rows']} sampled cases reviewed.", "",
              "| Language | Outcome | Sampled / available | " + " | ".join(f.replace("_", " ") for f in FLAGS) + " |",
              "|---|---|---|" + "---|" * len(FLAGS)]
    for language, item in errors["by_language"].items():
        for case in CASES:
            flags = item["flags"][case]
            lines.append(f"| {NAMES[language]} | {CASE_NAMES[case]} | {item['sampled'][case]} / {item[case]} | "
                         + " | ".join(str(flags[f]) for f in FLAGS) + " |")
    lines += ["", "Passage text and predictions stay in the private run folder."]
    return "\n".join(lines) + "\n"


def cmd_report(args):
    run, out = args.run, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    phase1 = read_json(ROOT / "phase_01_original_baseline" / "results" / "baseline_results.json")
    translated = read_json(run / "analysis" / "translated.json")
    native = {model: native_scores(run, model) for model in MODELS if (run / model / "metrics.json").exists()}
    dataset = native["phase_3b"][1]["dataset"]
    if any(manifest["dataset"] != dataset for _, manifest in native.values()):
        raise ValueError("The Native runs did not use the same verified dataset")
    pairs = {}
    for name, (control, candidate, _) in PAIRS.items():
        path = run / "analysis" / name
        if not path.exists():
            continue
        result = read_json(path)
        if result["input_sha256"] != {"gold": dataset["sha256"], "control": native[control][1]["predictions_sha256"],
                                      "candidate": native[candidate][1]["predictions_sha256"]}:
            raise ValueError(f"{name} was not computed from the reported predictions")
        pairs[name] = {(row["language"], row["metric"]): row for row in result["rows"]}
    errors = read_json(run / "analysis" / "native_error_counts.json")
    review = error_review(run)
    (out / "comparison_tables.md").write_text(report_markdown(phase1, translated, dataset, native, pairs, errors, review),
                                              encoding="utf-8")
    for name in (*PAIRS, "translated.json"):
        if (run / "analysis" / name).exists():
            (out / name).write_bytes((run / "analysis" / name).read_bytes())
    (out / "error_counts.json").write_bytes((run / "analysis" / "native_error_counts.json").read_bytes())
    write_json(out / "native_scores.json", {model: {
        "label": LABELS[model], "role": manifest["role"],
        "by_language": {l: {k: metrics["by_language"][l][k] for k in ("count", "em", "f1")} for l in NATIVE_LANGUAGES},
        "indic_macro": {m: mean(metrics["by_language"][l][m] for l in NATIVE_LANGUAGES) for m in METRICS}}
        for model, (metrics, manifest) in native.items()})

    roots = {arg.split("/outputs/")[0] for _, manifest in native.values() for arg in manifest["command"]
             if isinstance(arg, str) and "/outputs/" in arg and arg.startswith("/")}

    def relative(value):
        """Strip checkout roots from recorded paths so the public manifest holds no host paths."""
        if isinstance(value, str):
            for root in {str(ROOT), ROOT.as_posix(), *roots}:
                value = value.replace(root.rstrip("/") + "/", "")
            return value
        if isinstance(value, list):
            return [relative(item) for item in value]
        if isinstance(value, dict):
            return {key: relative(item) for key, item in value.items()}
        return value

    iterations, confidence, seed = bootstrap_settings()
    run_rel = relative(run.resolve().as_posix())
    step1 = "outputs/phase_07_comparative_evaluation/step1"
    gold = f"{step1}/phase6_test.shared.jsonl"
    module = "python -m phase_07_comparative_evaluation"
    commands = [f"{module}.evaluate_frozen --manifest {step1}/manifest.json --model {model} --output {run_rel}/{model} "
                f"--device cuda --batch-size {native[model][1]['batch_size']}" for model in native]
    commands += [f"python -m shared.evaluate {gold} {run_rel}/{model}/predictions.jsonl --output {run_rel}/{model}/metrics.json"
                 for model in native]
    commands += [f"{module}.statistical_analysis native --gold {gold} --control {run_rel}/{control}/predictions.jsonl "
                 f"--candidate {run_rel}/{candidate}/predictions.jsonl --control-label {control} --candidate-label {candidate} "
                 f"--output {run_rel}/analysis/{name}" for name, (control, candidate, _) in PAIRS.items() if name in pairs]
    commands += [f"{module}.statistical_analysis translated --output {run_rel}/analysis/translated.json",
                 f"{module}.statistical_analysis errors --gold {gold} --control-run {run_rel}/phase_3a --candidate-run "
                 f"{run_rel}/phase_3b --private-output {run_rel}/analysis/native_error_sample.private.csv "
                 f"--counts-output {run_rel}/analysis/native_error_counts.json",
                 f"{module}.statistical_analysis report --run {run_rel} --output-dir {relative(out.resolve().as_posix())}",
                 f"{module}.generate_plots --results {relative(out.resolve().as_posix())} "
                 f"--output-dir {relative((out / 'plots').resolve().as_posix())}"]
    write_json(out / "phase7_manifest.json", {
        "selected_model": "phase_3b", "run_id": run.name,
        "revisions": revisions(native["phase_3b"][1]["source"]),
        "native_dataset": dataset, "decoding": native["phase_3b"][1]["decoding"],
        "scorers": {"english": phase1["evaluation_protocol"], "translated_and_native": "agxqa-indic-v1 (EM and token-overlap F1, 0-100)"},
        "bootstrap": {"iterations": iterations, "confidence": confidence, "seed": seed,
                      "native_unit": "source documents within each language, identical draws for both models",
                      "translated_unit": "source documents"},
        "review_coverage": {"native_questions_human_reviewed": dataset["human_reviewed_records"],
                            "native_questions": dataset["records"], "error_sample": review},
        "models": {model: relative({key: manifest[key] for key in ("role", "label", "checkpoint", "batch_size", "device",
                                                                  "source", "command", "environment", "prediction_count",
                                                                  "evaluated_languages", "predictions_sha256", "long_gold_spans")})
                   | {"metrics_sha256": sha256(run / model / "metrics.json")} for model, (_, manifest) in native.items()},
        "paired_analysis_sha256": {name: sha256(run / "analysis" / name) for name in pairs},
        "translated_predictions_sha256": translated["predictions_sha256"],
        "english_source": "phase_01_original_baseline/results/baseline_results.json",
        "commands": commands,
        "decisions": {"selected_model": "phase_3b", "phase_5": "sweep selected alpha 0; no qualifying adaptation",
                      "native_adaptation_diagnostic": "declared before Native inference; reported separately"},
    })
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    native = commands.add_parser("native", help="Paired Native comparison with document bootstrap")
    native.add_argument("--gold", type=Path, required=True)
    native.add_argument("--control", type=Path, required=True, help="Control predictions.jsonl")
    native.add_argument("--candidate", type=Path, required=True, help="Candidate predictions.jsonl")
    native.add_argument("--output", type=Path, required=True)
    native.add_argument("--control-label", default="control")
    native.add_argument("--candidate-label", default="candidate")
    native.add_argument("--iterations", type=int)
    native.add_argument("--seed", type=int)
    translated = commands.add_parser("translated", help="Pinned Phase 3 translated comparison")
    translated.add_argument("--output", type=Path, required=True)
    errors = commands.add_parser("errors", help="Reproducible paired error sample")
    errors.add_argument("--gold", type=Path, required=True)
    errors.add_argument("--control-run", type=Path, required=True)
    errors.add_argument("--candidate-run", type=Path, required=True)
    errors.add_argument("--private-output", type=Path, required=True)
    errors.add_argument("--counts-output", type=Path, required=True)
    errors.add_argument("--per-case", type=int, default=25)
    errors.add_argument("--seed", type=int, default=42)
    report = commands.add_parser("report", help="Text-free tables and manifest from a completed run")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    handlers = {"native": cmd_native, "translated": cmd_translated, "errors": cmd_errors, "report": cmd_report}
    try:
        handlers[args.command](args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Phase 7 statistical_analysis {args.command}: {error}\n")


if __name__ == "__main__":
    main()
