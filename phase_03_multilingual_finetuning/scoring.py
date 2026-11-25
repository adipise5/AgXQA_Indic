"""Multilingual scores and matched question-level comparisons."""

from collections import Counter, defaultdict
import hashlib
from itertools import combinations
from math import isfinite
import random
from statistics import mean

from shared.metrics import best_scores
from .common import LANGUAGES


METRICS = ("exact_match", "f1")


def scored_predictions(rows, decoded, token_stats, experiment, seed):
    if len(rows) != len(decoded) or set(decoded) != {row["id"] for row in rows}:
        raise ValueError("Prediction IDs must cover the split exactly")
    english_contexts = {row["original_id"]: hashlib.sha256(row["context"].encode("utf-8")).hexdigest()
                        for row in rows if row["language"] == "en"}
    records = []
    for row in rows:
        span = decoded[row["id"]]
        left, right = span["predicted_start"], span["predicted_end"]
        answer = span["predicted_answer"]
        if not 0 <= left < right <= len(row["context"]) or row["context"][left:right] != answer:
            raise ValueError(f"Prediction span does not match the exact passage: {row['id']}")
        references = row["answers"]["text"]
        score = best_scores(answer, references)
        records.append({
            "id": row["id"], "original_id": row["original_id"], "language": row["language"],
            "experiment": experiment, "model": "FacebookAI/xlm-roberta-base", "seed": seed,
            "question": row["question"], "gold_answers": references, **span,
            "exact_match": score["em"], "f1": score["f1"],
            "category": row.get("category"), "human_reviewed": row.get("human_reviewed", False),
            "source_document_id": row["source_document_id"],
            "english_context_sha256": english_contexts.get(row["original_id"]),
            "context_chars": len(row["context"]), "answer_chars": len(references[0]),
            "context_tokens": token_stats[row["id"]]["context_tokens"],
            "window_count": token_stats[row["id"]]["windows"],
        })
    return records


def aggregate(records):
    values = defaultdict(list)
    for row in records:
        values[row["language"]].append(row)
    if not values or set(values) - set(LANGUAGES):
        raise ValueError("Expected scores for at least one supported language")
    language_scores = {
        lang: {**{key: 100 * mean(row[key] for row in values[lang]) for key in METRICS}, "count": len(values[lang])}
        for lang in LANGUAGES if lang in values
    }
    result = dict(language_scores)
    for name, langs in (("indic_macro", LANGUAGES[1:]), ("overall_macro", tuple(language_scores))):
        if set(langs) <= set(language_scores):
            result[name] = {**{key: mean(language_scores[lang][key] for lang in langs) for key in METRICS},
                            "count": sum(language_scores[lang]["count"] for lang in langs)}
    return result


def parallel_scores(records):
    by_id = defaultdict(dict)
    for row in records:
        group = by_id[row["original_id"]]
        language = row["language"]
        if language in group:
            raise ValueError(f"Duplicate prediction: {row['id']}")
        if language not in LANGUAGES or row["id"] != f"{row['original_id']}_{language}":
            raise ValueError(f"Prediction ID/language mismatch: {row['id']}")
        if row["exact_match"] not in (0, 1) or not isfinite(row["f1"]) or not 0 <= row["f1"] <= 1:
            raise ValueError(f"Invalid score: {row['id']}")
        group[language] = row
    return by_id


def consistency(records):
    groups = parallel_scores(records)
    complete = [group for group in groups.values() if set(group) == set(LANGUAGES)]
    if not complete:
        raise ValueError("No complete parallel question groups")
    n = len(complete)
    exact = Counter(sum(group[lang]["exact_match"] for lang in LANGUAGES) for group in complete)
    return {
        "complete_groups": n, "excluded_groups": len(groups) - n,
        "correct_in_n_languages": [{"correct_languages": k, "count": exact[k], "percent": 100 * exact[k] / n}
                                   for k in range(len(LANGUAGES) + 1)],
        "pairwise_agreement": [
            {"language_a": lang, "language_b": other,
             "percent_agree": 100 * sum(g[lang]["exact_match"] == g[other]["exact_match"] for g in complete) / n}
            for lang, other in combinations(LANGUAGES, 2)],
    }


def compare_inputs(a, b):
    a_groups, b_groups = parallel_scores(a), parallel_scores(b)
    if not a_groups or set(a_groups) != set(b_groups):
        raise ValueError("The experiments must cover the same original question IDs")
    for identifier, group in a_groups.items():
        other = b_groups[identifier]
        if set(group) != set(LANGUAGES) or set(other) != set(group):
            raise ValueError(f"Missing matched language: {identifier}")
        for lang, row in group.items():
            if any(row[key] != other[lang][key] for key in
                   ("question", "gold_answers", "source_document_id", "english_context_sha256")):
                raise ValueError(f"Different test annotations: {identifier}/{lang}")
    return a_groups, b_groups


def percentile(ordered, probability):
    location = (len(ordered) - 1) * probability
    low = int(location)
    upper = min(low + 1, len(ordered) - 1)
    return ordered[low] + (location - low) * (ordered[upper] - ordered[low])


def bootstrap(a_groups, b_groups, iterations, confidence, seed, unit):
    if iterations < 1 or not 0 < confidence < 1 or unit not in ("question", "context", "document"):
        raise ValueError("Invalid paired bootstrap settings")
    buckets = defaultdict(list)
    for identifier in sorted(a_groups):
        key = identifier
        if unit != "question":
            field = "source_document_id" if unit == "document" else "english_context_sha256"
            key = a_groups[identifier]["en"][field]
            if not key:
                raise ValueError(f"Missing resampling identity for {identifier}")
        buckets[key].append(identifier)
    populations = [(name, (name,)) for name in LANGUAGES]
    populations += [("indic_macro", LANGUAGES[1:]), ("overall_macro", LANGUAGES)]
    keys = [(population, metric) for population, _ in populations for metric in METRICS]
    units = [
        [sum(100 * mean(b_groups[q][lang][metric] - a_groups[q][lang][metric] for lang in langs) for q in members)
         for _, langs in populations for metric in METRICS] + [len(members)]
        for members in buckets.values()
    ]
    rng = random.Random(seed)
    samples = [[] for _ in keys]
    for _ in range(iterations):
        picked = rng.choices(units, k=len(units))
        size = sum(bucket[-1] for bucket in picked)
        for sample, totals in zip(samples, zip(*picked)):
            sample.append(sum(totals) / size)
    tail = (1 - confidence) / 2
    return [
        {"language": name, "metric": key, "delta_pp": sum(bucket[i] for bucket in units) / len(a_groups),
         "lower_pp": percentile(sorted(sample), tail), "upper_pp": percentile(sorted(sample), 1 - tail),
         "unit": unit, "units": len(units), "question_groups": len(a_groups),
         "iterations": iterations, "confidence": confidence, "seed": seed}
        for i, ((name, key), sample) in enumerate(zip(keys, samples))
    ]


def review_sample(a_groups, b_groups, size, seed):
    if size < 1:
        raise ValueError("Review sample size must be positive")
    result = []
    for lang in LANGUAGES[1:]:
        wrong = [q for q in sorted(a_groups) if not a_groups[q][lang]["exact_match"] or not b_groups[q][lang]["exact_match"]]
        rng = random.Random(f"{seed}:{lang}")
        for identifier in rng.sample(wrong, min(size, len(wrong))):
            for group in (a_groups, b_groups):
                row = group[identifier][lang]
                result.append({**row, "model": row["experiment"], "gold_answer": " | ".join(row["gold_answers"]),
                               "em": row["exact_match"], "error_category": "", "reviewer": "", "comments": ""})
    return result
