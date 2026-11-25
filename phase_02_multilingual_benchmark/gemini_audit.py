"""Measure coverage, extractability, and split overlap for the retained release."""

from collections import Counter, defaultdict
from math import ceil
import random
from unicodedata import normalize

from .pipeline import COUNTS, LANGUAGES, digest, reference_urls, span_valid


ALL_LANGUAGES = ("en", *LANGUAGES)


def excluded_groups(sources, cohort, raw, validation_raw):
    retained = {row["id"] for row in cohort}
    return [{"original_id": row["id"], "split": row["split"], "category": row["category"],
             "source_document_id": row["source_document_id"],
             "flat_languages_present": [lang for lang in ALL_LANGUAGES if f"{row['id']}_{lang}" in raw],
             "raw_validation_languages_present": sorted(validation_raw.get(row["id"], {}).get("translations", {})),
             "reason": "incomplete_parallel_flat_export" if any(f"{row['id']}_{lang}" in raw for lang in ALL_LANGUAGES)
                       else "absent_from_all_flat_exports",
             "original_per_id_rejection_reason": None}
            for row in sources if row["id"] not in retained]


def exclusion_distribution(sources, cohort):
    retained = {row["id"] for row in cohort}
    result = {}
    for split in COUNTS:
        subset = [row for row in sources if row["split"] == split]
        result[split] = {}
        for key in ("category", "source_document_id"):
            all_counts = Counter(row[key] for row in subset)
            missing = Counter(row[key] for row in subset if row["id"] not in retained)
            result[split][key] = {name: {"source_groups": total, "excluded_groups": missing[name],
                                        "exclusion_rate": missing[name] / total}
                                  for name, total in sorted(all_counts.items())}
    return result


def extractability(rows, sources):
    statistics = defaultdict(Counter)
    for row in rows:
        context = row["context"]
        answers = row["answers"]["text"]
        starts = row["answers"]["answer_start"]
        source_count = len(sources[row["original_id"]]["answers"]["text"])
        stat = statistics[(row["split"], row["language"])]
        stat["rows"] += 1
        stat["multiple_reference_rows"] += len(answers) > 1
        stat["all_references_found_raw"] += bool(answers) and all(text.strip() and text in context for text in answers)
        nfc_context = normalize("NFC", context)
        stat["all_references_found_nfc"] += bool(answers) and all(text.strip() and normalize("NFC", text) in nfc_context for text in answers)
        stat["all_offsets_valid"] += bool(answers) and len(answers) == len(starts) and all(
            span_valid(context, text, offset) for text, offset in zip(answers, starts))
        stat["english_source_references"] += source_count
        stat["exported_references"] += len(answers)
        stat["rows_with_fewer_references_than_source"] += len(answers) < source_count
        stat["reference_count_equal_to_source"] += len(answers) == source_count
        stat["reference_count_shortfall"] += max(0, source_count - len(answers))
    return {f"{split}/{language}": dict(values) for (split, language), values in sorted(statistics.items())}


def validate_parallel(rows, cohort):
    expected = {(source["id"], language): source["split"] for source in cohort for language in ALL_LANGUAGES}
    seen = set()
    for row in rows:
        key = row["original_id"], row["language"]
        if key in seen or row["id"] != f"{key[0]}_{key[1]}" or row["split"] != expected.get(key):
            raise ValueError(f"Duplicate, unexpected or wrong-split parallel row: {row['id']}")
        seen.add(key)
    if seen != set(expected):
        raise ValueError(f"Missing {len(set(expected) - seen)} parallel rows")


def train_test_overlap(rows):
    train = defaultdict(set)
    test = defaultdict(list)
    documents = {row["source_document_id"] for row in rows if row["split"] == "train"}
    for row in rows:
        if row["split"] == "train":
            train[row.get("language", "en")].add(row["context"])
        elif row["split"] == "test":
            test[row.get("language", "en")].append(row)
    result = {}
    for language, group in sorted(test.items()):
        matching = [row["original_id"] for row in group if row["context"] in train[language]]
        result[language] = {"test_rows": len(group), "exact_train_context_rows": len(matching),
                            "shared_document_rows": sum(row["source_document_id"] in documents for row in group),
                            "exact_train_context_test_ids": sorted(matching)}
    return result


def validation_components(train_ids, keys, seed):
    by_key = defaultdict(set)
    for identifier in train_ids:
        for key in keys[identifier]:
            by_key[key].add(identifier)
    remaining = set(train_ids)
    components = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        block, queue = {first}, [first]
        while queue:
            for key in keys[queue.pop()]:
                neighbors = by_key[key] & remaining
                remaining.difference_update(neighbors)
                block.update(neighbors)
                queue.extend(neighbors)
        components.append(block)

    random.Random(seed).shuffle(components)
    selected = set()
    target = ceil(len(train_ids) * .1)
    for block in components[:-1]:
        selected.update(block)
        if len(selected) >= target:
            break
    return selected


def document_disjoint(rows, cohort, seed):
    """Keep test frozen and filter validation, then train, by shared document or passage."""
    keys = {row["id"]: {row["source_document_id"], *reference_urls(row["references"])} for row in cohort}
    for row in rows:
        passage = " ".join(normalize("NFC", row["context"]).casefold().split())
        keys[row["original_id"]].add("passage:" + digest(passage))

    higher_priority = set()
    kept, dropped = set(), []
    for split in ("test", "validation", "train"):
        current = set()
        for source in cohort:
            if source["split"] != split:
                continue
            identifier = source["id"]
            if overlap := keys[identifier] & higher_priority:
                dropped.append({"original_id": identifier, "split": split,
                                "reason": "shares_document_or_passage_with_retained_higher_priority_split",
                                "matching_key_types": sorted({"passage" if key.startswith("passage:") else "document"
                                                              for key in overlap})})
            else:
                kept.add(identifier)
                current.update(keys[identifier])
        higher_priority.update(current)
    retained = [row for row in cohort if row["id"] in kept]
    reassigned = set()
    if not any(row["split"] == "validation" for row in retained):
        train_ids = [row["id"] for row in retained if row["split"] == "train"]
        reassigned = validation_components(train_ids, keys, seed)
        retained = [{**row, "split": "validation", "official_split": "train"} if row["id"] in reassigned else row
                    for row in retained]
    split_keys = {split: set() for split in COUNTS}
    for source in retained:
        split_keys[source["split"]].update(keys[source["id"]])
    if any(split_keys[left] & split_keys[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Document-disjoint construction retained a shared key")
    return retained, dropped, sorted(reassigned)
