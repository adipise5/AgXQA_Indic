"""Sliding-window QA preprocessing and character-offset decoding."""

from collections import Counter, defaultdict
from math import isfinite


def label_span(offsets, start, end):
    left = next((i for i, pair in enumerate(offsets) if pair and pair[0] <= start < pair[1]), None)
    right = next((i for i in range(len(offsets) - 1, -1, -1)
                  if offsets[i] and offsets[i][0] < end <= offsets[i][1]), None)
    if left is not None and right is not None and right >= left:
        return left, right
    return None


def answer_bounds(row):
    spans = []
    for text, start in zip(row["answers"]["text"], row["answers"]["answer_start"]):
        if not text.strip() or type(start) is not int or start < 0 or row["context"][start:start + len(text)] != text:
            raise ValueError(f"Invalid answer offset: {row['id']}")
        spans.append((start + len(text) - len(text.lstrip()), start + len(text.rstrip())))
    if not spans or len(spans) != len(row["answers"]["text"]) or len(spans) != len(row["answers"]["answer_start"]):
        raise ValueError(f"Invalid answer arrays: {row['id']}")
    return spans


def prepare_features(rows, tokenizer, settings, labels=False):
    if not tokenizer.is_fast or tokenizer.padding_side != "right" or tokenizer.cls_token_id is None:
        raise ValueError("Use an XLM-R fast tokenizer with right padding and a CLS token")
    max_length, stride = settings["max_seq_length"], settings["doc_stride"]
    if max_length < 8 or max_length > tokenizer.model_max_length or not 0 <= stride < max_length:
        raise ValueError("Invalid sequence length or stride for this tokenizer")
    ids = [row["id"] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Tokenization requires a nonempty split with unique IDs")

    features, statistics = [], {}
    pair_special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    for start in range(0, len(rows), 64):
        batch = rows[start:start + 64]
        questions = [row["question"].lstrip() for row in batch]
        contexts = [row["context"] for row in batch]
        question_ids = tokenizer(questions, add_special_tokens=False, verbose=False)["input_ids"]
        context_ids = tokenizer(contexts, add_special_tokens=False, verbose=False)["input_ids"]
        bounds = [answer_bounds(row) for row in batch]
        for row, question, context in zip(batch, question_ids, context_ids):
            capacity = max_length - len(question) - pair_special_tokens
            if capacity <= stride:
                raise ValueError(f"{row['id']}: stride {stride} is not below available context capacity {capacity}")
            statistics[row["id"]] = {"language": row["language"], "question_tokens": len(question),
                                     "context_tokens": len(context), "windows": 0, "positive_windows": 0,
                                     "input_tokens": 0, "best_gold_token_length": None}
        encoding = tokenizer(questions, contexts, max_length=max_length, stride=stride,
                             truncation="only_second", return_overflowing_tokens=True,
                             return_offsets_mapping=True, padding=False, verbose=False)
        for i, j in enumerate(encoding["overflow_to_sample_mapping"]):
            identifier = batch[j]["id"]
            offsets = [tuple(pair) if sequence == 1 and pair[1] > pair[0] else None
                       for pair, sequence in zip(encoding["offset_mapping"][i], encoding.sequence_ids(i))]
            inputs = {key: encoding[key][i] for key in tokenizer.model_input_names if key in encoding}
            cls = inputs["input_ids"].index(tokenizer.cls_token_id)
            spans = [span for left, right in bounds[j] if (span := label_span(offsets, left, right)) is not None]
            if labels:
                inputs["start_positions"], inputs["end_positions"] = spans[0] if spans else (cls, cls)
            features.append({"id": identifier, "inputs": inputs, "offsets": offsets})
            stat = statistics[identifier]
            stat["windows"] += 1
            stat["input_tokens"] += len(inputs["input_ids"])
            if spans:
                stat["positive_windows"] += 1
                shortest = min(end - begin + 1 for begin, end in spans)
                previous = stat["best_gold_token_length"]
                stat["best_gold_token_length"] = shortest if previous is None else min(shortest, previous)
    return features, statistics


def summarize_tokenization(statistics, max_answer_length):
    by_language = defaultdict(list)
    for identifier, record in statistics.items():
        by_language[record["language"]].append((identifier, record))
    summary = {}
    for language, examples in sorted(by_language.items()):
        records = [record for _, record in examples]
        counts = Counter(record["windows"] for record in records)
        lengths = sorted(record["context_tokens"] for record in records)
        summary[language] = {
            "examples": len(records), "windows": sum(record["windows"] for record in records),
            "extra_overflow_windows": sum((windows - 1) * count for windows, count in counts.items()),
            "window_count_histogram": dict(sorted(counts.items())),
            "mean_question_tokens": sum(record["question_tokens"] for record in records) / len(records),
            "mean_context_tokens": sum(lengths) / len(lengths), "max_context_tokens": lengths[-1],
            "unpadded_feature_tokens": sum(record["input_tokens"] for record in records),
            "without_gold_in_any_window": [identifier for identifier, record in examples if not record["positive_windows"]],
            "gold_exceeds_decode_limit": [identifier for identifier, record in examples
                                          if (record["best_gold_token_length"] or 0) > max_answer_length],
        }
    return summary


def decode_logits(rows, features, start_logits, end_logits, settings):
    if len(features) != len(start_logits) or len(features) != len(end_logits):
        raise ValueError("Feature and logit counts differ")
    by_id = {row["id"]: row for row in rows}
    best = {}
    width = settings["n_best_size"]
    max_answer = settings["max_answer_length"]
    if width < 1 or max_answer < 1:
        raise ValueError("Decoding limits must be positive")
    for feature, starts, ends in zip(features, start_logits, end_logits):
        offsets = feature["offsets"]
        starts, ends = starts[:len(offsets)], ends[:len(offsets)]
        if len(starts) != len(offsets) or len(ends) != len(offsets) or not all(
                isfinite(float(value)) for values in (starts, ends) for value in values):
            raise ValueError("Invalid QA logits")
        valid = [i for i, pair in enumerate(offsets) if pair is not None]
        if not valid:
            continue
        start_candidates = sorted(valid, key=lambda i: (-starts[i], i))[:width]
        end_candidates = sorted(valid, key=lambda i: (-ends[i], i))[:width]
        pairs = [(i, j) for i in start_candidates for j in end_candidates
                 if i <= j < i + max_answer and offsets[i][0] < offsets[j][1]]
        if not pairs:
            pairs = [(i, i) for i in valid]
        i, j = max(pairs, key=lambda pair: (float(starts[pair[0]]) + float(ends[pair[1]]), -pair[0], -pair[1]))
        identifier = feature["id"]
        score = float(starts[i]) + float(ends[j])
        if identifier not in best or score > best[identifier]["span_score"]:
            left, right = offsets[i][0], offsets[j][1]
            best[identifier] = {"predicted_answer": by_id[identifier]["context"][left:right],
                                "predicted_start": left, "predicted_end": right, "span_score": score}
    if set(best) != set(by_id):
        raise ValueError("A test example has no valid predicted context span")
    return best
