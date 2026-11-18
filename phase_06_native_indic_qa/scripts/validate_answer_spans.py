"""Step 8a: validate generated QA candidates and compute answer offsets in code.

The LLM is never trusted for offsets. For each candidate:
- the answer must be an exact contiguous substring of the stored (NFC) context;
  only boundary punctuation/whitespace may be trimmed - no fuzzy re-writing
- if the answer occurs more than once, the evidence sentence picks the occurrence
- the question must be in the target script and language (Hindi vs Marathi checked)
- answer length, answer-in-question leakage and question length are checked
- soft issues (no question word, possible yes/no, evidence not found, copied
  context wording) are kept as warnings for human reviewers rather than rejected

Usage:
    python phase_06_native_indic_qa/scripts/validate_answer_spans.py --languages all
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

from rapidfuzz import fuzz

from common import (CONTEXTS_DIR, DIFFICULTIES, DOMAINS, GENERATED_DIR, QUESTION_TYPES, RESULTS_DIR,
                    VALIDATED_DIR, language_ok, load_config, nfc, read_json, read_jsonl, selected_languages,
                    word_count, write_json, write_jsonl)

BOUNDARY_CHARS = " \t\n\"'“”‘’«»()[]{}.,;:!?।॥-–—"

WH_WORDS = {
    "hi": ["क्या", "कौन", "कब", "कहाँ", "कहां", "कैसे", "कितना", "कितनी", "कितने", "किस", "किन", "क्यों", "कौनसा",
           "कौन-सा", "कौनसी", "कौन-सी", "किसे", "किसको", "किससे", "किसके"],
    "mr": ["काय", "कोण", "कधी", "कुठे", "कोठे", "कसे", "कसा", "कशी", "किती", "कोणत", "कोणी", "का ", "कशा", "कशाच",
           "कशामुळे", "कशासाठी", "कोणाल"],
    "bn": ["কী", "কি", "কোন", "কখন", "কোথায়", "কেন", "কত", "কীভাবে", "কিভাবে", "কে ", "কার", "কাকে", "কোনটি"],
    "ta": ["என்ன", "எது", "எவை", "எப்போது", "எங்கு", "எங்கே", "ஏன்", "எவ்வளவு", "எத்தனை", "எப்படி", "யார்", "எந்த",
           "எதனால்", "எதற்கு", "எவ்வாறு", "எதை", "எதில்", "எவற்", "எதன்", "எத்தகைய", "எவர்"],
}


def find_all(haystack: str, needle: str) -> list[int]:
    starts, i = [], haystack.find(needle)
    while i != -1:
        starts.append(i)
        i = haystack.find(needle, i + 1)
    return starts


def locate_answer(context: str, answer: str) -> tuple[str, list[int]]:
    """Return (answer_as_in_context, start offsets). Exact match only, after trimming
    boundary punctuation and allowing whitespace differences."""
    for candidate in (answer, answer.strip(BOUNDARY_CHARS)):
        if candidate and (starts := find_all(context, candidate)):
            return candidate, starts
    trimmed = answer.strip(BOUNDARY_CHARS)
    if trimmed:
        pattern = r"\s+".join(re.escape(part) for part in trimmed.split())
        matches = list(re.finditer(pattern, context))
        if matches:
            text = matches[0].group(0)
            return text, find_all(context, text)
    return answer, []


def locate_evidence(context: str, evidence: str, threshold: float) -> tuple[int, int] | None:
    if not evidence:
        return None
    start = context.find(evidence)
    if start != -1:
        return start, start + len(evidence)
    alignment = fuzz.partial_ratio_alignment(evidence, context, score_cutoff=threshold)
    if alignment is None:
        return None
    return alignment.dest_start, alignment.dest_end


def clean_enum(value, allowed: list[str], aliases: dict[str, str] | None = None) -> str | None:
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    key = (aliases or {}).get(key, key)
    return key if key in allowed else None


DOMAIN_ALIASES = {"water_management": "irrigation", "irrigation_water_management": "irrigation",
                  "irrigation_&_water_management": "irrigation", "soil": "soil_management",
                  "fertilizers": "fertilizers_nutrition", "plant_nutrition": "fertilizers_nutrition",
                  "nutrient_management": "fertilizers_nutrition", "diseases": "plant_diseases",
                  "plant_disease": "plant_diseases", "pests": "pest_management",
                  "harvesting": "harvesting_storage", "post_harvest": "harvesting_storage",
                  "technology": "agricultural_technology", "crop_management": "crop_production"}
TYPE_ALIASES = {"benefit_based": "benefit", "benefits": "benefit", "method": "procedural",
                "cause": "causal", "condition": "conditional", "numeric": "numerical", "time": "temporal"}


def validate_item(item, ctx: dict, lang: str, cfg: dict) -> dict:
    fails, warns = [], []
    if not isinstance(item, dict):
        return {"status": "failed", "fail_reasons": ["not_an_object"], "warnings": []}
    question = nfc(str(item.get("question") or "")).strip()
    answer_raw = nfc(str(item.get("answer_text") or "")).strip()
    evidence = nfc(str(item.get("evidence_sentence") or "")).strip()
    context = ctx["context"]

    result = {"question": question, "answer_raw": answer_raw, "evidence_sentence": evidence,
              "question_type": clean_enum(item.get("question_type"), QUESTION_TYPES, TYPE_ALIASES),
              "difficulty": clean_enum(item.get("difficulty"), DIFFICULTIES),
              "domain": clean_enum(item.get("domain"), DOMAINS, DOMAIN_ALIASES)}
    if not question:
        fails.append("empty_question")
    if not answer_raw:
        fails.append("empty_answer")
    if fails:
        return {**result, "status": "failed", "fail_reasons": fails, "warnings": warns}

    answer, starts = locate_answer(context, answer_raw)
    if not starts:
        fails.append("answer_not_in_context")
    else:
        if answer != answer_raw:
            warns.append("answer_boundary_trimmed")
        start = starts[0]
        span = locate_evidence(context, evidence, cfg["evidence_match_threshold"])
        if span is None:
            warns.append("evidence_not_found")
        else:
            inside = [s for s in starts if span[0] - 5 <= s and s + len(answer) <= span[1] + 5]
            if inside:
                start = inside[0]
            else:
                warns.append("answer_outside_evidence")
                start = min(starts, key=lambda s: abs(s - span[0]))
        if len(starts) > 1:
            warns.append("answer_occurs_multiple_times" if span else "ambiguous_answer_occurrence")
        assert context[start:start + len(answer)] == answer
        result["answers"] = {"text": [answer], "answer_start": [start]}

    if len(answer) < cfg["min_answer_chars"]:
        fails.append("answer_too_short")
    if word_count(answer) > cfg["max_answer_words"]:
        fails.append("answer_too_long")
    if len(answer) > cfg["max_answer_context_fraction"] * len(context):
        fails.append("answer_is_most_of_context")

    ok, reason = language_ok(question, lang, cfg["min_question_script_ratio"])
    if not ok:
        fails.append(f"question_language:{reason}")
    qwords = word_count(question)
    if qwords < cfg["min_question_words"]:
        fails.append("question_too_short")
    if qwords > cfg["max_question_words"]:
        fails.append("question_too_long")
    if len(answer) >= 4 and answer in question:
        fails.append("answer_in_question")
    if question.rstrip("?？ ") == answer:
        fails.append("question_equals_answer")

    if word_count(answer) > cfg.get("long_answer_words", 10):
        warns.append("long_answer")
    if answer_raw.rstrip().endswith(("।", "॥", ".")) and word_count(answer) >= 5:
        warns.append("answer_is_full_sentence")
    if not question.endswith(("?", "？")):
        warns.append("no_question_mark")
    if not any(w in question + " " for w in WH_WORDS[lang]):
        warns.append("no_question_word_possible_yes_no")
    if fuzz.partial_ratio(question.rstrip("?？ "), context) >= 95 and qwords >= 6:
        warns.append("question_copies_context_wording")
    for field in ("question_type", "difficulty", "domain"):
        if result[field] is None:
            warns.append(f"invalid_{field}")

    return {**result, "status": "failed" if fails else "passed", "fail_reasons": fails, "warnings": warns}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", default="all")
    args = parser.parse_args()
    cfg = load_config()["validation"]

    report = read_json(RESULTS_DIR / "validation_report.json", {}) or {}
    for lang in selected_languages(args.languages):
        contexts = {c["context_id"]: c for c in read_jsonl(CONTEXTS_DIR / f"{lang}_contexts.jsonl")}
        latest = {}
        for row in read_jsonl(GENERATED_DIR / f"{lang}_raw.jsonl"):
            latest[row["context_id"]] = row  # the newest attempt per context wins
        out, fail_counts, warn_counts = [], Counter(), Counter()
        ctx_status = Counter(r["status"] for r in latest.values())
        for cid, row in latest.items():
            ctx = contexts.get(cid)
            if ctx is None or row["status"] != "ok":
                if ctx is None:
                    fail_counts["context_missing_rechunked"] += len(row.get("items") or [])
                continue
            for i, item in enumerate(row["items"], start=1):
                checked = validate_item(item, ctx, lang, cfg)
                fail_counts.update(checked["fail_reasons"])
                warn_counts.update(checked["warnings"])
                out.append({
                    "candidate_id": f"{cid}_Q{i}",
                    "context_id": cid, "document_id": ctx["document_id"], "language": lang,
                    "title": ctx["title"], "context": ctx["context"], **checked,
                    "domain_hint": ctx.get("domain_hint"), "page_start": ctx.get("page_start"),
                    "page_end": ctx.get("page_end"), "source_url": ctx.get("source_url"),
                    "region": ctx.get("region"), "model": row.get("model_version") or row["model"],
                    "prompt_version": row["prompt_version"],
                })
        write_jsonl(VALIDATED_DIR / f"{lang}_validated.jsonl", out)
        passed = sum(r["status"] == "passed" for r in out)
        report[lang] = {
            "contexts_with_responses": len(latest), "context_status": dict(ctx_status),
            "candidates": len(out), "passed": passed, "failed": len(out) - passed,
            "pass_rate": round(passed / len(out), 3) if out else None,
            "failure_reasons": dict(fail_counts.most_common()),
            "warnings_on_all_candidates": dict(warn_counts.most_common()),
        }
        print(f"[{lang}] {passed}/{len(out)} candidates passed; top failures "
              f"{dict(fail_counts.most_common(4))}")
    write_json(RESULTS_DIR / "validation_report.json", report)


if __name__ == "__main__":
    main()
