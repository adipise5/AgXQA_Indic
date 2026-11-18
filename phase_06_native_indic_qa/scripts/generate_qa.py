"""Step 7: generate native-language QA candidates for each context with Gemini.

One API call per context, using prompts/qa_generation_prompt.txt verbatim with
$LANGUAGE and $CONTEXT filled in. Questions are generated directly in the target
language from the native context - nothing is generated in English or translated.

Raw responses are appended to data/native_indic_qa/generated_qa/<lang>_raw.jsonl
together with the model, prompt version/hash and generation settings, so runs can
be resumed (already-processed contexts are skipped) and reported exactly.

Usage:
    python phase_06_native_indic_qa/scripts/generate_qa.py --dry-run
    python phase_06_native_indic_qa/scripts/generate_qa.py --languages hi --limit 5
    python phase_06_native_indic_qa/scripts/generate_qa.py --languages all --limit 130
    python phase_06_native_indic_qa/scripts/generate_qa.py --report-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from string import Template
from zoneinfo import ZoneInfo

from common import (CONTEXTS_DIR, GENERATED_DIR, LANGUAGES, PROMPT_PATH, REPO_ROOT, RESULTS_DIR,
                    append_jsonl, load_config, read_json, read_jsonl, selected_languages, sha256_text,
                    write_json)

COUNTER_PATH = GENERATED_DIR / "api_request_counter.json"


def pacific_day() -> str:
    # Gemini's requests-per-day quota resets at midnight Pacific time.
    return datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()


def calls_today() -> dict[str, int]:
    """Calls made today, per API-key environment variable name."""
    state = read_json(COUNTER_PATH, {}) or {}
    if state.get("pacific_day") != pacific_day():
        return {}
    calls = state.get("calls", {})
    return {"unattributed": int(calls)} if isinstance(calls, int) else {k: int(v) for k, v in calls.items()}


def reserve_request(key_env: str, limit: int) -> bool:
    calls = calls_today()
    if calls.get(key_env, 0) >= limit:
        return False
    # Count before calling, so a crash can never under-count a request.
    calls[key_env] = calls.get(key_env, 0) + 1
    write_json(COUNTER_PATH, {"pacific_day": pacific_day(), "calls": calls})
    return True


class KeyPool:
    """API keys tried in order, each with its own daily budget. A key is retired for
    this run when its budget is spent or it returns quota errors twice in a row.
    Only environment-variable names are logged, never key values."""

    def __init__(self, specs: list[dict]):
        self.specs = [s for s in specs if os.getenv(s["env"])]
        self.missing = [s["env"] for s in specs if not os.getenv(s["env"])]
        self.index, self.clients, self.strikes = 0, {}, 0

    def current(self) -> dict | None:
        while self.index < len(self.specs):
            spec = self.specs[self.index]
            if calls_today().get(spec["env"], 0) < spec["max_calls_per_pacific_day"]:
                return spec
            print(f"  {spec['env']}: daily budget used - switching key")
            self.retire()
        return None

    def client(self, spec: dict):
        from google import genai
        if spec["env"] not in self.clients:
            self.clients[spec["env"]] = genai.Client(api_key=os.environ[spec["env"]])
        return self.clients[spec["env"]]

    def quota_error(self) -> None:
        self.strikes += 1
        if self.strikes >= 2:
            print(f"  {self.specs[self.index]['env']}: repeated quota errors - switching key")
            self.retire()

    def ok(self) -> None:
        self.strikes = 0

    def retire(self) -> None:
        self.index += 1
        self.strikes = 0


def key_specs(gen_cfg: dict) -> list[dict]:
    if gen_cfg.get("api_keys"):
        return gen_cfg["api_keys"]
    return [{"env": gen_cfg["api_key_env"], "max_calls_per_pacific_day": gen_cfg["max_calls_per_pacific_day"]}]


def order_contexts(contexts: list[dict], domain: str | None) -> list[dict]:
    """Round-robin across documents so a --limit run still covers many sources."""
    if domain:
        contexts = [c for c in contexts if c.get("domain_hint") == domain]
    by_doc = defaultdict(list)
    for c in contexts:
        by_doc[c["document_id"]].append(c)
    queues = [by_doc[d] for d in sorted(by_doc)]
    ordered = []
    while any(queues):
        for q in queues:
            if q:
                ordered.append(q.pop(0))
    return ordered


def parse_items(text: str) -> list:
    try:
        parsed = json.loads(text or "")
    except json.JSONDecodeError:
        # The model occasionally emits a stray backslash ("Invalid \escape"); escape
        # backslashes that do not start a valid JSON escape and try once more.
        parsed = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text or ""))
    if isinstance(parsed, dict):
        for key in ("qa_pairs", "questions", "items", "data", "results"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        # A single QA object returned without the array.
        if "question" in parsed:
            return [parsed]
        raise ValueError(f"JSON object without a QA array (keys: {list(parsed)[:6]})")
    if not isinstance(parsed, list):
        raise ValueError("response is not a JSON array")
    return parsed


def build_report(gen_cfg: dict, prompt_hash: str) -> dict:
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "configured": gen_cfg, "current_prompt_sha256": prompt_hash, "languages": {}}
    for lang in LANGUAGES:
        rows = read_jsonl(GENERATED_DIR / f"{lang}_raw.jsonl")
        if not rows:
            continue
        contexts = read_jsonl(CONTEXTS_DIR / f"{lang}_contexts.jsonl")
        status = Counter(r["status"] for r in rows)
        report["languages"][lang] = {
            "contexts_available": len(contexts),
            "contexts_processed": len(rows),
            "status": dict(status),
            "candidates": sum(len(r.get("items") or []) for r in rows),
            "api_calls": sum(r.get("attempts", 1) for r in rows),
            "models_used": dict(Counter(r.get("model_version") or r["model"] for r in rows)),
            "prompt_versions_used": dict(Counter(f"{r['prompt_version']}:{r['prompt_sha256'][:12]}"
                                                 for r in rows)),
            "temperatures_used": dict(Counter(str(r["generation_config"].get("temperature")) for r in rows)),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", default="all")
    parser.add_argument("--limit", type=int, default=None, help="Max new contexts per language this run")
    parser.add_argument("--domain", default=None, help="Only contexts with this domain_hint (e.g. irrigation)")
    parser.add_argument("--dry-run", action="store_true", help="Show counts and one filled prompt; no API calls")
    parser.add_argument("--retry-errors", action="store_true", help="Re-run contexts whose last attempt failed")
    parser.add_argument("--report-only", action="store_true", help="Rebuild results/generation_report.json only")
    args = parser.parse_args()

    gen_cfg = load_config()["generation"]
    template_text = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_hash = sha256_text(template_text)
    template = Template(template_text)
    if args.report_only:
        write_json(RESULTS_DIR / "generation_report.json", build_report(gen_cfg, prompt_hash))
        print("Wrote results/generation_report.json")
        return

    plan = {}
    for lang in selected_languages(args.languages):
        raw_path = GENERATED_DIR / f"{lang}_raw.jsonl"
        done = {}
        for r in read_jsonl(raw_path):
            done[r["context_id"]] = r["status"]  # last record per context wins
        skip = {cid for cid, s in done.items() if s != "error" or not args.retry_errors}
        pending = [c for c in order_contexts(read_jsonl(CONTEXTS_DIR / f"{lang}_contexts.jsonl"), args.domain)
                   if c["context_id"] not in skip]
        plan[lang] = pending[:args.limit] if args.limit is not None else pending
        print(f"[{lang}] processed {len(done)} | pending this run {len(plan[lang])}")

    used = calls_today()
    budgets = ", ".join(f"{s['env']} {used.get(s['env'], 0)}/{s['max_calls_per_pacific_day']}"
                        for s in key_specs(gen_cfg))
    print(f"Model {gen_cfg['model']} | temperature {gen_cfg['temperature']} | prompt {gen_cfg['prompt_version']} "
          f"({prompt_hash[:12]}) | calls today: {budgets}")

    if args.dry_run:
        sample = next((ctxs[0] for ctxs in plan.values() if ctxs), None)
        if sample:
            lang = sample["language"]
            print("\n--- sample prompt ---\n" + template.substitute(
                LANGUAGE=f"{LANGUAGES[lang]['name']} ({LANGUAGES[lang]['native']})", CONTEXT=sample["context"]))
        return

    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    pool = KeyPool(key_specs(gen_cfg))
    if pool.missing:
        print(f"(not set, skipped: {', '.join(pool.missing)})")
    if not pool.specs:
        raise SystemExit("No API key found - set GEMINI_API_KEY in .env")
    request_config = {"temperature": gen_cfg["temperature"], "response_mime_type": "application/json"}

    # Interleave languages so a budget stop leaves every language with some progress.
    queue = []
    for i in range(max((len(v) for v in plan.values()), default=0)):
        queue.extend(plan[lang][i] for lang in plan if i < len(plan[lang]))

    for n, ctx in enumerate(queue, start=1):
        lang = ctx["language"]
        prompt = template.substitute(LANGUAGE=f"{LANGUAGES[lang]['name']} ({LANGUAGES[lang]['native']})",
                                     CONTEXT=ctx["context"])
        record = {"context_id": ctx["context_id"], "document_id": ctx["document_id"], "language": lang,
                  "model": gen_cfg["model"], "model_version": None,
                  "prompt_version": gen_cfg["prompt_version"], "prompt_sha256": prompt_hash,
                  "generation_config": request_config, "status": "error", "items": None,
                  "error": None, "raw_text": None, "attempts": 0}
        stop = False
        attempt = 0
        while attempt < gen_cfg["max_retries"]:
            spec = pool.current()
            if spec is None or not reserve_request(spec["env"], spec["max_calls_per_pacific_day"]):
                print("All API keys have used their daily budget - run again after midnight Pacific time.")
                stop = True
                break
            attempt += 1
            record["attempts"] = attempt
            record["api_key_slot"] = spec["env"]  # variable name only, never the key
            try:
                response = pool.client(spec).models.generate_content(model=gen_cfg["model"], contents=prompt,
                                                                     config=request_config)
                record["model_version"] = getattr(response, "model_version", None)
                record["raw_text"] = response.text
                items = parse_items(response.text)
                record.update(status="ok" if items else "empty", items=items, error=None)
                pool.ok()
                break
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"[:500]
                print(f"  {ctx['context_id']} attempt {attempt} ({spec['env']}): {record['error'][:140]}")
                if "RESOURCE_EXHAUSTED" in str(error) or "429" in str(error):
                    pool.quota_error()  # second strike moves to the next key
                    attempt -= 1        # a quota refusal is not a content failure
                    time.sleep(20)
                    continue
                time.sleep(8 * attempt)
        if record["attempts"]:
            if record["status"] != "error":
                record["raw_text"] = None  # parsed items are kept; raw text only for failures
            record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            append_jsonl(GENERATED_DIR / f"{lang}_raw.jsonl", record)
            print(f"[{n}/{len(queue)}] {ctx['context_id']}: {record['status']} "
                  f"({len(record['items'] or [])} candidates)")
        if stop:
            break
        time.sleep(gen_cfg["rate_limit_delay_seconds"])

    write_json(RESULTS_DIR / "generation_report.json", build_report(gen_cfg, prompt_hash))
    print("Wrote results/generation_report.json")


if __name__ == "__main__":
    main()
