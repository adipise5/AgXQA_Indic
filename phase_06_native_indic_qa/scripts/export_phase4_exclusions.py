"""Write the list of documents Phase 4 (Indic-AEC) must NOT ingest.

Reads every sources/<language>_sources.csv and writes
sources/phase4_exclusion_list.csv for Aditya Pise. Columns follow the Phase 4
sources.csv in the team repo (document_id, language, source_url, title,
publisher, ...) plus:

- normalized_url: canonical key (see common.normalize_url). Phase 4 should
  normalise its own URLs with the same function before comparing, because
  one Vikaspedia article has many URL spellings.
- status: reserved_phase6 (include=yes, will become benchmark contexts) or
  candidate_phase6 (include=no today, e.g. OCR backlog; still keep out of
  Phase 4 so it stays available as held-out data).

A URL list cannot catch the same text republished elsewhere, so once Phase 4
has a corpus, run check_data_leakage.py --phase4-corpus as well.

Usage:
    python phase_06_native_indic_qa/scripts/export_phase4_exclusions.py
"""

from __future__ import annotations

from datetime import date

from common import LANGUAGES, SOURCES_DIR, normalize_url, read_csv, rel, write_csv

FIELDS = ["document_id", "language", "source_url", "normalized_url", "title", "publisher", "status",
          "reserved_on"]


def main() -> None:
    rows, seen = [], {}
    for lang, cfg in LANGUAGES.items():
        for row in read_csv(SOURCES_DIR / f"{cfg['folder']}_sources.csv"):
            url = (row.get("source_url") or "").strip()
            key = normalize_url(url)
            if key in seen:
                print(f"! {row['document_id']} has the same normalized URL as {seen[key]}")
            seen[key] = row["document_id"]
            include = (row.get("include") or "yes").strip().lower() not in ("no", "n", "false", "0")
            rows.append({"document_id": row["document_id"], "language": lang, "source_url": url,
                         "normalized_url": key, "title": row.get("document_title", ""),
                         "publisher": row.get("organization", ""),
                         "status": "reserved_phase6" if include else "candidate_phase6",
                         "reserved_on": date.today().isoformat()})
    out = SOURCES_DIR / "phase4_exclusion_list.csv"
    write_csv(out, rows, FIELDS)
    reserved = sum(r["status"] == "reserved_phase6" for r in rows)
    print(f"{len(rows)} documents ({reserved} reserved, {len(rows) - reserved} candidates) -> {rel(out)}")


if __name__ == "__main__":
    main()
