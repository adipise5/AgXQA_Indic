"""Step 1-2: collect native documents listed in sources/<language>_sources.csv.

Each per-language CSV is curated by hand (see README). For each row this script
either downloads `source_url` or copies a file you already saved (`local_path`),
stores it under data/native_indic_qa/raw_documents/<language>/, hashes it and
writes sources/source_manifest.csv (plus a copy in the data directory).

Usage:
    python phase_06_native_indic_qa/scripts/download_documents.py --dry-run
    python phase_06_native_indic_qa/scripts/download_documents.py --languages hi,mr
"""

from __future__ import annotations

import argparse
import mimetypes
import shutil
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests

from common import (DATA_DIR, LANGUAGES, MANIFEST_FIELDS, RAW_DOCS_DIR, REPO_ROOT, SOURCES_DIR,
                    read_csv, rel, selected_languages, sha256_file, write_csv)

SOURCE_FIELDS = [
    "document_id", "document_title", "language", "organization", "source_url",
    "publication_year", "region", "domain", "license", "language_origin",
    "local_path", "include", "notes",
]
# Generic client name: no personal details are sent to document hosts.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-dataset-builder/1.0)"}
EXT_BY_TYPE = {"application/pdf": ".pdf", "text/html": ".html", "application/xhtml+xml": ".html"}


def guess_extension(url: str, content_type: str, first_bytes: bytes) -> str:
    if first_bytes.startswith(b"%PDF"):
        return ".pdf"
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype in EXT_BY_TYPE:
        return EXT_BY_TYPE[ctype]
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in (".pdf", ".html", ".htm", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        return ".html" if suffix == ".htm" else suffix
    return mimetypes.guess_extension(ctype) or ".bin"


def download(url: str, dest_stem: Path, timeout: int) -> Path:
    with requests.get(url, headers=HEADERS, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        chunks = response.iter_content(chunk_size=1 << 16)
        first = next(chunks, b"")
        ext = guess_extension(url, response.headers.get("Content-Type", ""), first)
        dest = dest_stem.with_suffix(ext)
        tmp = dest.with_suffix(ext + ".part")
        with tmp.open("wb") as out:
            out.write(first)
            for chunk in chunks:
                out.write(chunk)
        tmp.replace(dest)
    return dest


def existing_file(dest_stem: Path) -> Path | None:
    matches = [p for p in dest_stem.parent.glob(dest_stem.name + ".*") if not p.name.endswith(".part")]
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", default="all", help="Comma-separated codes (hi,mr,bn,ta) or 'all'")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched, write nothing")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between downloads (be polite)")
    args = parser.parse_args()

    manifest_rows: dict[str, dict] = {}
    # Keep manifest rows for languages not processed in this run.
    for row in read_csv(SOURCES_DIR / "source_manifest.csv"):
        manifest_rows[row["document_id"]] = row

    hashes: dict[str, str] = {r["document_hash"]: r["document_id"] for r in manifest_rows.values()
                              if r.get("document_hash")}
    failures = []

    for lang in selected_languages(args.languages):
        cfg = LANGUAGES[lang]
        source_csv = SOURCES_DIR / f"{cfg['folder']}_sources.csv"
        rows = read_csv(source_csv)
        print(f"\n[{lang}] {len(rows)} rows in {rel(source_csv)}")
        out_dir = RAW_DOCS_DIR / cfg["folder"]
        out_dir.mkdir(parents=True, exist_ok=True)

        for index, row in enumerate(rows, start=1):
            if (row.get("include") or "yes").strip().lower() in ("no", "n", "false", "0"):
                continue
            doc_id = (row.get("document_id") or "").strip() or f"{lang.upper()}_DOC_{index:03d}"
            url = (row.get("source_url") or "").strip()
            local = (row.get("local_path") or "").strip()
            row_lang = (row.get("language") or lang).strip()
            if row_lang != lang:
                print(f"  ! {doc_id}: language column says {row_lang!r} but file is {source_csv.name}")
            dest_stem = out_dir / doc_id

            if args.dry_run:
                print(f"  {doc_id}: {'copy ' + local if local else url or '(no url or local_path)'}")
                continue

            try:
                current = existing_file(dest_stem)
                if current and not args.force:
                    path = current
                elif local:
                    src = Path(local)
                    src = src if src.is_absolute() else REPO_ROOT / src
                    if not src.exists():
                        raise FileNotFoundError(f"local_path not found: {src}")
                    path = dest_stem.with_suffix(src.suffix.lower() or ".bin")
                    if src.resolve() != path.resolve():
                        shutil.copy2(src, path)
                elif url:
                    path = download(url, dest_stem, args.timeout)
                    time.sleep(args.delay)
                else:
                    raise ValueError("row has neither source_url nor local_path")
            except Exception as error:  # keep going; report at the end
                failures.append({"document_id": doc_id, "error": f"{type(error).__name__}: {error}"})
                print(f"  x {doc_id}: {type(error).__name__}: {error}")
                continue

            digest = sha256_file(path)
            if digest in hashes and hashes[digest] != doc_id:
                print(f"  ! {doc_id}: identical file to {hashes[digest]} (same SHA-256) - consider removing one")
            hashes[digest] = doc_id
            previous = manifest_rows.get(doc_id, {})
            manifest_rows[doc_id] = {
                "document_id": doc_id,
                "document_title": row.get("document_title", ""),
                "language": lang,
                "organization": row.get("organization", ""),
                "source_url": url,
                "publication_year": row.get("publication_year", ""),
                "region": row.get("region", ""),
                "domain": row.get("domain", ""),
                "file_path": rel(path),
                "license": row.get("license", "") or "unverified",
                "document_hash": digest,
                "language_origin": row.get("language_origin", "") or "unknown",
                "download_date": previous.get("download_date") if previous.get("document_hash") == digest
                else date.today().isoformat(),
            }
            print(f"  ok {doc_id} -> {rel(path)}")

    if args.dry_run:
        return
    rows_out = sorted(manifest_rows.values(), key=lambda r: (r["language"], r["document_id"]))
    write_csv(SOURCES_DIR / "source_manifest.csv", rows_out, MANIFEST_FIELDS)
    write_csv(DATA_DIR / "source_manifest.csv", rows_out, MANIFEST_FIELDS)
    print(f"\nManifest: {len(rows_out)} documents -> {rel(SOURCES_DIR / 'source_manifest.csv')}")
    unverified = [r["document_id"] for r in rows_out if r["license"] in ("", "unverified", "unknown")]
    if unverified:
        print(f"License not yet verified for {len(unverified)} document(s): {', '.join(unverified[:10])}"
              f"{' ...' if len(unverified) > 10 else ''}")
    if failures:
        print(f"{len(failures)} failure(s):")
        for f in failures:
            print(f"  {f['document_id']}: {f['error']}")


if __name__ == "__main__":
    main()
