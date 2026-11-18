"""Phase 4 document acquisition and fail-closed corpus construction.

Phase 6 modules are imported for pure helpers only; their CLI entry points are
never executed, so benchmark manifests and datasets remain untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
P6 = ROOT / 'phase_06_native_indic_qa'
sys.path.insert(0, str(P6 / 'scripts'))
common = importlib.import_module('common')
extractor = importlib.import_module('extract_text')
cleaner = importlib.import_module('clean_text')
from rapidfuzz import fuzz


def read_rows(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Missing input: {path}')
    return common.read_csv(path)


def read_lines(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Missing input: {path}')
    return common.read_jsonl(path)


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def shingles(text):
    words = common.norm_for_match(text).split()
    return {tuple(words[i:i+5]) for i in range(len(words)-4)}


class Exclusions:
    def __init__(self):
        reserved = read_rows(P6 / 'sources/phase4_exclusion_list.csv')
        manifest = read_rows(P6 / 'sources/source_manifest.csv')
        if len(reserved) != 84 or len(manifest) != 70:
            raise ValueError('Phase 6 registry changed: review and update the pinned baseline')
        self.urls = {common.normalize_url(r['source_url']) for r in reserved}
        self.hashes = {r['document_hash'] for r in manifest}
        self.titles = [(r['document_id'], common.norm_for_match(r['title'])) for r in reserved]

    def check(self, url, raw_hash=None):
        if common.normalize_url(url) in self.urls:
            return 'phase6_reserved_url'
        host = (urlparse(url).hostname or '').lower()
        if host == 'vikaspedia.in' or host.endswith('.vikaspedia.in'):
            return 'vikaspedia_excluded_for_v1'
        if raw_hash in self.hashes:
            return 'phase6_raw_hash'
        return None

    def titles_matching(self, title):
        key = common.norm_for_match(title)
        return [doc for doc, other in self.titles if key and fuzz.token_sort_ratio(key, other) >= 90]


def validate_sources(rows):
    ids = set()
    for r in rows:
        ident = r.get('document_id', '')
        if not re.fullmatch(r'AEC_(HI|MR|BN|TA)_[A-Za-z0-9_-]+', ident) or ident in ids:
            raise ValueError(f'Invalid or repeated document_id: {ident}')
        ids.add(ident)
        if r.get('language') not in common.LANGUAGES or not ident.startswith('AEC_' + r['language'].upper() + '_'):
            raise ValueError(f'Invalid language: {ident}')
        if urlparse(r.get('source_url', '')).scheme not in ('https', 'http'):
            raise ValueError(f'Invalid source URL: {ident}')


def acquire(cfg, dry_run=False):
    import requests
    rows = read_rows(resolve(cfg['sources']))
    validate_sources(rows)
    guard = Exclusions()
    out = resolve(cfg['work_dir'])
    accepted, report = [], []
    for row in rows:
        reason = guard.check(row['source_url'])
        if row.get('permission_status') != 'approved' or not row.get('permission_reference'):
            reason = reason or 'permission_pending'
        if row.get('language_origin') != 'native_verified':
            reason = reason or 'native_origin_unverified'
        if not row.get('license') or not row.get('document_family_id'):
            reason = reason or 'missing_license_or_family'
        if guard.titles_matching(row.get('title', '')):
            reason = reason or 'phase6_similar_title_review_required'
        result = {'document_id': row['document_id'], 'status': reason or ('ready' if dry_run else 'pending')}
        if not reason and not dry_run:
            try:
                # Resolve redirects manually: check each destination before fetching it.
                from urllib.parse import urljoin
                url = row['source_url']
                for _ in range(6):
                    if guard.check(url):
                        raise ValueError('Redirect points to an excluded source')
                    with requests.get(url, timeout=45, allow_redirects=False, stream=True) as response:
                        if response.is_redirect:
                            url = urljoin(url, response.headers['Location'])
                            continue
                        response.raise_for_status()
                        raw = bytearray()
                        for block in response.iter_content(65536):
                            raw.extend(block)
                            if len(raw) > cfg['max_download_bytes']:
                                raise ValueError('Document exceeds download size limit')
                        break
                else:
                    raise ValueError('Too many redirects')
                raw_hash = hashlib.sha256(raw).hexdigest()
                if guard.check(url, raw_hash):
                    raise ValueError('Reserved document hash or URL')
                suffix = '.pdf' if raw.startswith(b'%PDF') else '.html'
                path = out / 'raw' / (row['document_id'] + suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                accepted.append({**row, 'resolved_url': url, 'sha256': raw_hash,
                                 'accessed_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                 'file_path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)})
                result['status'] = 'downloaded'
            except Exception as exc:
                result.update(status='failed', error=str(exc))
        report.append(result)
    if not dry_run:
        common.write_jsonl(out / 'manifest.jsonl', accepted)
    common.write_json(resolve(cfg['report_dir']) / 'acquisition_report.json', report)
    return Counter(r['status'] for r in report)


def extract(cfg, ocr=False):
    out = resolve(cfg['work_dir'])
    manifest = read_lines(out / 'manifest.jsonl')
    if not manifest:
        raise ValueError('No approved acquired documents')
    records, report = [], []
    settings = common.load_config()['extraction']
    cleaning = common.load_config()['cleaning']
    for row in manifest:
        path, lang = resolve(row['file_path']), row['language']
        if common.sha256_file(path) != row['sha256']:
            raise ValueError(f'Raw input changed: {path}')
        try:
            pages = (extractor.extract_pdf(path, lang, settings, ocr) if path.suffix == '.pdf'
                     else extractor.extract_html(path, lang))
        except (OSError, RuntimeError, ValueError) as exc:
            report.append({'document_id': row['document_id'], 'words': 0, 'error': str(exc)})
            continue
        repeated = cleaner.find_repeated_lines(pages, cleaning)
        paras, rejected = [], Counter()
        for page in pages:
            if page.get('needs_ocr') or not common.language_ok(page['text'], lang, 0.35)[0]:
                rejected['unusable_page'] += 1
                continue
            is_html = page['method'] in ('trafilatura', 'beautifulsoup', 'embedded_article_html')
            for para in cleaner.page_paragraphs(common.nfc(page['text']), page['page_number'], repeated, is_html):
                text = para['text']
                ok, reason = common.language_ok(text, lang, 0.6)
                if not ok or cleaner.is_corrupted(text) or cleaner.CREDIT_RE.match(text):
                    rejected[reason or 'corrupt_or_credit'] += 1
                    continue
                if not para['is_heading'] and len(text.split()) < 3:
                    rejected['short_fragment'] += 1
                    continue
                paras.append({**para, 'method': page['method']})
        text = '\n'.join(p['text'] for p in paras)
        record = {**row, 'text': text, 'paragraphs': paras, 'clean_text_sha256': digest(text),
                  'word_count': len(text.split())}
        records.append(record)
        report.append({'document_id': row['document_id'], 'words': record['word_count'],
                       'rejected': dict(rejected), 'requires_human_quality_review': True})
    common.write_jsonl(out / 'documents.jsonl', records)
    common.write_json(resolve(cfg['report_dir']) / 'quality_report.json', report)
    return {'documents': len(records)}


def load_heldout(directory, report_path=None):
    rows = []
    for lang, expected in {'hi': 393, 'mr': 203, 'bn': 146, 'ta': 85}.items():
        contexts = read_lines(Path(directory) / f'{lang}_contexts.jsonl')
        if len(contexts) != expected:
            raise ValueError(f'Incomplete Phase 6 contexts for {lang}: {len(contexts)}/{expected}')
        rows.extend(contexts)
    if len({r['context_id'] for r in rows}) != 827 or len({r['document_id'] for r in rows}) != 70:
        raise ValueError('Invalid held-out IDs or document coverage')
    by_id = {r['context_id']: r for r in rows}
    verified = 0
    for folder in ('hindi', 'marathi', 'bengali', 'tamil'):
        for release in read_lines(ROOT / 'data/native_indic_qa/final' / f'{folder}.jsonl'):
            ctx = by_id.get(release['context_id'])
            if (ctx is None or ctx['document_id'] != release['source_document_id']
                    or digest(ctx['context']) != release['context_sha256']):
                raise ValueError('Phase 6 context hash mismatch')
            verified += 1
    if verified != 2351:
        raise ValueError(f'Expected 2351 Phase 6 release rows, found {verified}')
    # Re-extract the locally reconstructed snapshots so the full text and
    # reported hashes always refer to the same files.
    manifest = read_rows(P6 / 'sources/source_manifest.csv')
    if {r['document_id'] for r in rows} != {r['document_id'] for r in manifest}:
        raise ValueError('Held-out document IDs differ from the Phase 6 manifest')
    regenerated_manifest = {r['document_id']: r for r in read_rows(Path(directory).parent / 'source_manifest.csv')}
    raw_matches = 0
    changed = []
    for source in manifest:
        ident = source['document_id']
        snapshot = regenerated_manifest.get(ident)
        if not snapshot or common.normalize_url(snapshot['source_url']) != common.normalize_url(source['source_url']):
            raise ValueError(f'Missing or unrecognized Phase 6 snapshot: {ident}')
        raw = resolve(snapshot['file_path'])
        folder = common.LANGUAGES[source['language']]['folder']
        expected_raw = Path(directory).parent / 'raw_documents' / folder / (ident + '.html')
        if raw.resolve() != expected_raw.resolve() or common.sha256_file(raw) != snapshot['document_hash']:
            raise ValueError(f'Phase 6 snapshot path or hash mismatch: {ident}')
        if snapshot['document_hash'] == source['document_hash']:
            raw_matches += 1
        else:
            changed.append(ident)
        text = '\n'.join(page['text'] for page in extractor.extract_html(raw, source['language']))
        if not text.strip():
            raise ValueError(f'Empty full-document comparison: {ident}')
        # Windows catch copied subsections inside larger documents.
        words = common.norm_for_match(text).split()
        for start in range(0, len(words), 100):
            window = words[start:start + 200]
            if len(window) >= 20:
                rows.append({'context_id': f'{ident}_FULL_{start}', 'context': ' '.join(window)})
    if report_path is not None:
        common.write_json(report_path, {
            'phase6_documents': len(manifest), 'raw_byte_matches': raw_matches,
            'changed_raw_snapshots': changed, 'verified_qa_records': verified,
            'qa_contexts': len(by_id), 'full_document_windows': len(rows) - len(by_id),
            'full_document_reference': 'reconstructed public snapshots',
        })
    return rows


def audit_documents(documents, heldout, threshold=0.5):
    """Return candidate IDs to reject, including distributed corpus overlap."""
    held = [(row['context_id'], shingles(row['context'])) for row in heldout]
    index = defaultdict(list)
    for reference_id, reference in held:
        if not reference:
            raise ValueError(f'Held-out passage has no 5-grams: {reference_id}')
        for gram in reference:
            index[gram].append(reference_id)
    sizes = {reference_id: len(reference) for reference_id, reference in held}
    corpus_hits = defaultdict(set)
    pairs = defaultdict(list)
    for row in documents:
        candidate = shingles(row['text'])
        hits = Counter()
        for gram in candidate:
            for reference_id in index.get(gram, ()):
                corpus_hits[reference_id].add(gram)
                hits[reference_id] += 1
        for reference_id, count in hits.items():
            direct = count / min(sizes[reference_id], len(candidate))
            pairs[reference_id].append((row['document_id'], direct))
    findings = []
    for reference_id, matches in pairs.items():
        share = len(corpus_hits[reference_id]) / sizes[reference_id]
        for document_id, direct in matches:
            if direct >= threshold or share >= threshold:
                findings.append({'document_id': document_id, 'context_id': reference_id,
                                 'pair_containment': direct, 'corpus_containment': share})
    findings.sort(key=lambda row: (row['document_id'], row['context_id']))
    return findings


def audit(cfg, documents):
    report_dir = resolve(cfg['report_dir'])
    heldout = load_heldout(resolve(cfg['phase6_contexts']), report_dir / 'phase6_reference.json')
    comparable = [row for row in documents if shingles(row['text'])]
    if not comparable:
        raise ValueError('No extracted documents contain enough text for overlap checking')
    findings = audit_documents(comparable, heldout)
    common.write_json(report_dir / 'overlap_report.json', {
        'candidate_documents': len(documents), 'text_documents_checked': len(comparable),
        'unscannable_document_ids': [row['document_id'] for row in documents if not shingles(row['text'])],
        'reference_windows': len(heldout),
        'documents_with_hits': len({row['document_id'] for row in findings}),
        'findings': findings,
    })
    return findings


def load_candidates(cfg):
    docs = read_lines(resolve(cfg['work_dir']) / 'documents.jsonl')
    if not docs:
        raise ValueError('Empty corpus input')
    validate_sources(docs)
    return docs


def build(cfg):
    import jsonschema
    out = resolve(cfg['work_dir'])
    docs = load_candidates(cfg)
    guard = Exclusions()
    findings = audit(cfg, docs)
    blocked = {r['document_id'] for r in findings}
    review_path = cfg.get('quality_reviews')
    review_rows = read_rows(resolve(review_path)) if review_path else []
    reviews = {row['document_id']: row for row in review_rows}
    if len(reviews) != len(review_rows):
        raise ValueError('Duplicate quality review ID')
    exclusions, kept, seen = [], [], set()
    for row in sorted(docs, key=lambda r: r['document_id']):
        reason = guard.check(row['source_url'], row.get('sha256')) or guard.check(row.get('resolved_url', row['source_url']))
        if row['document_id'] in blocked:
            reason = reason or 'phase6_passage_overlap'
        if guard.titles_matching(row['title']):
            reason = reason or 'phase6_similar_title_review_required'
        if row.get('permission_status') != 'approved' or not row.get('permission_reference') or row.get('language_origin') != 'native_verified':
            reason = reason or 'permission_or_origin_pending'
        review = reviews.get(row['document_id'], {}) if review_path else row
        accepted = review.get('quality_status') == 'accepted'
        if review_path:
            accepted = accepted and review.get('reviewer') and review.get('clean_text_sha256') == digest(row['text'])
        if not accepted:
            reason = reason or 'quality_review_pending'
        if len(row['text'].split()) < cfg['min_document_words']:
            reason = reason or 'too_short'
        key = common.norm_for_match(row['text'])
        own_shingles = shingles(row['text'])
        def duplicate(other):
            other_shingles = shingles(other['text'])
            overlap = len(own_shingles & other_shingles)
            containment = overlap / min(len(own_shingles), len(other_shingles)) if own_shingles and other_shingles else 0
            return containment >= 0.5 or fuzz.ratio(key, common.norm_for_match(other['text'])) >= 92
        if key in seen or any(duplicate(k) for k in kept):
            reason = reason or 'internal_duplicate'
        if reason:
            exclusions.append({'document_id': row['document_id'], 'reason': reason})
        else:
            seen.add(key)
            kept.append({**row, 'quality_status': 'accepted',
                         'quality_reviewer': review.get('reviewer', '')})
    common.write_json(resolve(cfg['report_dir']) / 'leakage_report.json',
                      {'candidate_documents': len(docs),
                       'findings': findings, 'exclusions': exclusions})
    if not kept:
        raise ValueError('No accepted documents; inspect leakage report')
    # Families are global across languages, preventing translated editions crossing splits.
    families = sorted({r['document_family_id'] for r in kept})
    if len(families) < 2:
        raise ValueError('At least two document families required')
    random.Random(cfg['seed']).shuffle(families)
    by_language = {
        lang: {row['document_family_id'] for row in kept if row['language'] == lang}
        for lang in cfg['languages']
    }
    validation = set()
    for lang, group in by_language.items():
        if len(group) < 2:
            raise ValueError(f'{lang} needs at least two accepted document families')
        target = max(1, round(len(group) * cfg['validation_fraction']))
        for family in families:
            if len(group & validation) >= target:
                break
            if family in group and all(members - (validation | {family}) for members in by_language.values()):
                validation.add(family)
    schema = json.loads((ROOT / 'schemas/corpus.schema.json').read_text())
    for row in kept:
        row['split'] = 'validation' if row['document_family_id'] in validation else 'train'
        row['clean_text_sha256'] = digest(row['text'])
        jsonschema.validate(row, schema)
    for lang in cfg['languages']:
        if {r['split'] for r in kept if r['language'] == lang} != {'train', 'validation'}:
            raise ValueError(f'{lang} needs accepted families in both splits; expand/rebalance collection')
    tokenizer_cfg = cfg.get('tokenizer')
    if tokenizer_cfg:
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download

        tokenizer_path = hf_hub_download(tokenizer_cfg['id'], 'tokenizer.json',
                                         revision=tokenizer_cfg['revision'],
                                         cache_dir=str(out / 'tokenizer_cache'))
        tokenizer = Tokenizer.from_file(tokenizer_path)
        for row in kept:
            row['token_count'] = len(tokenizer.encode(row['text'], add_special_tokens=False).ids)
    # Fresh version directory prevents a failed run from masquerading as a new release.
    version = digest(json.dumps(kept, ensure_ascii=False, sort_keys=True))[:16]
    release = out / 'releases' / version
    for split in ('train', 'validation'):
        common.write_jsonl(release / f'{split}.jsonl', (r for r in kept if r['split'] == split))
    common.write_jsonl(release / 'corpus.jsonl', kept)
    breakdown = defaultdict(lambda: {'documents': 0, 'words': 0})
    for row in kept:
        for dimension in ('language', 'split', 'publisher', 'domain'):
            bucket = breakdown[(dimension, row.get(dimension) or 'unknown')]
            bucket['documents'] += 1
            bucket['words'] += len(row['text'].split())
            if tokenizer_cfg:
                bucket['tokens'] = bucket.get('tokens', 0) + row['token_count']
    stats = {'breakdown': [{dimension: label, **values, 'dimension': dimension}
                           for (dimension, label), values in sorted(breakdown.items())],
             'version': version, 'documents': len(kept), 'languages': dict(Counter(r['language'] for r in kept)),
             'words': sum(len(r['text'].split()) for r in kept), 'exclusions': len(exclusions),
             'tokenizer': tokenizer_cfg,
             'seed': cfg['seed'], 'phase6_baseline': cfg['phase6_baseline'],
             'file_sha256': {name: common.sha256_file(release / name)
                             for name in ('corpus.jsonl', 'train.jsonl', 'validation.jsonl')}}
    common.write_json(release / 'statistics.json', stats)
    return stats


def main(stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if stage is None:
        parser.add_argument('stage', choices=('download', 'extract', 'audit', 'build'))
    parser.add_argument('--config', default=str(ROOT / 'phase_04_indic_aec_corpus/config.json'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--ocr', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    try:
        selected = stage or args.stage
        if selected == 'download':
            result = acquire(cfg, args.dry_run)
        elif selected == 'extract':
            result = extract(cfg, args.ocr)
        elif selected == 'audit':
            result = {'overlap_findings': len(audit(cfg, load_candidates(cfg)))}
        else:
            result = build(cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
