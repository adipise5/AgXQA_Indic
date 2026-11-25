"""Verify a reviewed Indic-AEC release and write a text-free submission audit."""

import argparse
from collections import Counter, defaultdict
import json
from importlib.metadata import version
from pathlib import Path
import platform
import shutil
from tempfile import TemporaryDirectory

from jsonschema import Draft202012Validator
from tokenizers import Tokenizer

from . import pipeline

THRESHOLDS = (0.30, 0.40, 0.50)


def character_grams(text):
    # Compare across OCR word splits without dropping Indic combining marks.
    joined = ''.join(pipeline.common.norm_for_match(text).split())
    return {joined[offset:offset + 10] for offset in range(len(joined) - 9)}


def scan(documents, references, features):
    comparable = [row for row in references if features(row['context'])]
    if not comparable or any(not features(row['text']) for row in documents):
        raise ValueError('Insufficient text for comparison')
    findings = pipeline.audit_documents(documents, comparable, THRESHOLDS[0], features=features)
    return {
        'thresholds': {str(value): sum(max(hit['pair_containment'], hit['corpus_containment']) >= value
                                       for hit in findings) for value in THRESHOLDS},
        'references_compared': len(comparable),
        'short_reference_ids': sorted({row['context_id'] for row in references}
                                      - {row['context_id'] for row in comparable}),
        'findings': findings,
    }


def detector_controls(heldout_dir, languages):
    result = {}
    for language in languages:
        source = next(row for row in pipeline.read_lines(heldout_dir / f'{language}_contexts.jsonl')
                      if len(pipeline.shingles(row['context'])) >= 25)
        original = pipeline.common.norm_for_match(source['context'])
        compact = ''.join(original.split())
        variants = {'exact': original, 'spaces_removed': compact,
                    'spaces_and_periodic_characters_removed': ''.join(char for i, char in enumerate(compact, 1)
                                                                      if i % 29)}
        reference = [{'context_id': source['context_id'], 'context': original}]
        result[language] = {name: {
            'word_5gram_flag': bool(pipeline.audit_documents([{'document_id': name, 'text': text}], reference)),
            'character_10gram_flag': bool(pipeline.audit_documents([{'document_id': name, 'text': text}], reference,
                                                                   features=character_grams)),
        } for name, text in variants.items()}
    return result


def audit_english_source(source_dir, documents):
    phase1 = pipeline.ROOT / 'phase_01_original_baseline/results/dataset_statistics.json'
    upstream = json.loads(phase1.read_text())
    passages = {}
    for split in ('validation', 'test'):
        spec = upstream['audit']['splits'][split]
        path = source_dir / Path(spec['path']).name
        if pipeline.common.sha256_file(path) != spec['sha256']:
            raise ValueError(f'Official English {split} does not match the Phase 1 source hash')
        for row in pipeline.read_lines(path):
            passages.setdefault(row['context'], {'context_id': row['id'], 'context': row['context']})
    return {'revision': upstream['revision'], 'scope': 'official English validation and test, literal overlap only',
            'word_5gram': scan(documents, list(passages.values()), pipeline.shingles)}


def verify_release(config, release):
    stats = json.loads((release / 'statistics.json').read_text())
    files = {name: pipeline.common.sha256_file(release / name) for name in ('corpus.jsonl', 'train.jsonl', 'validation.jsonl')}
    if stats['file_sha256'] != files:
        raise ValueError('Release file checksums disagree with statistics')
    documents = pipeline.read_lines(release / 'corpus.jsonl')
    pipeline.validate_sources(documents)
    computed_version = pipeline.digest(json.dumps(documents, ensure_ascii=False, sort_keys=True))[:16]
    if release.name != stats['version'] or stats['version'] != computed_version:
        raise ValueError('Release version does not match its records')
    reviews = {row['document_id']: row for row in pipeline.read_rows(pipeline.resolve(config['quality_reviews']))}
    registered = pipeline.government_book_registry(config)
    validator = Draft202012Validator(json.loads((pipeline.ROOT / 'schemas/corpus.schema.json').read_text()))
    cache = pipeline.resolve(config['work_dir']) / 'tokenizer_cache'
    spec = stats['tokenizer']
    from huggingface_hub import hf_hub_download
    tokenizer_path = Path(hf_hub_download(spec['id'], 'tokenizer.json', revision=spec['revision'],
                                          cache_dir=cache, local_files_only=True))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    counts = defaultdict(Counter)
    ledger = []
    guard = pipeline.Exclusions()
    for row in documents:
        validator.validate(row)
        document_id = row['document_id']
        upstream = registered.get(document_id, {})
        fields = ('source_url', 'license', 'permission_reference', 'permission_status', 'language_origin', 'document_family_id')
        if (upstream.get('permission_status') != 'approved' or upstream.get('language_origin') != 'native_verified'
                or any(row[field] != upstream.get(field) for field in fields)):
            raise ValueError(f'Current source registry disagrees with approved record {document_id}')
        if (guard.check(row['source_url'], row['sha256']) or guard.check(row.get('resolved_url', row['source_url']))
                or guard.titles_matching(row['title'])):
            raise ValueError(f'Phase 6 source reservation matched {document_id}')
        review = reviews.get(document_id, {})
        if (review.get('quality_status') != 'accepted' or not review.get('reviewer')
                or review['reviewer'] != row['quality_reviewer'] or row['quality_status'] != 'accepted'
                or review.get('clean_text_sha256') != pipeline.digest(row['text'])
                or review['clean_text_sha256'] != row['clean_text_sha256']):
            raise ValueError(f'No matching accepted text review for {document_id}')
        evidence = {row['file_path']: row['sha256'], **row.get('source_evidence_sha256', {})}
        if any(pipeline.common.sha256_file(pipeline.resolve(path)) != sha for path, sha in evidence.items()):
            raise ValueError(f'Source capture has changed for {document_id}')
        tokens = len(tokenizer.encode(row['text'], add_special_tokens=False).ids)
        if tokens != row['token_count'] or row['word_count'] != len(row['text'].split()):
            raise ValueError(f'Length check failed for {document_id}')
        counts[row['split'], row['language']].update(documents=1, words=row['word_count'], tokens=tokens)
        ledger.append({key: row[key] for key in ('document_id', 'title', 'language', 'publisher', 'domain', 'split',
                                               *fields, 'word_count', 'token_count', 'clean_text_sha256', 'quality_reviewer')}
                      | {'capture_sha256': row['sha256'], 'origin_reference': row.get('origin_reference')})
    splits = {name: pipeline.read_lines(release / f'{name}.jsonl') for name in ('train', 'validation')}
    for name, rows in splits.items():
        if rows != [row for row in documents if row['split'] == name] or {row['language'] for row in rows} != set(config['languages']):
            raise ValueError(f'Release {name} rows or language coverage disagree with the corpus')
    if sum(map(len, splits.values())) != len(documents):
        raise ValueError('Unexpected corpus split')
    for field in ('document_family_id', 'source_url', 'clean_text_sha256'):
        if {row[field] for row in splits['train']} & {row[field] for row in splits['validation']}:
            raise ValueError(f'Cross-split overlap in {field}')
    paragraphs = defaultdict(set)
    for row in documents:
        for paragraph in row['paragraphs']:
            normalized = pipeline.common.norm_for_match(paragraph['text'])
            if len(normalized.split()) >= 20:
                paragraphs[normalized].add(row['document_family_id'])
    return documents, {
        'version': stats['version'], 'documents': len(documents), 'approved_reviews_verified': len(ledger),
        'families': len({row['document_family_id'] for row in documents}),
        'publishers': len({row['publisher'] for row in documents}),
        'cross_family_exact_paragraphs_20_words_or_more': sum(len(group) > 1 for group in paragraphs.values()),
        'file_sha256': files | {'statistics.json': pipeline.common.sha256_file(release / 'statistics.json')},
        'tokenizer': spec | {'file_sha256': pipeline.common.sha256_file(tokenizer_path)},
        'coverage': [{'split': split, 'language': language, **counts[split, language]}
                     for split in splits for language in config['languages']],
        'sources': ledger,
    }


def verify_rebuild(config, release, stage):
    work = pipeline.resolve(config['work_dir'])
    shutil.copyfile(work / 'manifest.jsonl', stage / 'manifest.jsonl')
    shutil.copytree(work / 'tokenizer_cache', stage / 'tokenizer_cache', symlinks=True)
    rebuild = config | {'work_dir': str(stage), 'report_dir': str(stage / 'reports')}
    pipeline.extract(rebuild)
    stats = pipeline.build(rebuild)
    produced = stage / 'releases' / stats['version']
    checks = {name: (produced / name).read_bytes() == (release / name).read_bytes()
              for name in ('corpus.jsonl', 'train.jsonl', 'validation.jsonl', 'statistics.json')}
    if stats['version'] != release.name or not all(checks.values()):
        raise ValueError(f'Capture rebuild differs from the release: {checks}')
    return {'version': stats['version'], 'identical_bytes': checks, 'inputs': 'saved source captures and pinned tokenizer'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--english-source', type=Path)
    parser.add_argument('--rebuild', action='store_true', help='Re-extract saved captures and compare release bytes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    documents, report = verify_release(config, args.release)
    report['build_inputs'] = {
        'seed': config['seed'],
        'source_registry_sha256': pipeline.common.sha256_file(pipeline.resolve(config['sources'])),
        'quality_reviews_sha256': pipeline.common.sha256_file(pipeline.resolve(config['quality_reviews'])),
        'capture_manifest_sha256': pipeline.common.sha256_file(pipeline.resolve(config['work_dir']) / 'manifest.jsonl'),
    }
    heldout_dir = pipeline.resolve(config['phase6_contexts'])
    with TemporaryDirectory() as temp:
        metadata = Path(temp) / 'phase6_reference.json'
        heldout = pipeline.load_heldout(heldout_dir, metadata)
        report['phase6'] = json.loads(metadata.read_text()) | {
            'baseline': config['phase6_baseline'],
            'word_5gram': scan(documents, heldout, pipeline.shingles),
            'character_10gram': scan(documents, heldout, character_grams),
            'synthetic_positive_controls_at_0_50': detector_controls(heldout_dir, config['languages']),
        }
        if args.rebuild:
            report['rebuild'] = verify_rebuild(config, args.release, Path(temp))
    report['phase2'] = {'retained_multilingual_evaluation': 'unavailable_for_this_audit'}
    if args.english_source:
        report['phase2']['preliminary_english'] = audit_english_source(args.english_source, documents)
    report['environment'] = {'python': platform.python_version(), 'platform': platform.platform(),
                             'dependencies': {name: version(name) for name in (
                                 'requests', 'regex', 'PyYAML', 'rapidfuzz', 'beautifulsoup4', 'trafilatura',
                                 'pymupdf', 'jsonschema', 'tokenizers', 'huggingface-hub', 'pytesseract', 'pillow')}}
    pipeline.common.write_json(args.output, report)


if __name__ == '__main__':
    main()
