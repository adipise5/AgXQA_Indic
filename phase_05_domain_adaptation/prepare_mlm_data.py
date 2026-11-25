"""Prepare document-disjoint token sequences for the Phase 5 local MLM pilot."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare(release, destination, languages, max_length, cache_dir):
    stats = json.loads((release / 'statistics.json').read_text(encoding='utf-8'))
    tokenizer_spec = stats['tokenizer']
    if tokenizer_spec['id'] != 'FacebookAI/xlm-roberta-base':
        raise ValueError('The Phase 5 pilot requires the pinned Phase 4 XLM-R base tokenizer')
    splits = ('train', 'validation')
    for split in splits:
        filename = f'{split}.jsonl'
        if sha256(release / filename) != stats['file_sha256'][filename]:
            raise ValueError(f'Phase 4 checksum mismatch for {filename}')
    if destination.exists():
        raise ValueError(f'Output already exists: {destination}')

    tokenizer_path = Path(hf_hub_download(
        tokenizer_spec['id'], 'tokenizer.json', revision=tokenizer_spec['revision'], cache_dir=cache_dir,
    ))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=max_length, stride=0)
    document_ids = set()
    family_splits = {}
    manifest = {
        'status': 'local_mlm_input_prepared',
        'phase4_version': stats['version'],
        'phase6_baseline': stats['phase6_baseline'],
        'tokenizer': {**tokenizer_spec, 'sha256': sha256(tokenizer_path), 'library_version': version('tokenizers')},
        'max_sequence_length': max_length,
        'splits': {},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as staging:
        output = Path(staging) / 'dataset'
        output.mkdir()
        for split in splits:
            filename = f'{split}.jsonl'
            counts = {'documents': 0, 'sequences': 0, 'content_tokens': 0}
            seen_languages = set()
            with (release / filename).open(encoding='utf-8') as source, \
                 (output / filename).open('w', encoding='utf-8') as target:
                for line in source:
                    row = json.loads(line)
                    ident, family = row['document_id'], row['document_family_id']
                    digest = hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
                    if (row['split'] != split or row['quality_status'] != 'accepted'
                            or not row['quality_reviewer'] or digest != row['clean_text_sha256']):
                        raise ValueError(f'Invalid split or reviewed text hash for {ident}')
                    if ident in document_ids or not family or family_splits.setdefault(family, split) != split:
                        raise ValueError(f'Duplicate document or family shared across splits: {ident}')
                    document_ids.add(ident)
                    seen_languages.add(row['language'])
                    counts['documents'] += 1
                    encoded = tokenizer.encode(row['text'])
                    document_tokens = 0
                    for index, chunk in enumerate([encoded, *encoded.overflowing]):
                        mask = chunk.special_tokens_mask
                        content_count = len(mask) - sum(mask)
                        if not content_count or len(mask) > max_length:
                            raise ValueError(f'Invalid token sequence for {ident}')
                        record = {
                            'document_id': ident, 'document_family_id': family,
                            'language': row['language'], 'split': split, 'sequence_index': index,
                            'clean_text_sha256': digest, 'input_ids': chunk.ids,
                            'attention_mask': chunk.attention_mask, 'special_tokens_mask': mask,
                        }
                        target.write(json.dumps(record, ensure_ascii=False) + '\n')
                        counts['sequences'] += 1
                        document_tokens += content_count
                    if document_tokens != row['token_count']:
                        raise ValueError(f'Token coverage differs from Phase 4 for {ident}')
                    counts['content_tokens'] += document_tokens
            if seen_languages != languages:
                raise ValueError(f'Unexpected language coverage in {split}: {sorted(seen_languages)}')
            manifest['splits'][split] = {
                **counts, 'languages': sorted(seen_languages),
                'source_sha256': stats['file_sha256'][filename], 'sha256': sha256(output / filename),
            }
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        output.rename(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('phase_05_domain_adaptation/config.json'))
    parser.add_argument('--corpus-release', type=Path, required=True)
    parser.add_argument('--tokenizer-cache', type=Path)
    parser.add_argument('--max-sequence-length', type=int, default=256, choices=range(3, 513), metavar='[3-512]')
    args = parser.parse_args()
    try:
        cfg = json.loads(args.config.read_text(encoding='utf-8'))
        release = args.corpus_release.resolve()
        destination = Path(cfg['output_dir']) / 'mlm_inputs' / f'{release.name}-len{args.max_sequence_length}'
        manifest = prepare(release, destination, set(cfg['mlm_languages']),
                           args.max_sequence_length, args.tokenizer_cache)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'{exc}\n')
    print(json.dumps({'output_dir': str(destination), **manifest}, indent=2))

if __name__ == '__main__':
    main()
