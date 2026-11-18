"""Rebuild and verify the held-out reference in an isolated work directory."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from pipeline import P6, ROOT, load_heldout, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='outputs/phase6_audit')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    directory = resolve(args.directory).resolve()
    if not directory.is_relative_to(ROOT / 'outputs'):
        parser.error('Choose a directory inside the repository outputs folder')
    if not args.verify_only:
        sources = directory / 'sources'
        sources.mkdir(parents=True, exist_ok=True)
        for path in (P6 / 'sources').glob('*.csv'):
            shutil.copy2(path, sources / path.name)
        env = {**os.environ, 'PHASE6_SANDBOX': str(directory), 'HF_HUB_OFFLINE': '1'}
        for stage in ('download_documents', 'extract_text', 'clean_text', 'chunk_documents'):
            subprocess.run([sys.executable, str(P6 / 'scripts' / f'{stage}.py')],
                           cwd=ROOT, env=env, check=True)
    report = directory / 'reference_report.json'
    load_heldout(directory / 'data/processed_contexts', report)
    print(report.read_text())


if __name__ == '__main__':
    main()
