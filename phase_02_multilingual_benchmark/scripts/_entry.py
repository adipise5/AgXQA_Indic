from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from phase_02_multilingual_benchmark.cli import run


def main(command):
    try:
        run(command)
    except (ValueError,FileNotFoundError) as error:
        raise SystemExit(f"Phase 2: {error}") from error
