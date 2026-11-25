"""Compatibility entry point for the reviewed Phase 2 CLI."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase_02_multilingual_benchmark.cli import run

if __name__ == "__main__":
    run("merge_languages")
