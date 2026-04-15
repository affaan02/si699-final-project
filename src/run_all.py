"""
run_all.py — Execute the full pipeline end to end
===================================================
Shortcut for: ingest → features → model → figures.
Run this once the raw CSVs are in data/db1b/ and data/t100/.

Usage:
    python src/run_all.py
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent
STEPS = ["ingest.py", "features.py", "model.py", "figures.py"]


def main():
    for step in STEPS:
        print(f"\n{'#' * 60}\n# Running {step}\n{'#' * 60}")
        result = subprocess.run([sys.executable, str(SRC / step)])
        if result.returncode != 0:
            print(f"\n[FAIL] {step} exited with code {result.returncode}")
            sys.exit(result.returncode)
    print("\n✅ Pipeline complete. Check outputs/figures/ and outputs/tables/")


if __name__ == "__main__":
    main()
