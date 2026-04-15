"""
inspect_data.py — Dump the schema of everything in data/db1b and data/t100
============================================================================
Run this AFTER downloading the BTS files and unzipping them. It prints
column names, dtypes, row counts, and a sample row for each file it finds.

If ingest.py breaks, paste the output of this script into Claude and the
ingestion code can be adapted to match your exact schema.

Usage:
    python src/inspect_data.py
"""

import pandas as pd
from pathlib import Path
from config import DB1B_DIR, T100_DIR


def inspect_dir(label: str, directory: Path):
    print(f"\n{'=' * 70}")
    print(f"  {label}   ({directory})")
    print("=" * 70)

    csvs = sorted(directory.glob("*.csv"))
    if not csvs:
        print(f"  (no CSVs found — download data and unzip into {directory})")
        return

    for f in csvs:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"\n  📄 {f.name}  ({size_mb:,.1f} MB)")

        # Read just the header + first 5 rows, don't blow up memory
        try:
            head = pd.read_csv(f, nrows=5, low_memory=False)
        except Exception as e:
            print(f"     ⚠️  Could not read: {e}")
            continue

        print(f"     columns ({len(head.columns)}):")
        for col in head.columns:
            dtype = str(head[col].dtype)
            sample = head[col].iloc[0] if len(head) > 0 else ""
            sample_str = str(sample)[:40]
            print(f"       - {col:<30} {dtype:<12}  e.g. {sample_str}")


def main():
    print("BTS Data Inspection")
    inspect_dir("DB1B (demand data)", DB1B_DIR)
    inspect_dir("T-100 (supply data)", T100_DIR)
    print("\n" + "=" * 70)
    print("  Paste this entire output into Claude if ingest.py breaks.")
    print("=" * 70)


if __name__ == "__main__":
    main()
