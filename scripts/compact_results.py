"""Consolidate legacy evaluation JSON files after archiving their original bytes."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.artifacts import compact_evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="stopped or copied training run directories")
    args = parser.parse_args()
    for directory in args.run_dirs:
        result = compact_evaluations(directory)
        print(f"{directory}: consolidated {result['removed_json_files']} JSON files", flush=True)
        if result["removed_json_files"]:
            print(f"  history: {result['history']}; original files: {result['archive']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
