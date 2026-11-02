"""Small atomic JSON snapshots and append-only JSONL logs."""

import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def record_evaluation(run_dir, result, best=False):
    root = Path(run_dir)
    append_jsonl(root / "evaluations.jsonl", result)
    if result["kind"] == "evaluation":
        write_json(root / "evaluation_latest.json", result)
        if best:
            write_json(root / "evaluation_best.json", result)
