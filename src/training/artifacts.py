"""Bounded evaluation snapshots, one JSONL history, and reversible legacy cleanup."""

import json
import os
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED


def _atomic_text(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, payload):
    _atomic_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def record_evaluation(run_dir, payload, *, kind="evaluation", best=False):
    """Append the complete report; replace only the latest/best periodic snapshots."""
    if kind not in ("evaluation", "video"):
        raise ValueError("evaluation kind must be evaluation or video")
    record = {**payload, "kind": kind}
    line = json.dumps(record, allow_nan=False) + "\n"
    root = Path(run_dir)
    with (root / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    if kind == "evaluation":
        write_json(root / "evaluation_latest.json", record)
        if best:
            write_json(root / "evaluation_best.json", record)


def _legacy_sources(root):
    sources = [(p, "evaluation") for p in root.glob("evaluation_*.json")
               if re.fullmatch(r"evaluation_\d+\.json", p.name)]
    sources += [(p, "video") for p in (root / "videos").glob("step_*/evaluation.json")]
    return sorted(sources)


def read_evaluations(run_dir, *, include_videos=False):
    """Read both layouts, deduplicating complete records if migration was interrupted."""
    root = Path(run_dir)
    records = []
    history = root / "evaluations.jsonl"
    if history.exists():
        for line in history.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("kind") not in ("evaluation", "video"):
                    raise ValueError(f"invalid evaluation kind in {history}")
                records.append(record)
    for path, kind in _legacy_sources(root):
        records.append({**json.loads(path.read_text(encoding="utf-8")), "kind": kind})
    unique = {}
    for record in records:
        key = json.dumps(record, sort_keys=True, allow_nan=False)
        unique.setdefault(key, record)
    result = [r for r in unique.values() if include_videos or r["kind"] == "evaluation"]
    return sorted(result, key=lambda r: (r["environment_steps"], r["kind"], r.get("update", r.get("updates", 0))))


def compact_evaluations(run_dir):
    """Archive original JSON bytes before removing redundant loose reports.

    Intended for stopped/copied runs. Source changes abort deletion. Checkpoints,
    TensorBoard, videos, external eval reports and other run files are untouched.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise ValueError(f"run directory does not exist: {root}")
    sources = _legacy_sources(root)
    if not sources:
        return {"run_dir": str(root), "removed_json_files": 0}
    originals = {path: path.read_bytes() for path, _ in sources}
    # Parse everything before creating an archive or changing the history.
    records = read_evaluations(root, include_videos=True)
    periodic = [r for r in records if r["kind"] == "evaluation"]
    best = max(periodic, key=lambda r: (r["summary"]["success_rate"], r["summary"]["mean_return"])) if periodic else None
    serialized = "".join(json.dumps(r, allow_nan=False) + "\n" for r in records)
    history = root / "evaluations.jsonl"
    old_history = history.read_bytes() if history.exists() else None
    archive = root / "evaluation_archive.zip"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=".evaluation_archive.", suffix=".zip", delete=False) as stream:
            temporary = Path(stream.name)
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as output:
            names = set()
            if archive.exists():
                with ZipFile(archive) as previous:
                    for info in previous.infolist():
                        output.writestr(info, previous.read(info.filename))
                        names.add(info.filename)
            for path, data in originals.items():
                name = path.relative_to(root).as_posix()
                if name in names:
                    with ZipFile(archive) as previous:
                        if previous.read(name) != data:
                            raise ValueError(f"archive already contains different data: {name}")
                else:
                    output.writestr(name, data)
        # Verify exact bytes and ZIP integrity before deleting any original.
        with ZipFile(temporary) as output:
            if output.testzip() is not None:
                raise ValueError("evaluation archive failed its integrity check")
            for path, data in originals.items():
                if output.read(path.relative_to(root).as_posix()) != data:
                    raise ValueError(f"evaluation archive verification failed: {path}")
        if (_legacy_sources(root) != sources
                or any(path.read_bytes() != data for path, data in originals.items())
                or (history.read_bytes() if history.exists() else None) != old_history):
            raise RuntimeError("evaluation files changed during compaction; stop the writer and retry")
        temporary.replace(archive)
        _atomic_text(history, serialized)
        # Round-trip the persisted history before removal, independently of originals.
        persisted = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
        if persisted != records:
            raise ValueError("evaluation history failed its round-trip check")
        if periodic:
            write_json(root / "evaluation_latest.json", periodic[-1])
            write_json(root / "evaluation_best.json", best)
        for path, data in originals.items():
            if path.read_bytes() != data:
                raise RuntimeError(f"evaluation file changed before removal; original retained: {path}")
            path.unlink()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"run_dir": str(root), "removed_json_files": len(sources),
            "evaluation_records": len(periodic), "video_records": len(records) - len(periodic),
            "archive": str(archive), "history": str(history)}
