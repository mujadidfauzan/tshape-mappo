"""Compaction must preserve history and original bytes before removing JSONs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from src.training.artifacts import compact_evaluations, read_evaluations, record_evaluation


class EvaluationArtifactTests(unittest.TestCase):
    def payload(self, step, success=0, reward=0):
        return {"environment_steps": step, "update": step // 2,
                "summary": {"success_rate": success, "mean_return": reward},
                "reports": [{"seed": 42, "note": "original data"}]}

    def test_writer_keeps_two_periodic_snapshots_and_one_complete_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = self.payload(2, 1, 20), self.payload(4, 0, 2)
            record_evaluation(root, first, best=True)
            record_evaluation(root, second)
            record_evaluation(root, self.payload(4), kind="video")
            self.assertEqual(len(read_evaluations(root)), 2)
            self.assertEqual(len(read_evaluations(root, include_videos=True)), 3)
            self.assertEqual(json.loads((root / "evaluation_best.json").read_text())["environment_steps"], 2)
            self.assertEqual(json.loads((root / "evaluation_latest.json").read_text())["environment_steps"], 4)
            self.assertEqual({p.name for p in root.iterdir()},
                             {"evaluations.jsonl", "evaluation_best.json", "evaluation_latest.json"})

    def test_compaction_preserves_original_bytes_all_reports_and_non_evaluation_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = {}
            for name, data in [("evaluation_000001.json", self.payload(2, .5, 10)),
                               ("evaluation_000002.json", self.payload(4, .2, 5)),
                               ("videos/step_000000004/evaluation.json", self.payload(4))]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = (json.dumps(data, indent=3) + "\n").encode()
                path.write_bytes(raw)
                originals[name] = raw
            # Existing JSONL can coexist with legacy reports without duplicates.
            record_evaluation(root, self.payload(4, .2, 5))
            for name in ("manifest.json", "eval_custom.json", "episodes.jsonl", "checkpoints/best.pt",
                         "videos/step_000000004/policy_seed_42.mp4"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"untouched")
            before = read_evaluations(root, include_videos=True)
            result = compact_evaluations(root)
            self.assertEqual(result["removed_json_files"], 3)
            self.assertEqual(read_evaluations(root, include_videos=True), before)
            with ZipFile(root / "evaluation_archive.zip") as archive:
                for name, raw in originals.items():
                    self.assertEqual(archive.read(name), raw)
                    self.assertFalse((root / name).exists())
            self.assertEqual(json.loads((root / "evaluation_best.json").read_text())["environment_steps"], 2)
            self.assertEqual(json.loads((root / "evaluation_latest.json").read_text())["environment_steps"], 4)
            self.assertEqual((root / "checkpoints/best.pt").read_bytes(), b"untouched")
            self.assertEqual((root / "eval_custom.json").read_bytes(), b"untouched")
            archive_bytes = (root / "evaluation_archive.zip").read_bytes()
            self.assertEqual(compact_evaluations(root)["removed_json_files"], 0)
            self.assertEqual((root / "evaluation_archive.zip").read_bytes(), archive_bytes)
            # Legacy writers may create more reports later; a new pass extends
            # the same archive instead of creating backup files per evaluation.
            extra = root / "evaluation_000003.json"
            extra.write_text(json.dumps(self.payload(6, .8, 15)))
            compact_evaluations(root)
            self.assertEqual(len(read_evaluations(root)), 3)
            with ZipFile(root / "evaluation_archive.zip") as archive:
                self.assertEqual(len(archive.namelist()), 4)

    def test_malformed_history_aborts_before_archiving_or_deleting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "evaluation_000001.json"
            first.write_text(json.dumps(self.payload(2)))
            second = root / "evaluation_000002.json"
            second.write_text('{"interrupted":')
            with self.assertRaises(json.JSONDecodeError):
                compact_evaluations(root)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertFalse((root / "evaluation_archive.zip").exists())
            self.assertFalse((root / "evaluations.jsonl").exists())

    def test_changed_sources_abort_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "evaluation_000001.json"
            path.write_text(json.dumps(self.payload(2)))
            def change_source(archive):
                path.write_text(json.dumps(self.payload(4)))
                return None
            with patch.object(ZipFile, "testzip", change_source):
                with self.assertRaisesRegex(RuntimeError, "changed during compaction"):
                    compact_evaluations(root)
            self.assertTrue(path.exists())
            self.assertFalse((root / "evaluations.jsonl").exists())
            self.assertFalse((root / "evaluation_archive.zip").exists())


if __name__ == "__main__":
    unittest.main()
