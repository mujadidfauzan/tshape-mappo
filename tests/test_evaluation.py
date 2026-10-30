"""Live evaluator progress, partial reports, and bounded video recording."""

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "install requirements-training.txt for policy evaluation tests")
class EvaluationTests(unittest.TestCase):
    def agent_and_config(self):
        from src.algorithms.mappo import MAPPO
        from src.envs.config import EnvironmentConfig
        from src.training.config import MAPPOConfig
        torch.set_num_threads(1)
        return MAPPO(34, 33, config=MAPPOConfig(actor_hidden_size=8, critic_hidden_size=8)), EnvironmentConfig(max_steps=2)

    def checkpoint(self, directory):
        from src.training.trainer import feature_schema
        agent, config = self.agent_and_config()
        xml = config.resolved_model_path.read_text()
        path = Path(directory) / "policy.pt"
        agent.save_checkpoint(path, {"feature_schema": feature_schema(), "model_xml": xml,
            "model_sha256": hashlib.sha256(xml.encode()).hexdigest(),
            "environment_config": config.to_dict(), "software": {}})
        return path

    def test_progress_is_emitted_during_each_episode_and_preserves_results_and_rng(self):
        from src.evaluation.evaluator import evaluate_policy
        agent, config = self.agent_and_config()
        before_rng = torch.get_rng_state().clone()
        before_count = agent.obs_normalizer.count.clone()
        events = []
        result = evaluate_policy(agent, config, episodes=2, seed=200000,
                                 progress_callback=events.append, progress_interval_steps=1)
        self.assertEqual([e["event"] for e in events],
            ["episode_start", "episode_progress", "episode_progress", "episode_end"] * 2)
        self.assertEqual([e["step"] for e in events if e["event"] == "episode_progress"], [1, 2, 1, 2])
        self.assertEqual([e["seed"] for e in events if e["event"] == "episode_end"], [200000, 200001])
        torch.testing.assert_close(torch.get_rng_state(), before_rng)
        torch.testing.assert_close(agent.obs_normalizer.count, before_count)
        self.assertTrue(agent.training)
        baseline = evaluate_policy(agent, config, episodes=2, seed=200000)
        self.assertEqual(result, baseline)

    def test_cli_saves_completed_episodes_before_interruption(self):
        from scripts.evaluate import main
        from src.evaluation.evaluator import evaluate_policy
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory)
            report_path = Path(directory) / "evaluation.json"
            def interrupt_second_episode(*args, **kwargs):
                callback = kwargs.pop("progress_callback")
                def progress(event):
                    if event["event"] == "episode_start" and event["episode"] == 2:
                        raise KeyboardInterrupt
                    callback(event)
                return evaluate_policy(*args, **kwargs, progress_callback=progress)
            argv = ["evaluate.py", str(checkpoint), "--episodes", "3", "--json", str(report_path), "--quiet"]
            with patch.object(sys, "argv", argv), patch("src.evaluation.evaluator.evaluate_policy", side_effect=interrupt_second_episode), redirect_stdout(io.StringIO()):
                code = main()
            self.assertEqual(code, 130)
            report = json.loads(report_path.read_text())
            self.assertFalse(report["complete"])
            self.assertEqual(report["completed_episodes"], 1)
            self.assertEqual(report["requested_episodes"], 3)
            self.assertEqual(report["summary"]["episodes"], 1)
            self.assertEqual(len(report["reports"]), 1)
            self.assertFalse(report_path.with_suffix(".json.tmp").exists())

    @unittest.skipUnless(shutil.which("ffmpeg") and sys.platform.startswith("linux"), "requires FFmpeg and Linux EGL")
    def test_cli_records_only_requested_video_count_and_streams_progress(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory)
            videos = Path(directory) / "videos"
            report_path = Path(directory) / "evaluation.json"
            command = [sys.executable, str(root / "scripts/evaluate.py"), str(checkpoint), "--episodes", "2",
                       "--video-dir", str(videos), "--video-episodes", "1", "--json", str(report_path),
                       "--progress-interval-steps", "1"]
            process = subprocess.run(command, cwd=root, env={**os.environ, "MUJOCO_GL": "egl"},
                                     capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertIn("Loading checkpoint:", process.stdout)
            self.assertIn("[1/2] seed=200000 starting", process.stdout)
            self.assertIn("step=1/2", process.stdout)
            self.assertIn("[2/2] seed=200001 done", process.stdout)
            self.assertEqual(len(list(videos.glob("*.mp4"))), 1)
            report = json.loads(report_path.read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(report["completed_episodes"], 2)
            self.assertIsNotNone(report["reports"][0]["video"])
            self.assertIsNone(report["reports"][1]["video"])
            self.assertEqual(report["torch_threads"], 1)


if __name__ == "__main__":
    unittest.main()
