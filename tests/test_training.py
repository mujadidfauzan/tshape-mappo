"""Tiny end-to-end smoke run; does not assert that an untrained policy succeeds."""

from dataclasses import replace
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


@unittest.skipIf(torch is None, "install requirements-training.txt for M3 tests")
class TrainingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe") and sys.platform.startswith("linux"),
                         "requires FFmpeg, ffprobe and Linux EGL for the video integration test")
    def test_episode_video_milestones_encode_1080p_and_resume_counts_completed_episodes(self):
        import yaml
        from src.algorithms.mappo import MAPPO
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = path / "mappo.yaml"
            config.write_text(yaml.safe_dump({"rollout_steps": 3, "ppo_epochs": 1,
                "actor_hidden_size": 8, "critic_hidden_size": 8,
                "eval_episodes": 1, "video_interval_episodes": 1}))
            environment = path / "environment.yaml"
            environment.write_text(yaml.safe_dump({"max_steps": 2, "randomize_reset": False}))
            run = path / "initial"
            variables = {**os.environ, "MUJOCO_GL": "egl"}
            command = [sys.executable, str(root / "scripts/train.py"), "--config", str(config),
                       "--env-config", str(environment), "--total-episodes", "2", "--run-dir", str(run)]
            completed = subprocess.run(command, cwd=root, env=variables, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            for episode in (1, 2):
                folder = run / "videos" / f"episode_{episode:06d}"
                report = json.loads((folder / "evaluation.json").read_text())
                self.assertEqual(report["training_episodes"], episode)
                video = report["reports"][0]["video"]
                probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,codec_name,nb_frames,r_frame_rate", "-of", "json",
                    video["path"]], capture_output=True, text=True, check=True)
                stream = json.loads(probe.stdout)["streams"][0]
                self.assertEqual((stream["width"], stream["height"]), (1920, 1080))
                self.assertEqual(stream["codec_name"], "h264")
                self.assertEqual(stream["r_frame_rate"], "20/1")
                self.assertEqual(int(stream["nb_frames"]), 3)
                self.assertEqual(report["checkpoint"], str(run / "checkpoints/latest.pt"))
                self.assertTrue(report["checkpoint_is_overwritten"])
            self.assertEqual({p.name for p in (run / "checkpoints").iterdir()}, {"best.pt", "latest.pt"})
            self.assertFalse(list(run.rglob("*.partial.mp4")))
            # Enabling video must preserve the sampled training trajectory,
            # the learned parameters, normalization and the training RNG.
            without_video = path / "without_video"
            command = [sys.executable, str(root / "scripts/train.py"), "--config", str(config),
                       "--env-config", str(environment), "--total-episodes", "2",
                       "--video-interval-episodes", "0", "--run-dir", str(without_video)]
            completed = subprocess.run(command, cwd=root, env=variables, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            _, recorded = MAPPO.from_checkpoint(run / "checkpoints/latest.pt")
            _, unrecorded = MAPPO.from_checkpoint(without_video / "checkpoints/latest.pt")
            for key, value in recorded["model"].items():
                torch.testing.assert_close(value, unrecorded["model"][key], rtol=0, atol=0)
            torch.testing.assert_close(recorded["training"]["torch_rng_state"],
                                       unrecorded["training"]["torch_rng_state"], rtol=0, atol=0)
            self.assertEqual((run / "episodes.jsonl").read_text(), (without_video / "episodes.jsonl").read_text())
            resumed = path / "resumed"
            command = [sys.executable, str(root / "scripts/train.py"), "--resume",
                       str(run / "checkpoints/latest.pt"), "--total-episodes", "3", "--run-dir", str(resumed)]
            completed = subprocess.run(command, cwd=root, env=variables, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            folders = list((resumed / "videos").iterdir())
            self.assertEqual([p.name for p in folders], ["episode_000003"])
            _, payload = MAPPO.from_checkpoint(resumed / "checkpoints/latest.pt")
            self.assertEqual(payload["training"]["completed_episodes"], 3)
            self.assertEqual({p.name for p in (resumed / "checkpoints").iterdir()}, {"best.pt", "latest.pt"})

    def test_short_run_writes_loadable_checkpoint_and_can_resume(self):
        from src.algorithms.mappo import MAPPO
        from src.evaluation.evaluator import evaluate_policy
        from src.training.config import MAPPOConfig
        from src.training.trainer import Trainer
        config = MAPPOConfig(total_steps=4, rollout_steps=2, ppo_epochs=1,
                             actor_hidden_size=8, critic_hidden_size=8,
                             eval_episodes=1, eval_interval=2, checkpoint_interval=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            trainer = Trainer(path / "initial", config=config,
                              environment_config={"max_steps": 2, "randomize_reset": False})
            summary = trainer.train()
            self.assertEqual(summary["environment_steps"], 4)
            checkpoint = path / "initial" / "checkpoints" / "latest.pt"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((path / "initial" / "updates.jsonl").is_file())
            agent, payload = MAPPO.from_checkpoint(checkpoint)
            self.assertEqual(payload["training"]["environment_steps"], 4)
            before_rng = torch.get_rng_state().clone()
            before_count = agent.obs_normalizer.count.clone()
            evaluation = evaluate_policy(agent, trainer.environment_config, episodes=1, seed=9000)
            self.assertEqual(evaluation["summary"]["episodes"], 1)
            torch.testing.assert_close(torch.get_rng_state(), before_rng)
            torch.testing.assert_close(agent.obs_normalizer.count, before_count)
            resumed = Trainer(path / "resumed", config=replace(config, total_steps=6), resume=checkpoint)
            result = resumed.train()
            self.assertEqual(result["environment_steps"], 6)
            self.assertTrue((path / "resumed" / "checkpoints" / "latest.pt").is_file())

    def test_latest_advances_while_best_retains_the_better_evaluation(self):
        from src.algorithms.mappo import MAPPO
        from src.training.config import MAPPOConfig
        from src.training.trainer import Trainer
        config = MAPPOConfig(total_steps=4, rollout_steps=2, ppo_epochs=1,
                             actor_hidden_size=8, critic_hidden_size=8,
                             eval_episodes=1, eval_interval=1, checkpoint_interval=1,
                             video_interval_episodes=0)
        scores = [{"summary": {"success_rate": 1.0, "mean_return": 20.0}},
                  {"summary": {"success_rate": 0.0, "mean_return": 2.0}}]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            trainer = Trainer(run, config=config, environment_config={"max_steps": 2})
            with patch("src.training.trainer.evaluate_policy", side_effect=scores):
                trainer.train()
            self.assertEqual({p.name for p in (run / "checkpoints").iterdir()}, {"best.pt", "latest.pt"})
            _, best = MAPPO.from_checkpoint(run / "checkpoints/best.pt")
            _, latest = MAPPO.from_checkpoint(run / "checkpoints/latest.pt")
            self.assertEqual(best["training"]["updates"], 1)
            self.assertEqual(latest["training"]["updates"], 2)
            self.assertEqual(latest["training"]["best_score"], [1.0, 20.0])

    def test_training_episode_seeds_skip_validation_range(self):
        from src.training.config import MAPPOConfig
        from src.training.trainer import Trainer
        # Only exercise the seed schedule; no simulator or networks needed here.
        trainer = Trainer.__new__(Trainer)
        trainer.config = MAPPOConfig(seed=42, eval_seed=44, eval_episodes=2)
        actual = []
        for i in range(6):
            trainer.episode_index = i
            actual.append(trainer.episode_seed())
        self.assertEqual(actual, [42, 43, 46, 47, 48, 49])


if __name__ == "__main__":
    unittest.main()
