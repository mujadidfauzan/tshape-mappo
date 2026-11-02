"""Small integration checks; no claim that a short training run learns the task."""

from contextlib import redirect_stdout
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None,"install requirements-training.txt")
class TrainingTests(unittest.TestCase):
    def config(self,**overrides):
        from src.training.config import MAPPOConfig
        return MAPPOConfig(total_steps=8,rollout_steps=4,ppo_epochs=1,actor_hidden_size=8,
                           critic_hidden_size=8,eval_interval_steps=4,
                           checkpoint_interval_steps=4,video_interval_steps=0,tensorboard=False,
                           **overrides)

    def test_training_checkpoint_evaluation_and_resume(self):
        from src.envs.config import load_config
        from src.evaluation.checkpoints import loaded_policy
        from src.evaluation.evaluator import evaluate_policy
        from src.training.trainer import Trainer
        cfg=self.config()
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            trainer=Trainer(root/"run",cfg,replace(load_config(),max_steps=3))
            result=trainer.train()
            self.assertEqual(result["environment_steps"],8)
            self.assertEqual({p.name for p in (root/"run/checkpoints").iterdir()},{"best.pt","latest.pt"})
            rows=[json.loads(l) for l in (root/"run/episodes.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["initial_state"]==rows[0]["initial_state"] for r in rows))
            for row in [json.loads(l) for l in (root/"run/updates.jsonl").read_text().splitlines()]:
                self.assertAlmostEqual(sum(row["reward_components_mean"].values()),row["reward_mean_step"],places=6)
            with loaded_policy(root/"run/checkpoints/latest.pt") as (agent,env,payload):
                report=evaluate_policy(agent,env,episodes=2)
                self.assertEqual(report["reports"][0]["initial_state"],report["reports"][1]["initial_state"])
                self.assertAlmostEqual(report["reports"][0]["episode_metrics"]["episode_return"],
                                       report["reports"][1]["episode_metrics"]["episode_return"])
                self.assertEqual(payload["training"]["task"],"fixed_easy_v1")
            resumed=Trainer(root/"resume",replace(cfg,total_steps=12),resume=root/"run/checkpoints/latest.pt")
            self.assertEqual(resumed.train()["environment_steps"],12)
            self.assertTrue((root/"resume/checkpoints/best.pt").exists())

    def test_removed_cli_commands_are_not_options(self):
        for option in ["--curriculum","--init-actor"]:
            result=subprocess.run([sys.executable,"scripts/train.py","--run-dir","unused",option,"x"],
                                  capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("unrecognized arguments",result.stderr)

    def test_resume_before_the_first_evaluation_is_valid(self):
        from src.envs.config import load_config
        from src.training.trainer import Trainer
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            first=Trainer(root/"first",self.config(),replace(load_config(),max_steps=2))
            first.checkpoint("latest.pt")
            first.env.close()
            first.logger.close()
            resumed=Trainer(root/"resume",self.config(),resume=root/"first/checkpoints/latest.pt")
            self.assertEqual(resumed.train()["environment_steps"],8)
            manifest=json.loads((root/"resume/manifest.json").read_text())
            self.assertIsNone(manifest["previous_run_best_score"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"FFmpeg required")
    def test_1080p_video_and_tensorboard_remain_available(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from src.envs.config import load_config
        from src.training.trainer import Trainer
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            cfg=replace(self.config(),total_steps=4,video_interval_steps=4,tensorboard=True)
            Trainer(root/"run",cfg,replace(load_config(),max_steps=2)).train()
            video=root/"run/videos/step_000000004/policy.mp4"
            result=subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
                "-show_entries","stream=width,height,codec_name,nb_frames","-of","json",str(video)],
                capture_output=True,text=True,check=True)
            stream=json.loads(result.stdout)["streams"][0]
            self.assertEqual((stream["width"],stream["height"]),(1920,1080))
            self.assertEqual(stream["codec_name"],"h264")
            self.assertEqual(int(stream["nb_frames"]),3)
            tags=EventAccumulator(str(root/"run/tensorboard")).Reload().Tags()["scalars"]
            self.assertIn("reward/distance_mean",tags)
            self.assertIn("reward/orientation_mean",tags)
