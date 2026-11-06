"""Small integration checks; no claim that a short training run learns the task."""

from contextlib import redirect_stdout
from copy import deepcopy
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
from unittest.mock import patch


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
            trainer=Trainer(root/"run",cfg,replace(load_config(),max_steps=3,randomize_agents=False))
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
                self.assertEqual(payload["training"]["task"],"easy_v2")
                # The existing fixed-four results must keep their original reset behavior.
                legacy=deepcopy(payload)
                legacy["training"]["task"]="fixed_easy_v1"
                legacy["training"]["environment_config"]["agent_positions"]=rows[0]["initial_state"]["agent_positions"]
                for key in ["num_agents","randomize_agents","spawn_gap_min","spawn_gap_max"]:
                    legacy["training"]["environment_config"].pop(key)
                legacy["mappo_config"].pop("eval_episodes")
                import torch
                torch.save(legacy,root/"legacy.pt")
                with loaded_policy(root/"legacy.pt") as (old_agent,old_env,_):
                    self.assertFalse(old_env.randomize_agents)
                    self.assertEqual(old_env.num_agents,4)
                    old_report=evaluate_policy(old_agent,old_env)
                    self.assertEqual(old_report["reports"][0]["initial_state"],rows[0]["initial_state"])
            resumed=Trainer(root/"resume",replace(cfg,total_steps=12),resume=root/"run/checkpoints/latest.pt")
            self.assertEqual(resumed.train()["environment_steps"],12)
            self.assertTrue((root/"resume/checkpoints/best.pt").exists())

    def test_removed_cli_commands_are_not_options(self):
        for option in ["--curriculum","--init-actor"]:
            result=subprocess.run([sys.executable,"scripts/train.py","--run-dir","unused",option,"x"],
                                  capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("unrecognized arguments",result.stderr)

    def test_startup_failures_close_resources_and_propagate(self):
        from src.training.trainer import Trainer
        for stage in ("reset", "checkpoint"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                trainer = Trainer(Path(directory)/"run", self.config())
                error = OSError(f"simulated {stage} failure")
                target = trainer.env if stage == "reset" else trainer
                with patch.object(target, stage, side_effect=error), \
                        patch.object(trainer.env, "close", wraps=trainer.env.close) as close_env, \
                        patch.object(trainer.logger, "close", wraps=trainer.logger.close) as close_logger:
                    with self.assertRaises(OSError) as raised:
                        trainer.train()
                    self.assertIs(raised.exception, error)
                    close_env.assert_called_once_with()
                    close_logger.assert_called_once_with()

    def test_population_training_random_resets_evaluation_and_rng_resume(self):
        from src.envs.config import load_config
        from src.evaluation.checkpoints import loaded_policy
        from src.evaluation.evaluator import evaluate_policy
        from src.training.trainer import Trainer
        for count in [1,2,8,16]:
            with self.subTest(num_agents=count),tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
                root=Path(directory)
                cfg=replace(self.config(),eval_episodes=2)
                trainer=Trainer(root/"run",cfg,replace(load_config(),max_steps=2,num_agents=count))
                self.assertEqual(trainer.agent.dimensions["num_agents"],count)
                trainer.train()
                rows=[json.loads(s) for s in (root/"run/episodes.jsonl").read_text().splitlines()]
                self.assertTrue(all(len(r["initial_state"]["agent_positions"])==count for r in rows))
                self.assertNotEqual(rows[0]["initial_state"]["agent_positions"],rows[1]["initial_state"]["agent_positions"])
                with loaded_policy(root/"run/checkpoints/latest.pt") as (agent,env_config,payload):
                    self.assertEqual(env_config.num_agents,count)
                    self.assertTrue(env_config.randomize_agents)
                    first=evaluate_policy(agent,env_config,episodes=2,seed=1000000)
                    second=evaluate_policy(agent,env_config,episodes=2,seed=1000000)
                    self.assertEqual(first,second)
                    self.assertNotEqual(first["reports"][0]["initial_state"],first["reports"][1]["initial_state"])
                    state=payload["training"]["environment_rng_state"]
                evaluations=[json.loads(s) for s in (root/"run/evaluations.jsonl").read_text().splitlines()]
                self.assertTrue(all(len(r["reports"])==2 for r in evaluations))
                resumed=Trainer(root/"resume",replace(cfg,total_steps=12),resume=root/"run/checkpoints/latest.pt")
                self.assertEqual(resumed.env.np_random.bit_generator.state,state)
                resumed.train()
                self.assertEqual(resumed.agent.dimensions["num_agents"],count)

    def test_cli_agent_count_overrides_default_yaml_and_rejects_resume_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=root/"mappo.yaml"
            import yaml
            config.write_text(yaml.safe_dump(replace(self.config(),total_steps=4).to_dict()))
            env_config=root/"environment.yaml"
            from src.envs.config import load_config
            env_config.write_text(yaml.safe_dump(replace(load_config(),max_steps=2).to_dict()))
            result=subprocess.run([sys.executable,"scripts/train.py","--run-dir",str(root/"run"),
                "--config",str(config),"--env-config",str(env_config),"--num-agents","8"],
                capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((root/"run/manifest.json").read_text())
            self.assertEqual(manifest["environment"]["num_agents"],8)
            self.assertIn("agents=8",result.stdout)
            result=subprocess.run([sys.executable,"scripts/train.py","--run-dir",str(root/"resume"),
                "--resume",str(root/"run/checkpoints/latest.pt"),"--total-steps","8","--num-agents","4"],
                capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("resume restores",result.stderr)
            self.assertFalse((root/"resume").exists())

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
