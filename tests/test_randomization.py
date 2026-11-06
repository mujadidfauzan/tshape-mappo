"""Independent pose/agent randomization, narrow doorway and saved reset settings."""

from contextlib import redirect_stdout
from dataclasses import replace
import importlib.util
import io
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.envs.config import load_config
from src.envs.tshape_env import TShapeParallelEnv


class RandomizationTests(unittest.TestCase):
    def env(self, path="configs/medium_narrow.yaml", **overrides):
        env = TShapeParallelEnv(load_config(path, **overrides))
        self.addCleanup(env.close)
        return env

    def test_all_randomization_combinations_are_safe_and_reproducible(self):
        for path in (None, "configs/medium_narrow.yaml"):
            for count in (4, 16):
                for agents, position, orientation in product((False, True), repeat=3):
                    with self.subTest(path=path, count=count, agents=agents,
                                      position=position, orientation=orientation):
                        env = self.env(path, num_agents=count, randomize_agents=agents,
                                       randomize_object_position=position,
                                       randomize_object_orientation=orientation)
                        sequences = []
                        formations = []
                        for _ in range(2):
                            sequence = []
                            for seed in (42, None, None):
                                _, infos = env.reset(seed=seed)
                                initial = infos["agent_0"]["initial_state"]
                                sequence.append(initial)
                                xy, yaw = np.array(initial["object_position"]), initial["object_yaw"]
                                if position:
                                    self.assertTrue(np.all(np.abs(xy - env.config.object_pose[:2])
                                                           <= np.array(env.config.object_position_range) + 1e-9))
                                else:
                                    np.testing.assert_allclose(xy, env.config.object_pose[:2])
                                if orientation:
                                    self.assertGreaterEqual(yaw, env.config.object_yaw_range[0])
                                    self.assertLessEqual(yaw, env.config.object_yaw_range[1])
                                else:
                                    self.assertAlmostEqual(yaw, env.config.object_pose[2])
                                if not agents:
                                    rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                                                         [np.sin(yaw), np.cos(yaw)]])
                                    formations.append((np.array(initial["agent_positions"]) - xy) @ rotation)
                                self.assertFalse(env.object_inside_goal())
                                self.assertTrue(env.physics.diagnostics()["stable"])
                                self.assertEqual(env.physics.boundary_penetration(), 0.)
                                self.assertFalse(any(c.dist < -1e-9 for c in
                                                     env.physics.data.contact[:env.physics.data.ncon]))
                                np.testing.assert_array_equal(env.physics.data.qvel, 0.)
                            sequences.append(sequence)
                        self.assertEqual(sequences[0], sequences[1])
                        if agents or position or orientation:
                            self.assertNotEqual(sequences[0][0], sequences[0][1])
                        else:
                            self.assertEqual(sequences[0][0], sequences[0][1])
                        for formation in formations[1:]:
                            np.testing.assert_allclose(formation, formations[0], atol=1e-9)

    def test_explicit_fixed_agents_follow_object_pose(self):
        positions = [[-1.68, .45], [-1.32, .45], [-1.61, -.26], [-1.61, .26]]
        env = self.env(randomize_agents=False, agent_positions=positions,
                       randomize_object_position=True, randomize_object_orientation=True)
        _, infos = env.reset(seed=7)
        initial = infos["agent_0"]["initial_state"]
        yaw = initial["object_yaw"]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        expected = (np.array(positions) - env.config.object_pose[:2]) @ rotation.T + initial["object_position"]
        np.testing.assert_allclose(initial["agent_positions"], expected)

    def test_invalid_ranges_and_impossible_layouts_fail_explicitly(self):
        for values in (dict(randomize_object_position=1), dict(randomize_object_orientation=1),
                       dict(object_position_range=[-1, 1]), dict(object_position_range=[1]),
                       dict(object_position_range=[np.nan, 1]), dict(object_yaw_range=[1, 1]),
                       dict(object_yaw_range=[1, -1]), dict(object_yaw_range=[0, np.inf]),
                       dict(randomize_object_position=True, object_position_range=[0, 0])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                load_config("configs/medium_narrow.yaml", **values)
        with self.assertRaisesRegex(ValueError, "after 100 attempts"):
            self.env(object_pose=(1.5, 0., 0.), randomize_object_position=True,
                     object_position_range=(.01, .01))

    def test_narrow_doorway_requires_alignment_and_remains_physically_passable(self):
        old = self.env("configs/medium.yaml")
        env = self.env()
        self.assertAlmostEqual(old.doorway.y_max - old.doorway.y_min, .9)
        self.assertAlmostEqual(env.doorway.y_max - env.doorway.y_min, .7)
        positions = [[-1.87, y] for y in (-.24, -.08, .08, .24)]
        env.reset(seed=42)
        env.physics.reset([0., 0., 0.], positions)
        self.assertFalse(env.physics.diagnostics()["stable"])
        env.physics.reset([0., 0., math.pi/2], positions)
        self.assertTrue(env.physics.diagnostics()["stable"])
        env = self.env(randomize_agents=False, object_pose=(-1.2, 0., math.pi/2),
                       agent_positions=positions)
        env.reset(seed=42)
        while env.agents:
            _, _, _, _, infos = env.step({name: [1., 0.] for name in env.agents})
            self.assertTrue(infos["agent_0"]["stable"], infos["agent_0"]["failure_details"])
        self.assertTrue(infos["agent_0"]["success"])
        self.assertLess(infos["agent_0"]["episode_step"], env.max_steps)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "install requirements-training.txt")
    def test_checkpoint_evaluation_and_resume_preserve_randomization(self):
        from src.evaluation.checkpoints import loaded_policy
        from src.evaluation.evaluator import evaluate_policy
        from src.training.config import MAPPOConfig
        from src.training.trainer import Trainer
        cfg = MAPPOConfig(total_steps=4, rollout_steps=2, ppo_epochs=1,
                          actor_hidden_size=8, critic_hidden_size=8, eval_episodes=2,
                          video_interval_steps=0, tensorboard=False)
        env_cfg = load_config("configs/medium_narrow.yaml", max_steps=2,
                              randomize_object_position=True, randomize_object_orientation=True)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            trainer = Trainer(root/"run", cfg, env_cfg)
            trainer.train()
            checkpoint = root/"run/checkpoints/latest.pt"
            with loaded_policy(checkpoint) as (agent, saved, payload):
                self.assertTrue(saved.randomize_object_position)
                self.assertTrue(saved.randomize_object_orientation)
                first = evaluate_policy(agent, saved, episodes=2, seed=123)
                self.assertEqual(first, evaluate_policy(agent, saved, episodes=2, seed=123))
                self.assertNotEqual(first["reports"][0]["initial_state"], first["reports"][1]["initial_state"])
                self.assertAlmostEqual(first["reports"][0]["initial_state"]["doorway"]["y_max"], .35)
                rng_state = payload["training"]["environment_rng_state"]
            resumed = Trainer(root/"resume", replace(cfg, total_steps=6), resume=checkpoint)
            self.assertEqual(resumed.env.np_random.bit_generator.state, rng_state)
            self.assertTrue(resumed.environment_config.randomize_object_position)
            self.assertTrue(resumed.environment_config.randomize_object_orientation)
            self.assertEqual(resumed.train()["environment_steps"], 6)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "install requirements-training.txt")
    def test_cli_flags_override_yaml_and_resume_rejects_task_overrides(self):
        from src.training.config import MAPPOConfig
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = MAPPOConfig(total_steps=2, rollout_steps=2, ppo_epochs=1, eval_episodes=1,
                              actor_hidden_size=8, critic_hidden_size=8,
                              video_interval_steps=0, tensorboard=False)
            (root/"mappo.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
            (root/"env.yaml").write_text(yaml.safe_dump(
                load_config("configs/medium_narrow.yaml", max_steps=2).to_dict()))
            result = subprocess.run([sys.executable, "scripts/train.py", "--run-dir", str(root/"run"),
                                     "--config", str(root/"mappo.yaml"), "--env-config", str(root/"env.yaml"),
                                     "--no-randomize-agents", "--randomize-object-position",
                                     "--randomize-object-orientation", "--object-position-range", ".2", ".3",
                                     "--object-yaw-range", "-1.0", "1.0"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            env = json.loads((root/"run/manifest.json").read_text())["environment"]
            self.assertFalse(env["randomize_agents"])
            self.assertTrue(env["randomize_object_position"])
            self.assertTrue(env["randomize_object_orientation"])
            self.assertEqual(env["object_position_range"], [.2, .3])
            self.assertEqual(env["object_yaw_range"], [-1., 1.])
            for option in ("--randomize-object-position", "--no-randomize-object-orientation",
                           "--object-position-range", "--object-yaw-range"):
                extra = ["0", "1"] if option.endswith("range") else []
                result = subprocess.run([sys.executable, "scripts/train.py", "--run-dir", str(root/"resume"),
                                         "--resume", str(root/"run/checkpoints/latest.pt"),
                                         "--total-steps", "4", option, *extra], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("resume restores", result.stderr)
            self.assertFalse((root/"resume").exists())
