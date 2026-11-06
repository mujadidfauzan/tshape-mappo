"""Single-doorway geometry, transport semantics, rewards and training integration."""

from contextlib import redirect_stdout
from dataclasses import replace
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import mujoco
import numpy as np
from pettingzoo.test import parallel_api_test

from src.envs.config import load_config
from src.envs.rewards import team_reward
from src.envs.tshape_env import TShapeParallelEnv


class MediumTests(unittest.TestCase):
    def env(self, **overrides):
        env = TShapeParallelEnv(load_config("configs/medium.yaml", **overrides))
        self.addCleanup(env.close)
        return env

    def zeros(self, env):
        return {name: np.zeros(2) for name in env.agents}

    def test_seeded_resets_observations_and_parallel_api(self):
        for count in (1, 4, 8, 16):
            with self.subTest(num_agents=count):
                env = self.env(num_agents=count, max_steps=3)
                first, info = env.reset(seed=42)
                initial = info["agent_0"]["initial_state"]
                second, info = env.reset(seed=42)
                self.assertEqual(initial, info["agent_0"]["initial_state"])
                self.assertEqual(env.state().shape, (22 + 4*count,))
                for name in first:
                    np.testing.assert_array_equal(first[name], second[name])
                    self.assertEqual(first[name].shape, (23 + 4*count,))
                    self.assertTrue(env.observation_space(name).contains(first[name]))
                self.assertTrue(env.physics.diagnostics()["stable"])
                self.assertEqual(env.physics.control_dt, .05)
                np.testing.assert_allclose(first["agent_0"][-3:-1], [.15, .9])
                self.assertIn("doorway", initial)
                parallel_api_test(env, num_cycles=5)

    def test_distance_reward_is_positive_bounded_and_increases_near_goal(self):
        config = load_config("configs/medium.yaml")
        values = []
        for distance in (4., 2., 1., 0.):
            reward, parts = team_reward(distance, distance, 0., False, False, config)
            expected = config.distance_weight * (1 - math.tanh(distance/config.distance_scale))
            self.assertAlmostEqual(reward, expected)
            self.assertGreater(reward, 0.)
            self.assertLessEqual(reward, config.distance_weight)
            values.append(reward)
            for yaw in (0., .7, math.pi):
                self.assertEqual(team_reward(distance, distance, yaw, False, False, config)[0], reward)
            self.assertEqual(parts["orientation"], 0.)
        self.assertEqual(values, sorted(values))
        self.assertGreater(team_reward(2., 1., 0., False, False, config)[1]["progress"], 0.)
        self.assertLess(team_reward(1., 2., 0., False, False, config)[1]["progress"], 0.)
        reward, parts = team_reward(1., 0., 0., True, True, config)
        self.assertEqual(reward, -config.failure_penalty)
        self.assertEqual(parts["success"], 0.)

    def test_route_distance_uses_opening_and_is_continuous(self):
        env = self.env()
        door = env.doorway
        target = np.array(env.config.goal_center)
        self.assertAlmostEqual(door.route_distance([-1.5, 0.], target), 3.)
        start = np.array([-1.5, 2.])
        self.assertGreater(door.route_distance(start, target), np.linalg.norm(target-start))
        for face in (door.x_min, door.x_max):
            for y in (0., door.y_max):
                before = door.route_distance([face-1e-7, y], target)
                after = door.route_distance([face+1e-7, y], target)
                self.assertLess(abs(before-after), 1e-6)

    def test_partial_crossing_does_not_succeed(self):
        env = self.env(max_steps=1)
        env.reset(seed=42)
        env.physics.reset([.5, 0., math.pi/2])
        self.assertFalse(env.object_inside_goal())
        _, _, terminated, truncated, infos = env.step(self.zeros(env))
        self.assertFalse(any(terminated.values()))
        self.assertTrue(all(truncated.values()))
        self.assertFalse(infos["agent_0"]["success"])
        self.assertLess(infos["agent_0"]["crossing_clearance_m"], 0.)

    def test_full_crossing_terminates_at_any_yaw_without_settling_or_target_containment(self):
        for yaw in (0., .7, math.pi/2, math.pi):
            with self.subTest(yaw=yaw):
                env = self.env(max_steps=1)
                env.reset(seed=42)
                env.physics.reset([1.2, 1.2, yaw])
                env.physics.data.qvel[env.physics.object_dofs[0]] = .4
                mujoco.mj_forward(env.physics.model, env.physics.data)
                obs, rewards, terminated, truncated, infos = env.step(self.zeros(env))
                info = infos["agent_0"]
                self.assertTrue(all(terminated.values()))
                self.assertFalse(any(truncated.values()))
                self.assertTrue(info["success"])
                self.assertEqual(info["end_reason"], "success")
                self.assertAlmostEqual(info["completion_time_s"], .05)
                self.assertEqual(info["reward_components"]["success"], 20.)
                self.assertEqual(len(set(rewards.values())), 1)
                self.assertTrue(all(np.isfinite(a).all() for a in obs.values()))
                self.assertEqual(env.step({}), ({}, {}, {}, {}, {}))

    def test_physics_failure_overrides_crossing_and_preserves_finite_features(self):
        for nonfinite in (False, True):
            with self.subTest(nonfinite=nonfinite):
                env = self.env(max_steps=1)
                original, _ = env.reset(seed=42)
                env.physics.reset([1.2, 1.2, 0.])
                if nonfinite:
                    env.physics.data.qpos[env.physics.object_qpos[0]] = np.nan
                else:
                    env.physics.data.qvel[env.physics.object_dofs[0]] = 6.
                obs, rewards, terminated, truncated, infos = env.step(self.zeros(env))
                self.assertTrue(all(terminated.values()))
                self.assertFalse(any(truncated.values()))
                self.assertFalse(infos["agent_0"]["success"])
                self.assertEqual(infos["agent_0"]["end_reason"], "physics_failure")
                self.assertEqual(rewards["agent_0"], -20.)
                np.testing.assert_array_equal(obs["agent_0"], original["agent_0"])

    def test_partition_blocks_agents_and_logs_obstacle_contact(self):
        env = self.env()
        env.reset(seed=42)
        positions, _ = env.physics.get_agents_state()
        positions[0] = [-.14, 1.]
        env.physics.reset(env.config.object_pose, positions)
        for _ in range(10):
            actions = self.zeros(env)
            actions["agent_0"] = [1., 0.]
            _, _, _, _, infos = env.step(actions)
            self.assertTrue(infos["agent_0"]["stable"], infos["agent_0"]["failure_details"])
        position = env.physics.get_agents_state()[0][0]
        self.assertLessEqual(position[0] + env.physics.radii[0], env.doorway.x_min + .005)
        metrics = infos["agent_0"]["episode_metrics"]
        self.assertGreater(metrics["obstacle_collision_events"], 0)
        self.assertEqual(metrics["wall_collision_events"], 0)

    def test_aligned_object_can_physically_pass_through_opening(self):
        # A feasibility check using real agent forces, not a learned policy or teleportation.
        positions = [[-1.87, y] for y in (-.24, -.08, .08, .24)]
        env = self.env(randomize_agents=False, object_pose=(-1.2, 0., math.pi/2),
                       agent_positions=positions)
        env.reset(seed=42)
        while env.agents:
            _, _, _, _, infos = env.step({name: [1., 0.] for name in env.agents})
            self.assertTrue(infos["agent_0"]["stable"], infos["agent_0"]["failure_details"])
        info = infos["agent_0"]
        self.assertTrue(info["success"])
        self.assertGreaterEqual(info["crossing_clearance_m"], 0.)
        self.assertLess(info["episode_step"], env.max_steps)

    def test_invalid_medium_layouts_are_rejected(self):
        for overrides in (dict(task="hard"), dict(crossing_margin=-1.),
                          dict(model_path="assets/mujoco/arena.xml"),
                          dict(task="easy"), dict(object_pose=(1.5, 0., 0.)),
                          dict(goal_center=(-1.5, 0.)), dict(goal_center=(1.5, 1.))):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.env(**overrides)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "install requirements-training.txt")
    def test_training_evaluation_checkpoint_and_resume(self):
        from src.evaluation.checkpoints import loaded_policy
        from src.evaluation.evaluator import evaluate_policy
        from src.training.config import MAPPOConfig
        from src.training.trainer import Trainer
        cfg = MAPPOConfig(total_steps=5, rollout_steps=4, ppo_epochs=1,
                          actor_hidden_size=8, critic_hidden_size=8, eval_interval_steps=4,
                          eval_episodes=2, checkpoint_interval_steps=4,
                          video_interval_steps=0, tensorboard=False)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            trainer = Trainer(root/"run", cfg, load_config("configs/medium.yaml", max_steps=2))
            self.assertEqual(trainer.train()["environment_steps"], 5)
            manifest = json.loads((root/"run/manifest.json").read_text())
            self.assertEqual(manifest["task"], "medium_v1")
            with loaded_policy(root/"run/checkpoints/latest.pt") as (agent, env_config, payload):
                self.assertEqual(payload["training"]["task"], "medium_v1")
                self.assertEqual(env_config.task, "medium")
                self.assertEqual(agent.dimensions["observation_size"], 39)
                first = evaluate_policy(agent, env_config, episodes=2, seed=123)
                self.assertEqual(first, evaluate_policy(agent, env_config, episodes=2, seed=123))
                self.assertIn("obstacle_collision_events", first["reports"][0]["episode_metrics"])
            resumed = Trainer(root/"resume", replace(cfg, total_steps=7),
                              resume=root/"run/checkpoints/latest.pt")
            self.assertEqual(resumed.environment_config.task, "medium")
            self.assertEqual(resumed.train()["environment_steps"], 7)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "install requirements-training.txt")
    def test_cli_training_evaluation_and_resume_restore_medium(self):
        from src.training.config import MAPPOConfig
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MAPPOConfig(total_steps=2, rollout_steps=2, ppo_epochs=1,
                                 actor_hidden_size=8, critic_hidden_size=8, eval_episodes=1,
                                 video_interval_steps=0, tensorboard=False)
            (root/"mappo.yaml").write_text(yaml.safe_dump(config.to_dict()))
            (root/"environment.yaml").write_text(yaml.safe_dump(
                load_config("configs/medium.yaml", max_steps=2).to_dict()))
            def run(*args):
                result = subprocess.run([sys.executable, *args], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result
            result = run("scripts/train.py", "--run-dir", str(root/"run"),
                         "--config", str(root/"mappo.yaml"), "--env-config", str(root/"environment.yaml"))
            self.assertIn("Medium:", result.stdout)
            checkpoint = str(root/"run/checkpoints/latest.pt")
            run("scripts/evaluate.py", checkpoint, "--device", "cpu", "--json", str(root/"report.json"))
            report = json.loads((root/"report.json").read_text())
            self.assertEqual(report["task"], "medium_v1")
            self.assertIn("doorway", report["reports"][0]["initial_state"])
            run("scripts/train.py", "--resume", checkpoint, "--total-steps", "3",
                "--run-dir", str(root/"resume"))
            self.assertEqual(json.loads((root/"resume/manifest.json").read_text())["task"], "medium_v1")
