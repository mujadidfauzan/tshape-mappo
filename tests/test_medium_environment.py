"""Medium geometry, randomized tasks, contact dynamics and training integration."""

from dataclasses import replace
import json
from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
from pettingzoo.test import parallel_api_test

from src.envs.config import load_config
from src.envs.tshape_env import TShapeParallelEnv


class MediumEnvironmentTests(unittest.TestCase):
    def make_env(self, **overrides):
        config = replace(load_config("configs/environment_medium.yaml"), **overrides)
        env = TShapeParallelEnv(config=config)
        self.addCleanup(env.close)
        return env

    def test_fixed_goal_random_object_full_yaw_and_nearby_agents(self):
        env = self.make_env()
        goals, poses, agents = [], [], []
        for seed in range(300):
            observations, infos = env.reset(seed=seed)
            p = env.physics
            obj = p.get_object_state()
            positions, velocities = p.get_agents_state()
            self.assertTrue(p.spawn_is_valid(env.config.spawn_clearance), f"seed={seed}")
            self.assertEqual(p.data.ncon, 0)
            gaps = p.object_surface_distances(positions) - p.radii
            self.assertTrue(np.all(gaps >= .04 - 1e-10))
            self.assertTrue(np.all(gaps <= .15 + 1e-10))
            self.assertTrue(env._obstacle_task_is_valid(env.config.arena_spawn_margin))
            self.assertLess(obj["position"][0] * env.goal_center[0], 0)
            self.assertGreaterEqual(env._distance_to_goal(), env.config.minimum_goal_distance)
            self.assertFalse(env.object_inside_goal())
            self.assertEqual(observations["agent_0"].shape, (34,))
            self.assertEqual(env.state().shape, (33,))
            np.testing.assert_array_equal(velocities, 0)
            np.testing.assert_array_equal(p.data.site_xpos[p.model.site("goal").id, :2], env.goal_center)
            self.assertEqual(infos["agent_0"]["difficulty"], "medium")
            self.assertFalse(infos["agent_0"]["initial_state"]["goal_randomized"])
            goals.append(env.goal_center.copy())
            poses.append([*obj["position"], obj["yaw"]])
            agents.append(positions)
        np.testing.assert_array_equal(goals, np.tile(env.config.goal_center, (300, 1)))
        self.assertGreater(np.ptp(np.array(poses)[:, 0]), 1)
        self.assertGreater(np.ptp(np.array(poses)[:, 1]), 3)
        self.assertLess(np.min(np.array(poses)[:, 2]), -3)
        self.assertGreater(np.max(np.array(poses)[:, 2]), 3)
        first, initial = env.reset(seed=42)
        qpos = env.physics.data.qpos.copy()
        env.reset(seed=43)
        replay, infos = env.reset(seed=42)
        np.testing.assert_array_equal(qpos, env.physics.data.qpos)
        self.assertEqual(initial["agent_0"]["initial_state"], infos["agent_0"]["initial_state"])
        for name in first:
            np.testing.assert_array_equal(first[name], replay[name])

    def test_fixed_reset_and_same_side_or_wall_spawns(self):
        env = self.make_env()
        env.reset(options={"randomize": False})
        self.assertTrue(env._obstacle_task_is_valid(.005))
        for pose in ([1.5, 0, 0], [0, 1.5, 0]):
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                env.reset(options={"randomize": False, "object_pose": pose})
        with self.assertRaises(ValueError):
            env.reset(options={"randomize": False,
                "agent_positions": [[0, 1.5], [-2, 2], [2, -2], [2, 2]]})

    def test_obstacle_clearance_rejects_nearby_agents_and_rotated_compound_boxes(self):
        env = self.make_env()
        p = env.physics
        positions = np.array([[-.15, 1.5], [-2, 2], [2, -2], [2, 2]])
        p.reset([-1.5, 0, 0], positions)
        self.assertEqual(p.data.ncon, 0)
        self.assertTrue(p.spawn_is_valid(.005))
        self.assertFalse(p.spawn_is_valid(.04))  # 2 cm agent-to-divider gap.
        positions[0] = [-2, -2]
        p.reset([-.42, 0, 0], positions)
        self.assertEqual(p.data.ncon, 0)
        self.assertTrue(p.object_clear_of_obstacles(.005))
        self.assertFalse(p.object_clear_of_obstacles(.04))
        for yaw in np.linspace(-np.pi, np.pi, 25):
            p.reset([0, 1.5, yaw], positions)
            self.assertFalse(p.object_clear_of_obstacles(.005))

    def test_sampled_object_poses_have_a_geometric_route_through_the_opening(self):
        # Pose sweeps are geometry probes, not policy actions or reward guidance.
        env = self.make_env()
        p = env.physics
        parking = np.array([[-2.7, -2.7], [-2.7, 2.7], [2.7, -2.7], [2.7, 2.7]])
        for seed in range(20):
            env.reset(seed=seed)
            obj = p.get_object_state()
            origin = obj["position"].copy()
            side = int(origin[0] > 0)
            lower, upper = env._rooms[side]
            center = (lower + upper) / 2
            destination = env.goal_center.copy()
            # Translate into open room center, rotate, cross, then enter goal.
            poses = [np.r_[origin + alpha * (center - origin), obj["yaw"]]
                     for alpha in np.linspace(0, 1, 10)]
            poses += [np.r_[center, yaw] for yaw in np.linspace(obj["yaw"], np.pi / 2, 30)]
            opposite = (env._rooms[1-side][0] + env._rooms[1-side][1]) / 2
            poses += [np.r_[center + alpha * (opposite - center), np.pi / 2]
                      for alpha in np.linspace(0, 1, 50)]
            poses += [np.r_[opposite + alpha * (destination - opposite), np.pi / 2]
                      for alpha in np.linspace(0, 1, 10)]
            for pose in poses:
                p.reset(pose, parking)
                self.assertTrue(p.object_clear_of_obstacles(.01), f"seed={seed}, pose={pose}")
                self.assertTrue(np.all(p.object_corners() >= p.lower))
                self.assertTrue(np.all(p.object_corners() <= p.upper))
            self.assertTrue(env.object_inside_goal())

    def test_agents_can_physically_push_rotated_object_through_opening(self):
        env = self.make_env()
        p = env.physics
        p.reset([-1.2, 0, np.pi / 2],
                np.array([[-1.84, -.12], [-1.84, .12], [-2, 2], [2, -2]]))
        actions = np.zeros((4, 2))
        actions[:2, 0] = 1
        peak = 0
        for _ in range(80):
            report = p.step(actions)
            peak = max(peak, report["max_penetration_m"])
            self.assertTrue(report["stable"], str(report["failure_details"]))
        self.assertGreater(p.object_corners()[:, 0].min(), .1)
        self.assertLess(peak, env.config.max_penetration)

    def test_unrotated_object_is_blocked_and_internal_wall_contacts_are_counted(self):
        env = self.make_env()
        p = env.physics
        parking = np.array([[-2, -2], [-2, 2], [2, -2], [2, 2]], dtype=float)
        p.reset([-.41, 0, 0], parking)
        p.data.qvel[p.object_dofs[0]] = 1
        mujoco.mj_forward(p.model, p.data)
        contacts = 0
        for _ in range(20):
            report = p.step(np.zeros((4, 2)))
            self.assertTrue(report["stable"], str(report["failure_details"]))
            contacts += report["wall_collision_events"]
        self.assertLess(p.get_object_state()["position"][0], 0)
        self.assertGreater(contacts, 0)
        # A maximum-force agent impact is checked against the same 5 mm limit.
        parking[0] = [-.15, 1.5]
        p.reset([-1.5, 0, 0], parking)
        p.data.qvel[p.agent_dofs[0, 0]] = 3
        mujoco.mj_forward(p.model, p.data)
        actions = np.zeros((4, 2))
        actions[0, 0] = 1
        contacts = 0
        for _ in range(10):
            report = p.step(actions)
            self.assertTrue(report["stable"], str(report["failure_details"]))
            contacts += report["wall_collision_events"]
        self.assertGreater(contacts, 0)
        self.assertLess(p.get_agents_state()[0][0, 0], -.1)

    def test_parallel_api_observations_state_and_reward_match_easy_contract(self):
        env = self.make_env(max_steps=8)
        parallel_api_test(env, num_cycles=12)
        observations, _ = env.reset(options={"randomize": False})
        easy = TShapeParallelEnv(config={"max_steps": 8, "randomize_reset": False})
        self.addCleanup(easy.close)
        pose = env.physics.get_object_state()
        easy_obs, _ = easy.reset(options={"randomize": False,
            "object_pose": [*pose["position"], pose["yaw"]],
            "agent_positions": env.physics.get_agents_state()[0]})
        for name in observations:
            np.testing.assert_array_equal(observations[name], easy_obs[name])
        np.testing.assert_array_equal(env.state(), easy.state())
        actions = {name: np.zeros(2) for name in env.agents}
        medium_step, easy_step = env.step(actions), easy.step(actions)
        self.assertEqual(medium_step[1], easy_step[1])
        self.assertEqual(medium_step[4]["agent_0"]["reward_components"],
                         easy_step[4]["agent_0"]["reward_components"])

    def test_layout_mismatch_and_obstructed_or_oversized_goal_fail_early(self):
        for changes in ({"model_path": "assets/mujoco/arena.xml"},
                        {"difficulty": "easy"}, {"goal_center": [0, 1.5]},
                        {"goal_half_size": [1.4, .65]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.make_env(**changes)

    @unittest.skipUnless(find_spec("torch"), "install requirements-training.txt for training integration")
    def test_training_checkpoint_resume_and_evaluation_preserve_medium_layout(self):
        from src.algorithms.mappo import MAPPO
        from src.evaluation.evaluator import evaluate_policy
        from src.training.config import MAPPOConfig
        from src.training.trainer import Trainer
        config = MAPPOConfig(total_steps=4, rollout_steps=2, ppo_epochs=1,
                             actor_hidden_size=8, critic_hidden_size=8, eval_episodes=1,
                             video_interval_steps=0, tensorboard=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = Trainer(root / "initial", config, replace(
                load_config("configs/environment_medium.yaml"), max_steps=2))
            trainer.train()
            agent, payload = MAPPO.from_checkpoint(root / "initial/checkpoints/latest.pt")
            self.assertEqual(payload["training"]["environment_config"]["difficulty"], "medium")
            self.assertIn('name="obstacle_upper"', payload["training"]["model_xml"])
            self.assertEqual(agent.dimensions["observation_size"], 34)
            self.assertEqual(agent.dimensions["state_size"], 33)
            resumed = Trainer(root / "resumed", config, resume=root / "initial/checkpoints/latest.pt",
                              resume_step_budget=6)
            self.assertEqual(resumed.env.config.difficulty, "medium")
            self.assertEqual(len(resumed.env.physics.obstacle_ids), 2)
            resumed.train()
            report = evaluate_policy(resumed.agent, resumed.environment_config, episodes=2, seed=200000)
            for episode in report["reports"]:
                self.assertEqual(episode["difficulty"], "medium")
                initial = episode["initial_state"]
                self.assertLess(initial["goal_center"][0] * initial["object_position"][0], 0)
            self.assertEqual({p.name for p in (root / "resumed/checkpoints").iterdir()},
                             {"best.pt", "latest.pt"})
            manifest = json.loads((root / "resumed/manifest.json").read_text())
            self.assertEqual(manifest["environment"]["difficulty"], "medium")


if __name__ == "__main__":
    unittest.main()
