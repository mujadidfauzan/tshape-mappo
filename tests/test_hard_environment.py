"""Fixed-target Hard course: geometry, contact transport and saved training."""

from dataclasses import replace
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
from pettingzoo.test import parallel_api_test

from src.envs.config import load_config, load_checkpoint_config
from src.envs.physics import TShapePhysics
from src.envs.tshape_env import TShapeParallelEnv


class HardEnvironmentTests(unittest.TestCase):
    def make_env(self, **overrides):
        env = TShapeParallelEnv(config=replace(load_config("configs/environment_hard.yaml"), **overrides))
        self.addCleanup(env.close)
        return env

    def test_fixed_target_and_randomized_starts_are_valid_replayable_and_widespread(self):
        env = self.make_env()
        p = env.physics
        poses, positions = [], []
        for seed in range(300):
            observations, infos = env.reset(seed=seed)
            obj = p.get_object_state()
            agents, velocities = p.get_agents_state()
            np.testing.assert_array_equal(env.goal_center, [3.25, 1.5])
            np.testing.assert_array_equal(p.data.site_xpos[p.model.site("goal").id, :2], env.goal_center)
            self.assertTrue(env._obstacle_task_is_valid(.15), f"seed={seed}")
            self.assertTrue(p.spawn_is_valid(.04), f"seed={seed}")
            self.assertEqual(p.data.ncon, 0)
            gaps = p.object_surface_distances(agents) - p.radii
            self.assertTrue(np.all(gaps >= .04 - 1e-10))
            self.assertTrue(np.all(gaps <= .15 + 1e-10))
            self.assertTrue(np.all(p.object_corners()[:, 0] < -1.2))
            self.assertGreaterEqual(env._distance_to_goal(), env.config.minimum_goal_distance)
            self.assertFalse(infos["agent_0"]["initial_state"]["goal_randomized"])
            self.assertEqual(observations["agent_0"].shape, (34,))
            self.assertEqual(env.state().shape, (33,))
            np.testing.assert_array_equal(velocities, 0)
            poses.append([*obj["position"], obj["yaw"]])
            positions.append(agents)
        self.assertGreater(np.ptp(np.array(poses)[:, 0]), 1.5)
        self.assertGreater(np.ptp(np.array(poses)[:, 1]), 3)
        self.assertLess(np.min(np.array(poses)[:, 2]), -3)
        self.assertGreater(np.max(np.array(poses)[:, 2]), 3)
        first, infos = env.reset(seed=2026)
        qpos = p.data.qpos.copy()
        env.reset(seed=2027)
        replay, replay_info = env.reset(seed=2026)
        np.testing.assert_array_equal(p.data.qpos, qpos)
        self.assertEqual(infos["agent_0"]["initial_state"], replay_info["agent_0"]["initial_state"])
        for name in first:
            np.testing.assert_array_equal(first[name], replay[name])

    def test_sampled_object_routes_pass_both_aligned_openings_and_enter_goal(self):
        # Only geometric feasibility is checked here. The policy receives no route.
        env = self.make_env()
        p = env.physics
        parking = np.array([[-4, -2.5], [-4, 2.5], [4, -2.5], [4, 2.5]])
        for seed in range(20):
            env.reset(seed=seed)
            obj = p.get_object_state()
            start = obj["position"].copy()
            center = np.array([-2.85, 0.])
            poses = [np.r_[start + a * (center - start), obj["yaw"]] for a in np.linspace(0, 1, 10)]
            poses += [np.r_[center, yaw] for yaw in np.linspace(obj["yaw"], np.pi / 2, 30)]
            waypoints = [center, [2.85, 0], env.goal_center.copy()]
            for previous, following in zip(waypoints, waypoints[1:]):
                for a in np.linspace(0, 1, 40):
                    poses.append(np.r_[(1-a) * np.asarray(previous) + a * np.asarray(following), np.pi / 2])
            for pose in poses:
                p.reset(pose, parking)
                self.assertTrue(p.object_clear_of_obstacles(.01), f"seed={seed}, pose={pose}")
                self.assertTrue(np.all(p.object_corners() >= p.lower))
                self.assertTrue(np.all(p.object_corners() <= p.upper))
            self.assertTrue(env.object_inside_goal())

    def test_each_opening_allows_physical_agent_push_and_blocks_unrotated_object(self):
        env = self.make_env()
        p = env.physics
        for x, gap_y in [(-1.1, 0), (1.1, 0)]:
            with self.subTest(divider=x):
                obj_x = x - 1.2
                agents = np.array([[obj_x - .64, gap_y - .12], [obj_x - .64, gap_y + .12],
                                   [-4, -2.5], [4, 2.5]])
                p.reset([obj_x, gap_y, np.pi / 2], agents)
                self.assertEqual(p.data.ncon, 0)
                action = np.zeros((4, 2))
                action[:2, 0] = 1
                for _ in range(70):
                    report = p.step(action)
                    self.assertTrue(report["stable"], str(report["failure_details"]))
                self.assertGreater(p.object_corners()[:, 0].min(), x + .1)
                p.reset([x - .41, gap_y, 0], np.array([[-4, -2.5], [-4, 2.5], [4, -2.5], [4, 2.5]]))
                p.data.qvel[p.object_dofs[0]] = 1
                mujoco.mj_forward(p.model, p.data)
                contacts = 0
                for _ in range(20):
                    report = p.step(np.zeros((4, 2)))
                    self.assertTrue(report["stable"], str(report["failure_details"]))
                    contacts += report["wall_collision_events"]
                self.assertGreater(contacts, 0)
                self.assertLess(p.get_object_state()["position"][0], x)

    def test_fixed_start_invalid_rooms_goal_and_layout_are_checked(self):
        env = self.make_env()
        env.reset(options={"randomize": False})
        self.assertTrue(env._obstacle_task_is_valid(.005))
        for pose in ([0, 0, 0], [3, 0, 0], [-1.1, -1.5, 0]):
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                env.reset(options={"randomize": False, "object_pose": pose})
        for override in ({"model_path": "assets/mujoco/arena_medium.xml"},
                         {"goal_center": [0, 0]}, {"goal_center": [1.1, 2]},
                         {"goal_half_size": [1.6, .65]}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.make_env(**override)

    def test_collision_geometry_clearances_and_parallel_api(self):
        env = self.make_env(max_steps=8)
        parallel_api_test(env, num_cycles=12)
        p = env.physics
        for x in (-1.1, 1.1):
            # No contact exists, but only 2 cm separates this agent from a wall.
            positions = np.array([[x - .15, 2.0], [-4, 2.5], [4, -2.5], [4, 2.5]])
            p.reset([-2.85, 0, 0], positions)
            self.assertEqual(p.data.ncon, 0)
            self.assertFalse(p.spawn_is_valid(.04))
            p.data.qvel[p.agent_dofs[0, 0]] = 3
            mujoco.mj_forward(p.model, p.data)
            action = np.zeros((4, 2))
            action[0, 0] = 1
            contacts = 0
            for _ in range(10):
                report = p.step(action)
                self.assertTrue(report["stable"], str(report["failure_details"]))
                contacts += report["wall_collision_events"]
            self.assertGreater(contacts, 0)

    def test_physical_parameters_match_easy_and_reward_uses_same_four_components(self):
        env = self.make_env(max_steps=1)
        easy = TShapePhysics()
        for field in ("body_mass", "body_inertia", "dof_damping", "dof_frictionloss",
                      "actuator_ctrlrange", "actuator_gear"):
            np.testing.assert_array_equal(getattr(easy.model, field), getattr(env.physics.model, field))
        self.assertEqual(easy.model.opt.timestep, env.physics.model.opt.timestep)
        for name in ("t_horizontal", "t_vertical", "t_bottom", "agent_0_geom"):
            np.testing.assert_array_equal(easy.model.geom(name).size, env.physics.model.geom(name).size)
        env.reset(options={"randomize": False})
        distance = env._distance_to_goal()
        _, rewards, terminated, truncated, infos = env.step({name: np.zeros(2) for name in env.agents})
        self.assertFalse(any(terminated.values()))
        self.assertTrue(all(truncated.values()))
        parts = infos["agent_0"]["reward_components"]
        current_distance = env._distance_to_goal()
        self.assertEqual(env.config.reward_mode, "pose")
        self.assertAlmostEqual(parts["progress"], env.config.progress_weight * (distance - current_distance))
        self.assertAlmostEqual(parts["distance"], -env.config.distance_weight *
                               current_distance / (current_distance + env.config.distance_scale))
        self.assertAlmostEqual(parts["orientation"], 0)
        self.assertAlmostEqual(rewards["agent_0"], sum(parts.values()))
        self.assertEqual(set(parts), {"distance", "orientation", "progress", "success", "failure"})

    @unittest.skipUnless(find_spec("torch"), "install requirements-training.txt for training integration")
    def test_hard_checkpoint_resume_and_evaluation_keep_fixed_target(self):
        from src.algorithms.mappo import MAPPO
        from src.evaluation.evaluator import evaluate_policy
        from src.training.config import MAPPOConfig
        from src.training.trainer import Trainer
        config = MAPPOConfig(total_steps=4, rollout_steps=2, ppo_epochs=1, actor_hidden_size=8,
                             critic_hidden_size=8, eval_episodes=1, video_interval_steps=0, tensorboard=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Trainer(root / "initial", config, replace(load_config("configs/environment_hard.yaml"),
                max_steps=2)).train()
            checkpoint = root / "initial/checkpoints/latest.pt"
            agent, payload = MAPPO.from_checkpoint(checkpoint)
            saved = payload["training"]["environment_config"]
            self.assertFalse(saved["randomize_goal"])
            self.assertEqual(saved["difficulty"], "hard")
            self.assertIn('name="obstacle_1_lower"', payload["training"]["model_xml"])
            self.assertEqual(agent.dimensions["observation_size"], 34)
            self.assertEqual(agent.dimensions["state_size"], 33)
            resumed = Trainer(root / "resumed", config, resume=checkpoint, resume_step_budget=6)
            self.assertEqual(len(resumed.env.physics.obstacle_ids), 4)
            self.assertEqual(resumed.agent.obs_normalizer.count.item(), agent.obs_normalizer.count.item())
            resumed.train()
            report = evaluate_policy(resumed.agent, resumed.environment_config, episodes=3, seed=200000)
            for episode in report["reports"]:
                self.assertEqual(episode["goal_center"], [3.25, 1.5])
                self.assertFalse(episode["initial_state"]["goal_randomized"])
                self.assertLess(episode["initial_state"]["object_position"][0], -1.2)
            manifest = json.loads((root / "resumed/manifest.json").read_text())
            self.assertFalse(manifest["goal_distribution_changed"])
            self.assertFalse(manifest["normalization_reset"])

    @unittest.skipUnless(find_spec("torch"), "install requirements-training.txt for migration test")
    def test_old_checkpoint_reset_migration_is_applied_and_recorded(self):
        import torch
        from src.algorithms.mappo import MAPPO
        from src.training.config import MAPPOConfig
        from src.training.trainer import Trainer
        config = MAPPOConfig(total_steps=2, rollout_steps=2, ppo_epochs=1, actor_hidden_size=8,
                             critic_hidden_size=8, eval_episodes=1, video_interval_steps=0, tensorboard=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Trainer(root / "initial", config, replace(load_config("configs/environment_medium.yaml"), max_steps=2)).train()
            _, payload = MAPPO.from_checkpoint(root / "initial/checkpoints/latest.pt")
            # Emulate a checkpoint written before the fixed-goal schema existed.
            payload["training"]["environment_config"].pop("randomize_goal")
            payload["training"]["environment_config"]["reset_distribution"] = "arena_wide"
            payload["training"]["environment_config"]["agent_position_noise"] = .02
            checkpoint = root / "old_random_medium.pt"
            torch.save(payload, checkpoint)
            old_env = TShapeParallelEnv(config=load_checkpoint_config(payload["training"]["environment_config"]))
            self.addCleanup(old_env.close)
            for seed in range(30):
                old_env.reset(seed=seed)
                np.testing.assert_array_equal(old_env.goal_center, [1.8, 1.5])
                positions, _ = old_env.physics.get_agents_state()
                self.assertTrue(np.all(old_env.physics.object_surface_distances(positions) - old_env.physics.radii <= .15 + 1e-10))
            resumed = Trainer(root / "resumed", config, resume=checkpoint, resume_step_budget=4)
            self.assertFalse(resumed.env.config.randomize_goal)
            self.assertEqual(resumed.agent.obs_normalizer.count.item(), 0)
            self.assertEqual(resumed.agent.state_normalizer.count.item(), 0)
            self.assertEqual(resumed.episode_stats.summary(), {"window_episodes": 0})
            manifest = json.loads((root / "resumed/manifest.json").read_text())
            self.assertTrue(manifest["goal_distribution_changed"])
            self.assertTrue(manifest["reset_distribution_changed"])
            self.assertTrue(manifest["normalization_reset"])
            resumed.train()
            for episode in map(json.loads, (root / "resumed/episodes.jsonl").read_text().splitlines()):
                self.assertEqual(episode["initial_state"]["goal_center"], [1.8, 1.5])

    def test_two_openings_are_aligned_and_agents_push_through_both_continuously(self):
        env = self.make_env()
        p = env.physics
        for prefix in ("obstacle_0", "obstacle_1"):
            upper, lower = p.model.geom(f"{prefix}_upper"), p.model.geom(f"{prefix}_lower")
            self.assertAlmostEqual(upper.pos[1] - upper.size[1], .4)
            self.assertAlmostEqual(lower.pos[1] + lower.size[1], -.4)
        p.reset([-2.3, 0, np.pi / 2], np.array([[-2.94, -.12], [-2.94, .12], [-4, -2.5], [4, 2.5]]))
        actions = np.zeros((4, 2))
        actions[:2, 0] = 1
        for _ in range(140):
            report = p.step(actions)
            self.assertTrue(report["stable"], str(report["failure_details"]))
        self.assertGreater(p.object_corners()[:, 0].min(), 1.2)

    @unittest.skipUnless(find_spec("torch"), "install requirements-training.txt for training validation")
    def test_training_refuses_an_obsolete_unaligned_hard_snapshot(self):
        import xml.etree.ElementTree as ET
        from src.training.config import MAPPOConfig
        from src.training.trainer import Trainer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = ET.parse("assets/mujoco/arena_hard.xml")
            for geom in tree.findall("./worldbody/geom"):
                if geom.get("name", "").startswith("obstacle_"):
                    pos = np.fromstring(geom.get("pos"), sep=" ")
                    size = np.fromstring(geom.get("size"), sep=" ")
                    bottom, top = ((.6, 2.9) if geom.get("name").endswith("upper") else (-2.9, -.2))
                    pos[1] = (bottom + top) / 2
                    size[1] = (top - bottom) / 2
                    geom.set("pos", " ".join(map(str, pos)))
                    geom.set("size", " ".join(map(str, size)))
            model = root / "unaligned.xml"
            tree.write(model)
            env = replace(load_config("configs/environment_hard.yaml"), model_path=str(model))
            cfg = MAPPOConfig(total_steps=2, video_interval_steps=0, tensorboard=False)
            with self.assertRaisesRegex(ValueError, "aligned openings"):
                Trainer(root / "run", cfg, env)


if __name__ == "__main__":
    unittest.main()
