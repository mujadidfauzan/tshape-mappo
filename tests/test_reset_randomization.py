"""Arena-wide seeded reset coverage, collision clearance and goal consistency."""

import unittest

import numpy as np

from src.envs.config import EnvironmentConfig, load_checkpoint_config
from src.envs.observations import ACTOR_FEATURES
from src.envs.tshape_env import TShapeParallelEnv


class ResetRandomizationTests(unittest.TestCase):
    def make_env(self, **config):
        env = TShapeParallelEnv(config=config or None)
        self.addCleanup(env.close)
        return env

    def test_default_reset_covers_arena_and_full_yaw_without_overlaps(self):
        env = self.make_env()
        p = env.physics
        targets, objects, yaws, agents = [], [], [], []
        for seed in range(200):
            env.reset(seed=seed)
            state = p.get_object_state()
            positions, velocities = p.get_agents_state()
            corners = p.object_corners()
            margin = env.config.arena_spawn_margin
            self.assertTrue(p.spawn_is_valid(clearance=env.config.spawn_clearance), msg=f"seed={seed}")
            self.assertEqual(p.data.ncon, 0)
            self.assertTrue(np.all(corners >= p.lower + margin - 1e-9))
            self.assertTrue(np.all(corners <= p.upper - margin + 1e-9))
            self.assertTrue(np.all(positions - p.radii[:, None] >= p.lower + margin - 1e-9))
            self.assertTrue(np.all(positions + p.radii[:, None] <= p.upper - margin + 1e-9))
            self.assertTrue(np.all(env.goal_center - env.goal_half_size >= p.lower + margin - 1e-9))
            self.assertTrue(np.all(env.goal_center + env.goal_half_size <= p.upper - margin + 1e-9))
            self.assertGreaterEqual(np.linalg.norm(state["position"] - env.goal_center), env.config.minimum_goal_distance)
            self.assertFalse(env.object_inside_goal())
            np.testing.assert_array_equal(velocities, 0)
            targets.append(env.goal_center.copy())
            objects.append(state["position"])
            yaws.append(state["yaw"])
            agents.append(positions)
        self.assertTrue(np.all(np.ptp(targets, axis=0) > 3.0))
        self.assertTrue(np.all(np.ptp(objects, axis=0) > 3.0))
        self.assertTrue(np.all(np.ptp(agents, axis=0) > 4.0))
        self.assertLess(min(yaws), -3.0)
        self.assertGreater(max(yaws), 3.0)

    def test_seed_replays_target_object_agents_and_observations(self):
        env = self.make_env()
        first, infos = env.reset(seed=2026)
        qpos = env.physics.data.qpos.copy()
        initial = infos["agent_0"]["initial_state"]
        env.reset(seed=2027)
        self.assertFalse(np.array_equal(env.goal_center, initial["goal_center"]))
        replay, replay_infos = env.reset(seed=2026)
        np.testing.assert_array_equal(qpos, env.physics.data.qpos)
        self.assertEqual(initial, replay_infos["agent_0"]["initial_state"])
        for name in env.agents:
            np.testing.assert_array_equal(first[name], replay[name])

    def test_target_marker_actor_critic_and_info_use_the_sampled_goal(self):
        env = self.make_env(max_steps=1)
        observations, infos = env.reset(seed=42)
        offset = 0
        for name, size in ACTOR_FEATURES:
            if name == "goal_relative":
                goal_slice = slice(offset, offset + size)
                break
            offset += size
        positions, _ = env.physics.get_agents_state()
        for i, name in enumerate(env.agents):
            np.testing.assert_allclose(observations[name][goal_slice], env.goal_center - positions[i], atol=1e-6)
        np.testing.assert_allclose(env.state()[7:9], env.goal_center, atol=1e-6)
        marker = env.physics.model.site("goal")
        np.testing.assert_array_equal(marker.pos[:2], env.goal_center)
        np.testing.assert_array_equal(env.physics.data.site_xpos[marker.id, :2], env.goal_center)
        initial = infos["agent_0"]["initial_state"]
        _, _, _, _, final = env.step({name: np.zeros(2) for name in env.agents})
        self.assertEqual(final["agent_0"]["initial_state"], initial)
        self.assertEqual(final["agent_0"]["goal_center"], env.goal_center.tolist())
        final["agent_0"]["initial_state"]["goal_center"][0] = 999
        self.assertNotEqual(final["agent_1"]["initial_state"]["goal_center"][0], 999)

    def test_old_checkpoint_keeps_legacy_reset_for_evaluation(self):
        config = EnvironmentConfig().to_dict()
        config.pop("reset_distribution")
        env = TShapeParallelEnv(config=load_checkpoint_config(config))
        self.addCleanup(env.close)
        for seed in range(10):
            env.reset(seed=seed)
            np.testing.assert_array_equal(env.goal_center, config["goal_center"])
            position = env.physics.get_object_state()["position"]
            self.assertTrue(np.all(np.abs(position) <= .1))
            self.assertLessEqual(abs(env.physics.get_object_state()["yaw"]), .15)


if __name__ == "__main__":
    unittest.main()
