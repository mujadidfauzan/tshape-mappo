"""M2 episode/API regression checks. Prepared for manual execution by the user."""

import unittest

import mujoco
import numpy as np
from pettingzoo.test import parallel_api_test

from src.envs.observations import OBSERVATION_SIZE, STATE_SIZE
from src.envs.tshape_env import TShapeParallelEnv


class EnvironmentTests(unittest.TestCase):
    def make_env(self, **config):
        env = TShapeParallelEnv(config={"randomize_reset": False, **config})
        self.addCleanup(env.close)
        return env

    def zeros(self, env):
        return {name: np.zeros(2, dtype=np.float32) for name in env.agents}

    def test_parallel_api(self):
        env = self.make_env(max_steps=8)
        parallel_api_test(env, num_cycles=12)

    def test_reset_returns_distinct_float32_observations_and_global_state(self):
        env = self.make_env()
        observations, infos = env.reset(seed=42)
        self.assertEqual(set(observations), set(env.possible_agents))
        self.assertEqual(set(infos), set(env.possible_agents))
        for name, observation in observations.items():
            self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
            self.assertEqual(observation.dtype, np.float32)
            self.assertTrue(env.observation_space(name).contains(observation))
        self.assertFalse(np.array_equal(observations["agent_0"], observations["agent_1"]))
        self.assertEqual(env.state().shape, (STATE_SIZE,))
        self.assertTrue(env.state_space.contains(env.state()))
        self.assertEqual(len(env.physics.object_geom_ids), 3)

    def test_seed_reproduces_spawn_actions_and_transitions(self):
        a = self.make_env(randomize_reset=True)
        b = self.make_env(randomize_reset=True)
        obs_a, _ = a.reset(seed=123)
        obs_b, _ = b.reset(seed=123)
        np.testing.assert_array_equal(a.physics.data.qpos, b.physics.data.qpos)
        self.assertTrue(a.physics.spawn_is_valid(clearance=0.005))
        initial = a.physics.data.qpos.copy()
        actions_a = {name: a.action_space(name).sample() for name in a.agents}
        actions_b = {name: b.action_space(name).sample() for name in b.agents}
        for name in a.agents:
            np.testing.assert_array_equal(obs_a[name], obs_b[name])
            np.testing.assert_array_equal(actions_a[name], actions_b[name])
        result_a, result_b = a.step(actions_a), b.step(actions_b)
        for name in a.possible_agents:
            np.testing.assert_array_equal(result_a[0][name], result_b[0][name])
            self.assertEqual(result_a[1][name], result_b[1][name])
        a.reset(seed=123)
        np.testing.assert_array_equal(a.physics.data.qpos, initial)
        a.reset(seed=124)
        self.assertFalse(np.array_equal(a.physics.data.qpos, initial))

    def test_invalid_actions_do_not_advance_the_episode(self):
        env = self.make_env()
        env.reset(seed=1)
        for actions in ({}, {**self.zeros(env), "extra": np.zeros(2)},
                        {**self.zeros(env), "agent_0": [np.nan, 0]},
                        {**self.zeros(env), "agent_0": [0, 0, 0]}):
            with self.subTest(actions=str(actions)):
                with self.assertRaises(ValueError):
                    env.step(actions)
                self.assertEqual(env.steps, 0)
                self.assertEqual(env.physics.data.time, 0)

    def test_step_before_reset_is_an_error(self):
        env = self.make_env()
        with self.assertRaises(RuntimeError):
            env.step({})

    def test_out_of_range_actions_are_clipped_to_xml_force_limits(self):
        env = self.make_env()
        env.reset(seed=42)
        actions = {name: np.array([2.0, -2.0]) for name in env.agents}
        env.step(actions)
        controls = env.physics.data.ctrl[env.physics.motor_ids]
        np.testing.assert_array_equal(controls[:, 0], env.physics.motor_limits[:, 0, 1])
        np.testing.assert_array_equal(controls[:, 1], env.physics.motor_limits[:, 1, 0])

    def test_zero_actions_have_shared_time_penalty_and_time_limit(self):
        env = self.make_env(max_steps=1)
        env.reset(seed=42)
        observations, rewards, terminated, truncated, infos = env.step(self.zeros(env))
        self.assertEqual(set(observations), set(env.possible_agents))
        self.assertEqual(set(rewards.values()), {-env.config.time_penalty})
        self.assertFalse(any(terminated.values()))
        self.assertTrue(all(truncated.values()))
        self.assertEqual(env.agents, [])
        self.assertEqual(infos["agent_0"]["end_reason"], "time_limit")
        self.assertAlmostEqual(infos["agent_0"]["elapsed_time_s"], 0.05)
        self.assertEqual(env.step({}), ({}, {}, {}, {}, {}))
        self.assertAlmostEqual(env.state()[-2], 0.0)

    def test_success_requires_a_slow_hold_and_takes_priority_over_time_limit(self):
        env = self.make_env(max_steps=3, success_hold_steps=3)
        env.reset(seed=7, options={"randomize": False, "object_pose": [1.8, 1.5, 0]})
        self.assertTrue(env.object_inside_goal())
        for i in range(3):
            _, rewards, terminated, truncated, infos = env.step(self.zeros(env))
            self.assertEqual(infos["agent_0"]["goal_hold_steps"], i + 1)
            self.assertEqual(all(terminated.values()), i == 2)
            self.assertFalse(any(truncated.values()))
        self.assertEqual(infos["agent_0"]["end_reason"], "success")
        self.assertAlmostEqual(rewards["agent_0"], env.config.success_reward - env.config.time_penalty)
        self.assertEqual(env.agents, [])

    def test_short_end_bar_counts_towards_goal_containment(self):
        env = self.make_env()
        env.reset(seed=7, options={"randomize": False, "object_pose": [1.8, 1.43, 0]})
        corners = env.physics.object_corners()
        lower, upper = env.goal_center - env.goal_half_size, env.goal_center + env.goal_half_size
        # Large bar + stem fit, but the short bar protrudes below the goal.
        self.assertTrue(np.all(corners[:8] >= lower) and np.all(corners[:8] <= upper))
        self.assertFalse(env.object_inside_goal())
        _, _, terminated, _, infos = env.step(self.zeros(env))
        self.assertFalse(any(terminated.values()))
        self.assertEqual(infos["agent_0"]["goal_hold_steps"], 0)

    def test_fast_motion_through_goal_does_not_succeed(self):
        env = self.make_env(success_hold_steps=1)
        env.reset(options={"randomize": False, "object_pose": [1.8, 1.5, 0]})
        env.physics.data.qvel[env.physics.object_dofs[0]] = 0.5
        mujoco.mj_forward(env.physics.model, env.physics.data)
        _, _, terminated, _, infos = env.step(self.zeros(env))
        self.assertFalse(any(terminated.values()))
        self.assertEqual(infos["agent_0"]["goal_hold_steps"], 0)

    def test_invalid_spawn_is_rejected(self):
        env = self.make_env()
        with self.assertRaises(ValueError):
            env.reset(options={"randomize": False, "agent_positions": np.zeros((4, 2))})

    def test_nonfinite_physics_terminates_with_finite_reward_and_observations(self):
        env = self.make_env()
        observations, _ = env.reset(seed=7)
        env.physics.data.qpos[env.physics.object_qpos[0]] = np.nan
        final_obs, rewards, terminated, truncated, infos = env.step(self.zeros(env))
        self.assertTrue(all(terminated.values()))
        self.assertFalse(any(truncated.values()))
        self.assertEqual(infos["agent_0"]["failure_reason"], "non_finite_state")
        for name in env.possible_agents:
            self.assertTrue(np.isfinite(rewards[name]))
            np.testing.assert_array_equal(final_obs[name], observations[name])
        self.assertTrue(np.isfinite(env.state()).all())

    def test_escaped_object_is_a_failure_even_without_wall_contact(self):
        env = self.make_env()
        env.reset()
        env.physics.data.qpos[env.physics.object_qpos[0]] = 4.0
        mujoco.mj_forward(env.physics.model, env.physics.data)
        _, rewards, terminated, _, infos = env.step(self.zeros(env))
        self.assertTrue(all(terminated.values()))
        self.assertEqual(infos["agent_0"]["failure_reason"], "excessive_penetration")
        self.assertAlmostEqual(rewards["agent_0"], -env.config.failure_penalty - env.config.time_penalty)

    def test_penetration_failure_identifies_the_contact_and_actual_state(self):
        env = self.make_env()
        observations, _ = env.reset(seed=42)
        p = env.physics
        stem = p.model.geom("t_vertical")
        # Put agent 2 six millimeters into the left face of the stem. This
        # intentional invalid state should fail before the next substep.
        position = np.array([stem.pos[0] - stem.size[0] - p.radii[2] + 0.006, -0.26])
        p.data.qpos[p.agent_qpos[2]] = position - p.model.body_pos[p.agent_bodies[2], :2]
        mujoco.mj_forward(p.model, p.data)
        actual_qpos = p.data.qpos.copy()
        final_obs, _, terminated, _, infos = env.step(self.zeros(env))
        self.assertTrue(all(terminated.values()))
        details = infos["agent_0"]["failure_details"]
        self.assertEqual(infos["agent_0"]["failure_reason"], "excessive_penetration")
        np.testing.assert_array_equal(details["qpos"], actual_qpos)
        contact = details["contacts"][0]
        self.assertEqual(set(contact["geoms"]), {"agent_2_geom", "t_vertical"})
        self.assertGreater(contact["penetration_m"], env.config.max_penetration)
        np.testing.assert_array_equal(final_obs["agent_0"], observations["agent_0"])
        details["contacts"][0]["geoms"].append("modified")
        self.assertNotIn("modified", infos["agent_1"]["failure_details"]["contacts"][0]["geoms"])

    def test_scripted_seed_46_does_not_exceed_contact_tolerance(self):
        # Regression for results/m2_validation.json (step 185, 5.065 mm).
        # This test is intentionally a full episode, not an isolated target check.
        from src.envs.scripted import scripted_actions
        env = self.make_env(randomize_reset=True, reset_distribution="legacy")
        env.reset(seed=46)
        info = None
        while env.agents:
            _, _, _, _, infos = env.step(scripted_actions(env))
            info = infos["agent_0"]
            self.assertTrue(info["stable"], msg=str(info.get("failure_details")))
        self.assertIsNotNone(info)
        self.assertTrue(info["success"], msg=str(info))

    def test_all_four_scripted_controls_are_bounded_and_finite(self):
        from src.envs.scripted import scripted_actions
        env = self.make_env()
        env.reset(seed=42)
        actions = scripted_actions(env)
        self.assertEqual(set(actions), set(env.possible_agents))
        for name, action in actions.items():
            self.assertTrue(env.action_space(name).contains(action))
            self.assertGreater(np.linalg.norm(action), 0)


if __name__ == "__main__":
    unittest.main()
