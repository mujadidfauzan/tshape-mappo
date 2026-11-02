"""Fixed-layout, contact, termination and Parallel API checks."""

from dataclasses import replace
import unittest

import mujoco
import numpy as np
from pettingzoo.test import parallel_api_test

from src.envs.config import load_config
from src.envs.scripted import scripted_actions
from src.envs.tshape_env import TShapeParallelEnv


class EasyTests(unittest.TestCase):
    def env(self, **overrides):
        env=TShapeParallelEnv(replace(load_config(),**overrides))
        self.addCleanup(env.close)
        return env

    def zeros(self,env):
        return {n:np.zeros(2,np.float32) for n in env.agents}

    def test_every_seed_and_episode_restores_the_exact_layout(self):
        env=self.env()
        first,_=env.reset(seed=0)
        qpos=env.physics.data.qpos.copy()
        for seed in [1,42,999]:
            env.step({n:env.action_space(n).sample() for n in env.agents})
            obs,infos=env.reset(seed=seed)
            np.testing.assert_array_equal(env.physics.data.qpos,qpos)
            np.testing.assert_array_equal(env.physics.data.qvel,0)
            initial=infos["agent_0"]["initial_state"]
            self.assertEqual(initial["goal_center"],list(env.config.goal_center))
            self.assertTrue(all(v<0 for v in initial["object_position"]))
            self.assertTrue(all(v>0 for v in initial["goal_center"]))
            for n in env.agents:
                np.testing.assert_array_equal(obs[n],first[n])
        self.assertEqual(first["agent_0"].shape,(34,))
        self.assertEqual(env.state().shape,(33,))
        self.assertFalse(env.object_inside_goal())

    def test_removed_modes_are_rejected(self):
        for values in [{"randomize_reset":False},{"difficulty":"medium"},{"num_agents":8}]:
            with self.subTest(values=values),self.assertRaises(ValueError):
                load_config(values)
        env=self.env()
        env.reset()
        original=env.physics.data.qpos.copy()
        env.reset(options={"randomize":True})
        np.testing.assert_array_equal(env.physics.data.qpos,original)

    def test_parallel_api(self):
        parallel_api_test(self.env(max_steps=3),num_cycles=5)

    def test_reference_controller_physically_completes_fixed_easy(self):
        env=self.env()
        env.reset()
        while env.agents:
            _,_,_,_,infos=env.step(scripted_actions(env))
            self.assertTrue(infos["agent_0"]["stable"],infos["agent_0"]["failure_details"])
        info=infos["agent_0"]
        self.assertTrue(info["success"])
        self.assertLess(info["episode_step"],env.max_steps)
        self.assertGreater(info["episode_metrics"]["object_path_length_m"],2.5)
        self.assertLess(info["episode_metrics"]["max_penetration_m"],.005)

    def test_timeout_and_shared_reward(self):
        env=self.env(max_steps=2)
        env.reset()
        for _ in range(2):
            obs,rewards,term,trunc,infos=env.step(self.zeros(env))
            self.assertEqual(len(set(rewards.values())),1)
            self.assertAlmostEqual(rewards["agent_0"],sum(infos["agent_0"]["reward_components"].values()))
        self.assertFalse(any(term.values()))
        self.assertTrue(all(trunc.values()))
        self.assertEqual(env.agents,[])
        self.assertEqual(env.step({}),({},{},{},{},{}))

    def test_complete_footprint_and_settling_are_required(self):
        env=self.env(max_steps=10)
        env.reset()
        gx,gy=env.config.goal_center
        env.physics.reset([gx,gy-.07,0],env.config.agent_positions)
        self.assertFalse(env.object_inside_goal())
        env.physics.reset([gx,gy,0],env.config.agent_positions)
        self.assertTrue(env.object_inside_goal())
        env.physics.data.qvel[env.physics.object_dofs[0]]=.5
        mujoco.mj_forward(env.physics.model,env.physics.data)
        _,_,term,_,info=env.step(self.zeros(env))
        self.assertFalse(any(term.values()))
        self.assertEqual(info["agent_0"]["goal_hold_steps"],0)
        env.physics.reset([gx,gy,0],env.config.agent_positions)
        for _ in range(5):
            _,_,term,_,info=env.step(self.zeros(env))
        self.assertTrue(all(term.values()))
        self.assertTrue(info["agent_0"]["success"])

    def test_invalid_actions_do_not_advance_time(self):
        env=self.env()
        env.reset()
        with self.assertRaises(ValueError):
            env.step({})
        with self.assertRaises(ValueError):
            env.step({**self.zeros(env),"agent_0":[np.nan,0]})
        self.assertEqual(env.steps,0)

    def test_nonfinite_physics_ends_with_finite_cached_features(self):
        env=self.env()
        original,_=env.reset()
        env.physics.data.qpos[env.physics.object_qpos[0]]=np.nan
        obs,reward,term,trunc,info=env.step(self.zeros(env))
        self.assertTrue(all(term.values()))
        self.assertFalse(any(trunc.values()))
        self.assertEqual(reward["agent_0"],-20)
        self.assertEqual(info["agent_0"]["failure_reason"],"non_finite_state")
        np.testing.assert_array_equal(obs["agent_0"],original["agent_0"])
        self.assertTrue(np.isfinite(env.state()).all())
