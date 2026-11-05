"""Fixed/randomized spawns, variable populations, contacts and Parallel API."""

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
        values = dict(randomize_agents=False)
        values.update(overrides)
        env=TShapeParallelEnv(replace(load_config(),**values))
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
        for values in [{"randomize_reset":False},{"difficulty":"medium"}]:
            with self.subTest(values=values),self.assertRaises(ValueError):
                load_config(values)
        env=self.env()
        env.reset()
        original=env.physics.data.qpos.copy()
        env.reset(options={"randomize":True})
        np.testing.assert_array_equal(env.physics.data.qpos,original)

    def test_parallel_api(self):
        for count in [1,2,4,8,16]:
            with self.subTest(num_agents=count):
                parallel_api_test(self.env(max_steps=3,num_agents=count,randomize_agents=True),num_cycles=5)

    def test_random_spawns_are_safe_near_the_rotated_object_for_all_counts(self):
        for count in [1,2,4,8,16]:
            with self.subTest(num_agents=count):
                env=self.env(num_agents=count,randomize_agents=True,object_pose=(-1.5,-1.5,.4))
                self.assertEqual(env.physics.model.nu,2*count)
                self.assertEqual(env.physics.model.nq,3+2*count)
                for seed in range(20):
                    obs,infos=env.reset(seed=seed)
                    self.assertEqual(len(obs),count)
                    self.assertEqual(env.state().shape,(17+4*count,))
                    self.assertTrue(all(v.shape==(18+4*count,) for v in obs.values()))
                    self.assertTrue(all(env.observation_space(n).contains(obs[n]) for n in obs))
                    self.assertTrue(env.physics.diagnostics()["stable"])
                    self.assertEqual(env.physics.boundary_penetration(),0)
                    self.assertFalse(any(c.dist<0 for c in env.physics.data.contact[:env.physics.data.ncon]))
                    positions,_=env.physics.get_agents_state()
                    for i,point in enumerate(positions):
                        distances=[]
                        for g in env.physics.object_geom_ids:
                            rot=env.physics.data.geom_xmat[g].reshape(3,3)[:2,:2]
                            local=rot.T@(point-env.physics.data.geom_xpos[g,:2])
                            distances.append(np.linalg.norm(np.maximum(abs(local)-env.physics.model.geom_size[g,:2],0)))
                        gap=min(distances)-env.physics.radii[i]
                        self.assertGreaterEqual(gap,env.config.spawn_gap_min-1e-9)
                        self.assertLessEqual(gap,env.config.spawn_gap_max+1e-9)
                        for j in range(i):
                            self.assertGreaterEqual(np.linalg.norm(point-positions[j]),
                                                    env.physics.radii[i]+env.physics.radii[j]+.01-1e-9)
                    np.testing.assert_allclose(infos["agent_0"]["initial_state"]["object_position"],[-1.5,-1.5])
                    np.testing.assert_array_equal(env.physics.data.qvel,0)
                _,_,_,_,info=env.step(self.zeros(env))
                self.assertEqual(len(info["agent_0"]["episode_metrics"]["agent_object_contact_time_s"]),count)

    def test_spawn_seed_reproduces_a_sequence_without_repeating_every_episode(self):
        env=self.env(num_agents=16,randomize_agents=True)
        sequences=[]
        for _ in range(2):
            sequence=[]
            for seed in [123,None,None]:
                env.reset(seed=seed)
                sequence.append(env.physics.get_agents_state()[0])
            sequences.append(sequence)
        for a,b in zip(*sequences):
            np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(sequences[0][0],sequences[0][1]))
        self.assertFalse(np.array_equal(sequences[0][1],sequences[0][2]))
        fixed=self.env(num_agents=8)
        fixed.reset(seed=1)
        first=fixed.physics.get_agents_state()[0]
        fixed.reset(seed=2)
        np.testing.assert_array_equal(first,fixed.physics.get_agents_state()[0])

    def test_invalid_population_and_spawn_settings_are_rejected(self):
        for values in [dict(num_agents=0),dict(num_agents=17),dict(num_agents=True),
                       dict(num_agents=2,agent_positions=[[0,0]]),dict(randomize_agents=1),
                       dict(spawn_gap_min=.2,spawn_gap_max=.1),dict(spawn_gap_min=.04,spawn_gap_max=.04)]:
            with self.subTest(values=values),self.assertRaises(ValueError):
                load_config(values)

    def test_reference_controller_physically_completes_fixed_easy(self):
        env=self.env()
        env.reset()
        while env.agents:
            _,_,_,_,infos=env.step(scripted_actions(env))
            self.assertTrue(infos["agent_0"]["stable"],infos["agent_0"]["failure_details"])
        info=infos["agent_0"]
        self.assertTrue(info["success"])
        self.assertEqual(info["episode_step"],env.max_steps)
        self.assertLess(info["completion_time_s"],info["elapsed_time_s"])
        self.assertEqual(info["end_reason"],"time_limit")
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
            _,_,term,trunc,info=env.step(self.zeros(env))
        self.assertFalse(any(term.values()))
        self.assertFalse(any(trunc.values()))
        self.assertTrue(info["agent_0"]["success"])
        self.assertEqual(info["agent_0"]["reward_components"]["success"],20)
        self.assertIsNone(info["agent_0"]["end_reason"])
        self.assertEqual(len(env.agents),env.config.num_agents)

    def test_success_bonus_is_once_and_episode_continues_after_leaving_and_reentering(self):
        env=self.env(max_steps=20)
        env.reset()
        gx,gy=env.config.goal_center
        env.physics.reset([gx,gy,0])
        bonuses=[]
        for _ in range(6):
            _,_,term,trunc,infos=env.step(self.zeros(env))
            bonuses.append(infos["agent_0"]["reward_components"]["success"])
            self.assertFalse(any(term.values()))
            self.assertFalse(any(trunc.values()))
        completed=infos["agent_0"]["completion_time_s"]
        self.assertAlmostEqual(completed,5*env.physics.control_dt)
        env.physics.reset(env.config.object_pose)
        obs,_,_,_,infos=env.step(self.zeros(env))
        self.assertEqual(infos["agent_0"]["goal_hold_steps"],0)
        self.assertTrue(infos["agent_0"]["success"])
        self.assertEqual(obs["agent_0"][-1],1.)
        env.physics.reset([gx,gy,0])
        while env.agents:
            _,_,term,trunc,infos=env.step(self.zeros(env))
            bonuses.append(infos["agent_0"]["reward_components"]["success"])
        self.assertEqual(sum(bonuses),20)
        self.assertTrue(all(trunc.values()))
        self.assertFalse(any(term.values()))
        self.assertEqual(infos["agent_0"]["completion_time_s"],completed)
        self.assertTrue(infos["agent_0"]["success"])
        self.assertEqual(infos["agent_0"]["end_reason"],"time_limit")
        env.reset()
        _,_,_,_,infos=env.step(self.zeros(env))
        self.assertFalse(infos["agent_0"]["success"])
        self.assertIsNone(infos["agent_0"]["completion_time_s"])

    def test_success_on_last_step_truncates_and_later_physics_failure_still_terminates(self):
        env=self.env(max_steps=5)
        env.reset()
        gx,gy=env.config.goal_center
        env.physics.reset([gx,gy,0])
        for _ in range(5):
            _,_,term,trunc,infos=env.step(self.zeros(env))
        self.assertTrue(infos["agent_0"]["success"])
        self.assertTrue(all(trunc.values()))
        self.assertFalse(any(term.values()))
        env=self.env(max_steps=10)
        env.reset()
        env.physics.reset([gx,gy,0])
        for _ in range(5):
            env.step(self.zeros(env))
        env.physics.data.qpos[env.physics.object_qpos[0]]=np.nan
        _,rewards,term,trunc,infos=env.step(self.zeros(env))
        self.assertTrue(all(term.values()))
        self.assertFalse(any(trunc.values()))
        self.assertFalse(infos["agent_0"]["success"])
        self.assertEqual(rewards["agent_0"],-20)
        self.assertEqual(infos["agent_0"]["end_reason"],"physics_failure")

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
