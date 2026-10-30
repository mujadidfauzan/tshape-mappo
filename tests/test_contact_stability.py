"""Contact regressions for the agent/agent and agent/wall M3 failures."""

import unittest

import mujoco
import numpy as np

from src.envs.config import load_config
from src.envs.physics import TShapePhysics


class ContactStabilityTests(unittest.TestCase):
    def run_collision(self, kind, speed, model_path=None, frame_skip=None):
        cfg = load_config()
        p = TShapePhysics(frame_skip=cfg.frame_skip if frame_skip is None else frame_skip,
                         model_path=cfg.resolved_model_path if model_path is None else model_path)
        positions = np.array([[-1.5, 1.5], [1.5, 1.5], [-.04, -1.5], [.04, -1.5]])
        if kind == "wall":
            positions[2] = [-2.85, -1.5]
            positions[3] = [1.5, -1.5]
        p.reset(agent_positions=positions)
        self.assertEqual(p.data.ncon, 0, "collision must start from a separated state")
        actions = np.zeros((4, 2))
        if kind == "pair":
            p.data.qvel[p.agent_dofs[2, 0]] = speed
            p.data.qvel[p.agent_dofs[3, 0]] = -speed
            actions[2, 0], actions[3, 0] = 1, -1
        else:
            p.data.qvel[p.agent_dofs[2, 0]] = -speed
            actions[2, 0] = -1
        mujoco.mj_forward(p.model, p.data)
        peak, seen_contact = 0.0, False
        for _ in range(round(.5 / p.control_dt)):
            report = p.step(actions, max_penetration=cfg.max_penetration)
            peak = max(peak, report["max_penetration_m"])
            seen_contact |= peak > 0
            self.assertTrue(report["stable"], msg=str(report["failure_details"]))
        self.assertTrue(seen_contact, "agents must collide rather than avoid the probe")
        self.assertLessEqual(peak, cfg.max_penetration)
        return peak

    def test_head_on_agents_remain_within_contact_tolerance(self):
        # 0.705 m/s each matches the observed ~1.41 m/s relative speed;
        # 3 m/s each is the free-space terminal speed under a full axis force.
        for speed in (.705, 1.5, 3.0):
            with self.subTest(speed=speed):
                self.run_collision("pair", speed)

    def test_agent_wall_impact_remains_within_contact_tolerance(self):
        for speed in (1.231, 3.0):
            with self.subTest(speed=speed):
                self.run_collision("wall", speed)

    def test_default_policy_interval_and_episode_duration_are_preserved(self):
        cfg = load_config()
        p = TShapePhysics()
        self.assertAlmostEqual(p.control_dt, .05)
        self.assertAlmostEqual(cfg.frame_skip * p.model.opt.timestep, .05)
        self.assertAlmostEqual(cfg.max_steps * p.control_dt, 25.0)


if __name__ == "__main__":
    unittest.main()
