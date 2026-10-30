"""Regression tests for sliding in the physics wrapper used by the environment."""

import unittest

import mujoco
import numpy as np

from src.envs.physics import TShapePhysics


class SurfaceResistanceTests(unittest.TestCase):
    def make_physics(self, frame_skip=10):
        physics = TShapePhysics(frame_skip=frame_skip)
        # Park all agents away so neither contacts nor walls can stop the T.
        parking = np.array([[-2., -2.], [2., -2.], [-2., 2.], [2., 2.]])
        physics.data.qpos[physics.agent_qpos] = (
            parking - physics.model.body_pos[physics.agent_bodies, :2]
        )
        mujoco.mj_forward(physics.model, physics.data)
        self.assertEqual(physics.data.ncon, 0)
        return physics

    def coast(self, physics, seconds):
        for _ in range(round(seconds / physics.control_dt)):
            physics.step(np.zeros((4, 2)))
            self.assertEqual(physics.data.ncon, 0, "contact confounds the resistance measurement")
            self.assertTrue(physics.is_stable())

    def test_linear_motion_settles_without_contact_in_both_directions(self):
        for frame_skip in (1, 10):
            for axis in (0, 1):
                for sign in (-1, 1):
                    with self.subTest(frame_skip=frame_skip, axis=axis, sign=sign):
                        p = self.make_physics(frame_skip)
                        p.data.qvel[p.object_dofs[axis]] = sign * 0.25
                        mujoco.mj_forward(p.model, p.data)
                        self.coast(p, 1.0)
                        state = p.get_object_state()
                        self.assertLess(np.linalg.norm(state["velocity"]), 0.001)
                        travel = sign * state["position"][axis]
                        self.assertGreater(travel, 0.01, "motion must not be frozen or teleported")
                        self.assertLess(travel, 0.08)
                        stopped_position = state["position"].copy()
                        self.coast(p, 2.0)
                        np.testing.assert_allclose(p.get_object_state()["position"], stopped_position, atol=1e-4)

    def test_rotation_settles_without_contact_in_both_directions(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                p = self.make_physics()
                p.data.qvel[p.object_dofs[2]] = sign * 0.5
                mujoco.mj_forward(p.model, p.data)
                self.coast(p, 1.0)
                state = p.get_object_state()
                self.assertLess(abs(state["angular_velocity"]), 0.001)
                self.assertGreater(sign * state["yaw"], 0.01)
                self.assertLess(sign * state["yaw"], 0.2)

    def test_small_disturbances_do_not_produce_continual_drift(self):
        for axis, force, tolerance in ((0, 0.2, 0.001), (1, 0.2, 0.001), (2, 0.02, 0.001)):
            for sign in (-1, 1):
                with self.subTest(axis=axis, sign=sign):
                    p = self.make_physics()
                    # Direct force is only a controlled resistance probe in this
                    # test. Transport controllers still actuate agents exclusively.
                    p.data.qfrc_applied[p.object_dofs[axis]] = sign * force
                    self.coast(p, 2.0)
                    self.assertLess(abs(p.data.qpos[p.object_qpos[axis]]), tolerance)

    def test_sustained_force_still_overcomes_resistance(self):
        for axis, extra_force, minimum in ((0, 1.0, 0.1), (1, 1.0, 0.1), (2, 0.2, 0.1)):
            with self.subTest(axis=axis):
                p = self.make_physics()
                # Keep this probe above the resistance the user has tuned.
                dof = p.object_dofs[axis]
                p.data.qfrc_applied[dof] = p.model.dof_frictionloss[dof] + extra_force
                self.coast(p, 2.0)
                self.assertGreater(p.data.qpos[p.object_qpos[axis]], minimum)


if __name__ == "__main__":
    unittest.main()
