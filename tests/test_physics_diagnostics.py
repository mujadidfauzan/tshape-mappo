"""Behavioral checks for the M1 diagnostic harness (standard-library unittest)."""

import unittest

import mujoco
import numpy as np

from scripts.test_physics import Physics, run_scenario


class PhysicsDiagnosticsTests(unittest.TestCase):
    def test_each_motor_moves_only_its_agent_axis_without_contact(self):
        physics = Physics()
        m, d = physics.model, physics.data
        for agent in range(4):
            for axis in range(2):
                with self.subTest(agent=agent, axis=axis):
                    physics.reset("idle")
                    d.ctrl[physics.motors[agent, axis]] = 0.25
                    for _ in range(20):
                        mujoco.mj_step(m, d)
                    expected_dof = physics.agent_dofs[agent, axis]
                    self.assertGreater(d.qvel[expected_dof], 0.01)
                    other_dofs = np.delete(d.qvel, expected_dof)
                    np.testing.assert_allclose(other_dofs, 0, atol=1e-12)

    def test_contact_targets_rotate_with_the_object(self):
        physics = Physics()
        m, d = physics.model, physics.data
        physics.reset("rotation")
        original_controls = physics.controller("rotation", 1.0)
        original_positions = d.xpos[physics.agent_bodies, :2].copy()
        rotation = np.array([[0., -1.], [1., 0.]])
        d.qpos[physics.object_qpos[2]] = np.pi / 2
        for agent in range(2):
            target = rotation @ original_positions[agent]
            d.qpos[physics.agent_qpos[agent]] = target - m.body_pos[physics.agent_bodies[agent], :2]
        mujoco.mj_forward(m, d)
        rotated_controls = physics.controller("rotation", 1.0)
        for agent in range(4):
            ids = physics.motors[agent]
            np.testing.assert_allclose(rotated_controls[ids], rotation @ original_controls[ids], atol=1e-12)

    def test_disabled_contacts_fail_translation_and_leave_object_stationary(self):
        physics = Physics()
        physics.model.geom_contype[physics.moving_geoms] = 0
        physics.model.geom_conaffinity[physics.moving_geoms] = 0
        report, _, _ = run_scenario(physics, "translation", 1.0, 1.0)
        self.assertFalse(report["passed"])
        self.assertEqual(report["object_contact_steps"], 0)
        self.assertAlmostEqual(report["displacement_m"], 0)
        self.assertIn("no agent-object contact detected", report["failures"])

    def test_reset_clears_motion_controls_external_forces_and_warnings(self):
        physics = Physics()
        physics.reset("translation")
        initial = physics.integration_snapshot()
        d = physics.data
        d.qpos[:] += 0.1
        d.qvel[:] = 0.2
        d.ctrl[:] = 0.5
        d.qacc_warmstart[:] = 0.3
        d.qfrc_applied[:] = 0.4
        d.xfrc_applied[:] = 0.6
        d.warning[0].number = 1
        d.time = 10
        physics.reset("translation")
        np.testing.assert_array_equal(physics.integration_snapshot(), initial)
        self.assertFalse(np.any(d.warning.number))

    def test_missing_wall_collisions_are_detected_by_footprint_guard(self):
        physics = Physics()
        for wall in physics.walls:
            physics.model.geom_contype[wall] = 0
            physics.model.geom_conaffinity[wall] = 0
        report, _, _ = run_scenario(physics, "wall", 6.0, 1.0)
        self.assertFalse(report["passed"])
        self.assertIn("penetration exceeds 5 mm tolerance", report["failures"])

    def test_release_has_zero_controls_for_every_agent(self):
        physics = Physics()
        physics.reset("release")
        physics.data.time = 2.5
        np.testing.assert_array_equal(physics.controller("release", 1.0), np.zeros(8))


if __name__ == "__main__":
    unittest.main()
