import unittest

import numpy as np

from src.envs.observations import ACTOR_FEATURES
from src.envs.tshape_env import TShapeParallelEnv


class ObservationTests(unittest.TestCase):
    def test_each_actor_can_identify_itself_and_its_relative_object(self):
        env = TShapeParallelEnv(config={"randomize_reset": False})
        self.addCleanup(env.close)
        observations, _ = env.reset(seed=42)
        positions, velocities = env.physics.get_agents_state()
        obj = env.physics.get_object_state()
        slices, start = {}, 0
        for name, size in ACTOR_FEATURES:
            slices[name] = slice(start, start + size)
            start += size
        for i, name in enumerate(env.possible_agents):
            obs = observations[name]
            np.testing.assert_allclose(obs[slices["self_position"]], positions[i], atol=1e-7)
            np.testing.assert_allclose(obs[slices["self_velocity"]], velocities[i], atol=1e-7)
            np.testing.assert_allclose(obs[slices["object_relative"]], obj["position"] - positions[i], atol=1e-7)
            others = [j for j in range(4) if j != i]
            np.testing.assert_allclose(obs[slices["other_relative_positions"]],
                                       (positions[others] - positions[i]).ravel(), atol=1e-7)
            self.assertEqual(obs[slices["remaining_time"]][0], 1.0)
            self.assertEqual(obs[slices["goal_hold_fraction"]][0], 0.0)
        observations["agent_0"][:] = 123.0
        self.assertFalse(np.all(observations["agent_1"] == 123.0))
        self.assertFalse(np.all(env.state() == 123.0))


if __name__ == "__main__":
    unittest.main()
