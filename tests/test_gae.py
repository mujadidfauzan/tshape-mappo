"""Analytic GAE cases protect terminal/truncation bootstrapping semantics."""

import unittest

import numpy as np

from src.algorithms.gae import compute_gae
from src.algorithms.rollout_buffer import RolloutBuffer


class GAETests(unittest.TestCase):
    def test_true_termination_never_bootstraps(self):
        advantage, returns = compute_gae([1], [2], [100], [True], [False], gamma=0.9)
        np.testing.assert_allclose(advantage, [-1])
        np.testing.assert_allclose(returns, [1])

    def test_truncation_bootstraps_but_does_not_leak_next_episode(self):
        advantage, returns = compute_gae([1, 100], [2, 0], [10, 0],
                                         [False, True], [True, False], gamma=0.9, gae_lambda=1)
        np.testing.assert_allclose(advantage, [8, 100])
        np.testing.assert_allclose(returns, [10, 100])

    def test_continuation_and_rollout_cut_use_next_state_value(self):
        advantage, returns = compute_gae([1, 2], [0, 0], [0, 3],
                                         [False, False], [False, False], gamma=0.5, gae_lambda=1)
        np.testing.assert_allclose(advantage, [2.75, 3.5])
        np.testing.assert_allclose(returns, [2.75, 3.5])

    def test_team_reward_is_not_multiplied_by_agent_count(self):
        buffer = RolloutBuffer(2, 4, 3, 5)
        for reward in (1, 2):
            buffer.add(np.zeros((4, 3)), np.zeros(5), np.zeros((4, 2)),
                       np.zeros(4), reward, 0, 0, True, False)
        batch = buffer.batch(0.99, 0.95)
        np.testing.assert_array_equal(batch["returns"], [1, 2])
        self.assertEqual(batch["observations"].shape, (2, 4, 3))
        self.assertEqual(batch["old_values"].shape, (2,))
        with self.assertRaises(ValueError):
            buffer.add(np.zeros((4, 3)), np.zeros(5), np.zeros((4, 2)),
                       np.zeros(4), 3, 0, 0, True, False)

    def test_invalid_shapes_and_nonfinite_rewards_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_gae([1, 2], [0], [0, 0], [False, False], [False, False])
        with self.assertRaises(ValueError):
            compute_gae([np.nan], [0], [0], [False], [False])


if __name__ == "__main__":
    unittest.main()
