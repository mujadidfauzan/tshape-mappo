import unittest

from src.envs.config import EnvironmentConfig
from src.envs.rewards import team_reward


class RewardTests(unittest.TestCase):
    def test_progress_and_retreat_have_opposite_shaping(self):
        config = EnvironmentConfig()
        closer, progress, parts = team_reward(2.0, 1.9, False, False, config)
        farther, _, _ = team_reward(1.9, 2.0, False, False, config)
        self.assertAlmostEqual(progress, 0.1)
        self.assertAlmostEqual(closer, sum(parts.values()))
        self.assertGreater(closer, 0)
        self.assertLess(farther, 0)

    def test_success_bonus_and_failure_penalty_are_explicit(self):
        config = EnvironmentConfig()
        success, _, _ = team_reward(1, 1, True, False, config)
        failed, progress, _ = team_reward(100, 0, False, True, config)
        self.assertAlmostEqual(success, config.success_reward - config.time_penalty)
        self.assertAlmostEqual(failed, -config.failure_penalty - config.time_penalty)
        self.assertEqual(progress, 0, "invalid physics must not earn progress reward")


if __name__ == "__main__":
    unittest.main()
