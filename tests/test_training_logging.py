"""Episode-window semantics, independent of reward scale and episode length."""

import unittest

from src.training.logging import EpisodeStats


class EpisodeStatsTests(unittest.TestCase):
    def info(self, reward, length, success, completion=None):
        return {"episode_step": length, "success": success, "stable": True,
                "distance_to_goal": 0.1, "completion_time_s": completion,
                "episode_metrics": {"episode_return": reward, "object_path_length_m": 1.,
                    "wall_collision_events": 0, "control_effort_N2_s": 2., "max_penetration_m": .001}}

    def test_window_counts_episodes_and_excludes_unsuccessful_completion_times(self):
        stats = EpisodeStats(2)
        self.assertEqual(stats.summary(), {"window_episodes": 0})
        stats.add(self.info(100, 10, True, 0.5))
        stats.add(self.info(-1, 500, False))
        stats.add(self.info(21, 100, True, 5.0))
        summary = stats.summary()
        self.assertEqual(summary["window_episodes"], 2)
        self.assertEqual(summary["ep_rew_mean"], 10)
        self.assertEqual(summary["ep_len_mean"], 300)
        self.assertEqual(summary["success_rate"], .5)
        self.assertEqual(summary["completion_time_mean_s"], 5.)
        restored = EpisodeStats(2, stats.state_dict())
        self.assertEqual(restored.summary(), summary)
