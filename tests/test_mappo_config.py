import unittest

from src.training.config import load_mappo_config


class MAPPOConfigTests(unittest.TestCase):
    def test_default_and_overrides(self):
        config = load_mappo_config(total_steps=128, device="cpu")
        self.assertEqual(config.total_steps, 128)
        self.assertEqual(config.rollout_steps, 1024)
        self.assertEqual(config.num_minibatches, 1)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.video_interval_episodes, 10000)
        self.assertEqual((config.video_width, config.video_height), (1920, 1080))

    def test_invalid_training_settings_are_rejected(self):
        for config in ({"rollout_steps": 0}, {"ppo_epochs": 1.5}, {"gamma": 1.1},
                       {"gae_lambda": -0.1}, {"actor_lr": "fast"}, {"device": False},
                       {"clip_ratio": 1.0}, {"target_kl": 0}, {"entropy_coef": float("nan")},
                       {"rollout_steps": 2, "num_minibatches": 3}, {"total_step": 5},
                       {"total_episodes": 0}, {"total_episodes": True},
                       {"video_interval_episodes": -1}, {"video_width": 1919},
                       {"video_height": 0}, {"video_crf": 52}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    load_mappo_config(config)


if __name__ == "__main__":
    unittest.main()
