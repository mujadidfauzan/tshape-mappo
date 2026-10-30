import unittest

from src.envs.config import load_config


class EnvironmentConfigTests(unittest.TestCase):
    def test_default_yaml_and_constructor_override(self):
        config = load_config(max_steps=17)
        self.assertEqual(config.max_steps, 17)
        self.assertEqual(config.frame_skip, 50)
        self.assertTrue(config.resolved_model_path.is_file())
        self.assertEqual(config.goal_center, (1.8, 1.5))

    def test_unknown_and_invalid_parameters_are_rejected(self):
        for config in ({"max_step": 5}, {"frame_skip": 1.5}, {"max_steps": 0},
                       {"goal_center": [float("nan"), 0]}, {"goal_half_size": [0, 1]},
                       {"object_position_noise": [-1, 0]}, {"time_penalty": -1},
                       {"randomize_reset": "false"}, {"reset_distribution": "unknown"},
                       {"arena_spawn_margin": -1}, {"minimum_goal_distance": -1}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    load_config(config)


if __name__ == "__main__":
    unittest.main()
