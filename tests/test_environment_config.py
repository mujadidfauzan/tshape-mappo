import unittest

from src.envs.config import load_config, load_checkpoint_config


class EnvironmentConfigTests(unittest.TestCase):
    def test_default_yaml_and_constructor_override(self):
        config = load_config(max_steps=17)
        self.assertEqual(config.max_steps, 17)
        self.assertEqual(config.frame_skip, 50)
        self.assertTrue(config.resolved_model_path.is_file())
        self.assertEqual(config.goal_center, (1.8, 1.5))

    def test_unknown_and_invalid_parameters_are_rejected(self):
        for config in ({"max_step": 5}, {"frame_skip": 1.5}, {"max_steps": 0},
                       {"difficulty": "unknown"}, {"difficulty": "medium", "reset_distribution": "legacy"},
                       {"randomize_goal": "false"}, {"difficulty": "hard", "randomize_goal": True},
                       {"goal_center": [float("nan"), 0]}, {"goal_half_size": [0, 1]},
                       {"agent_spawn_gap": [.2, .1]}, {"agent_spawn_gap": [.01, .15]},
                       {"agent_spawn_gap": [-1, 0]}, {"time_penalty": -1},
                       {"randomize_reset": "false"}, {"reset_distribution": "unknown"},
                       {"arena_spawn_margin": -1}, {"minimum_goal_distance": -1}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    load_config(config)

    def test_medium_yaml_and_old_checkpoint_defaults(self):
        config = load_config("configs/environment_medium.yaml")
        self.assertEqual(config.difficulty, "medium")
        self.assertEqual(config.max_steps, 1500)
        self.assertFalse(config.randomize_goal)
        self.assertTrue(config.resolved_model_path.is_file())
        self.assertEqual(load_checkpoint_config(config.to_dict()), config)
        legacy = load_config().to_dict()
        legacy.pop("difficulty")
        legacy.pop("reset_distribution")
        legacy.pop("randomize_goal")
        old = load_checkpoint_config(legacy)
        self.assertEqual((old.difficulty, old.reset_distribution), ("easy", "near_object"))
        self.assertTrue(old.randomize_goal)

    def test_hard_yaml_and_goal_defaults(self):
        config = load_config("configs/environment_hard.yaml")
        self.assertEqual(config.difficulty, "hard")
        self.assertEqual(config.max_steps, 2500)
        self.assertFalse(config.randomize_goal)
        self.assertTrue(config.resolved_model_path.is_file())
        self.assertTrue(load_config({}).randomize_goal)
        self.assertFalse(load_config({"difficulty": "medium"}).randomize_goal)
        self.assertFalse(load_config({"difficulty": "hard"}).randomize_goal)


if __name__ == "__main__":
    unittest.main()
