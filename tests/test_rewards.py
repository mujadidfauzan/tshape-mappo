import unittest
import math

from src.envs.config import EnvironmentConfig
from src.envs.rewards import team_reward, wrapped_yaw_error


class RewardTests(unittest.TestCase):
    def test_progress_and_retreat_have_opposite_shaping(self):
        config = EnvironmentConfig(reward_mode="baseline")
        closer, progress, parts = team_reward(2.0, 1.9, False, False, config)
        farther, _, _ = team_reward(1.9, 2.0, False, False, config)
        self.assertAlmostEqual(progress, 0.1)
        self.assertAlmostEqual(closer, sum(parts.values()))
        self.assertGreater(closer, 0)
        self.assertLess(farther, 0)

    def test_success_bonus_and_failure_penalty_are_explicit(self):
        config = EnvironmentConfig(reward_mode="baseline")
        success, _, _ = team_reward(1, 1, True, False, config)
        failed, progress, _ = team_reward(100, 0, False, True, config)
        self.assertAlmostEqual(success, config.success_reward - config.time_penalty)
        self.assertAlmostEqual(failed, -config.failure_penalty - config.time_penalty)
        self.assertEqual(progress, 0, "invalid physics must not earn progress reward")

    def test_pose_reward_matches_four_component_definition(self):
        config = EnvironmentConfig()
        reward, progress, parts = team_reward(1.2, 1, False, False, config, yaw=math.pi/2)
        self.assertAlmostEqual(progress, .2)
        self.assertAlmostEqual(parts["distance"], -.1 * .5)
        self.assertAlmostEqual(parts["orientation"], -.05 * math.exp(-1) * .5)
        self.assertAlmostEqual(parts["progress"], 2 * .2)
        self.assertEqual(parts["success"], 0)
        self.assertEqual(parts["failure"], 0)
        self.assertNotIn("time", parts)
        self.assertAlmostEqual(reward, -.05 - .025*math.exp(-1) + .4)
        self.assertAlmostEqual(reward, sum(parts.values()))

    def test_stationary_pose_gets_no_positive_reward_for_loitering(self):
        config = EnvironmentConfig()
        near, _, _ = team_reward(.2,.2,False,False,config,yaw=0)
        far, _, _ = team_reward(2,2,False,False,config,yaw=0)
        at_goal, _, _ = team_reward(0,0,False,False,config,yaw=0)
        self.assertLess(far, near)
        self.assertLess(near, 0)
        self.assertEqual(at_goal, 0)

    def test_pose_orientation_wraps_at_pi_and_distinguishes_unequal_ends(self):
        self.assertAlmostEqual(abs(wrapped_yaw_error(-math.pi+.01,math.pi-.01)), .02)
        config = EnvironmentConfig(goal_yaw=math.pi-.01)
        a=team_reward(0,0,False,False,config,yaw=-math.pi+.01)[2]["orientation"]
        b=team_reward(0,0,False,False,config,yaw=math.pi+.01)[2]["orientation"]
        self.assertAlmostEqual(a,b)
        self.assertAlmostEqual(team_reward(0,0,False,False,EnvironmentConfig(),yaw=math.pi)[2]["orientation"],-.05)

    def test_orientation_follows_configured_goal_and_is_weaker_far_from_goal(self):
        config = EnvironmentConfig(goal_yaw=math.pi/2)
        aligned=team_reward(0,0,False,False,config,yaw=math.pi/2)[2]["orientation"]
        near=team_reward(.1,.1,False,False,config,yaw=0)[2]["orientation"]
        far=team_reward(3,3,False,False,config,yaw=0)[2]["orientation"]
        self.assertEqual(aligned, 0)
        self.assertLess(near,far)
        self.assertLess(far,0)

    def test_pose_progress_rewards_closing_and_penalizes_retreat(self):
        config=EnvironmentConfig()
        closer=team_reward(1.1,1,False,False,config,yaw=0)[2]["progress"]
        farther=team_reward(.9,1,False,False,config,yaw=0)[2]["progress"]
        self.assertAlmostEqual(closer,.2)
        self.assertAlmostEqual(farther,-.2)

    def test_success_and_invalid_physics_have_explicit_finite_terminal_rewards(self):
        config=EnvironmentConfig()
        reward, _, parts=team_reward(0,0,True,False,config,yaw=0)
        self.assertEqual(reward,config.success_reward)
        self.assertEqual(parts["success"],config.success_reward)
        reward, progress, parts=team_reward(float('inf'),float('nan'),True,True,config,yaw=float('nan'))
        self.assertEqual(reward,-config.failure_penalty)
        self.assertEqual(progress,0)
        self.assertTrue(all(value==0 for key,value in parts.items() if key!='failure'))
        with self.assertRaisesRegex(ValueError,"finite object yaw"):
            team_reward(1,1,False,False,config)

    def test_pose_ignores_legacy_time_penalty_and_scales_can_be_tuned(self):
        config=EnvironmentConfig(time_penalty=100,distance_scale=2,distance_weight=.4,
                                 orientation_weight=0,progress_weight=0,success_reward=7)
        reward, _, _=team_reward(2,2,False,False,config,yaw=0)
        self.assertEqual(reward,-.2)
        self.assertEqual(team_reward(0,0,True,False,config,yaw=0)[0],7)


if __name__ == "__main__":
    unittest.main()
