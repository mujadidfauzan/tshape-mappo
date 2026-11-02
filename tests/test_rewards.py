"""Checks for the single four-component Easy reward."""

import math
import unittest

from src.envs.config import EnvironmentConfig
from src.envs.rewards import team_reward,wrapped_yaw_error


class RewardTests(unittest.TestCase):
    def test_formula_and_component_sum(self):
        c=EnvironmentConfig()
        reward,parts=team_reward(1.2,1,math.pi/2,False,False,c)
        self.assertAlmostEqual(parts["distance"],-.05)
        self.assertAlmostEqual(parts["orientation"],-.025*math.exp(-1))
        self.assertAlmostEqual(parts["progress"],.4)
        self.assertEqual(parts["success"],0)
        self.assertAlmostEqual(reward,sum(parts.values()))
        self.assertNotIn("time",parts)

    def test_waiting_is_nonpositive_and_retreat_loses_progress(self):
        c=EnvironmentConfig()
        self.assertLess(team_reward(1,1,0,False,False,c)[0],0)
        self.assertEqual(team_reward(0,0,0,False,False,c)[0],0)
        self.assertLess(team_reward(.5,1,0,False,False,c)[1]["progress"],0)

    def test_angle_wrap_distinguishes_unequal_ends(self):
        self.assertAlmostEqual(abs(wrapped_yaw_error(-math.pi+.01,math.pi-.01)),.02)
        self.assertAlmostEqual(abs(wrapped_yaw_error(math.pi,0)),math.pi)

    def test_success_bonus_and_safety_override(self):
        c=EnvironmentConfig()
        self.assertEqual(team_reward(0,0,0,True,False,c)[0],20)
        reward,parts=team_reward(float("nan"),float("nan"),float("nan"),True,True,c)
        self.assertEqual(reward,-20)
        self.assertEqual(parts["success"],0)
        self.assertEqual(parts["progress"],0)
