import unittest

import torch

from src.agents.ppo_agent import PPOAgent
from src.config import PPOConfig


class TestPPOFixes(unittest.TestCase):
    def _agent(self, scale):
        return PPOAgent(4, 4, PPOConfig(reward_scale=scale, hidden=(8,)), seed=0)

    def test_reward_scale_scales_returns(self):
        outs = []
        for scale in (1.0, 0.02):
            a = self._agent(scale)
            for t in range(5):  # zero critic values -> returns are pure scaled discounted rewards
                a.store([0.0] * 4, 0, 0.0, 100.0 if t == 4 else -0.05, 0.0, t == 4)
            outs.append(a._gae(0.0)[0])
        self.assertTrue(torch.allclose(torch.as_tensor(outs[1]), torch.as_tensor(outs[0]) * 0.02, atol=1e-5))

    def test_policy_stats_uniform(self):
        a = self._agent(1.0)
        torch.nn.init.zeros_(a.model.actor.weight); torch.nn.init.zeros_(a.model.actor.bias)
        ent, pmax = a.policy_stats([0.0] * 4)
        self.assertAlmostEqual(ent, float(torch.log(torch.tensor(4.0))), places=5)
        self.assertAlmostEqual(pmax, 0.25, places=5)

    def test_default_scale_is_original_behaviour(self):
        self.assertEqual(PPOConfig().reward_scale, 1.0)


if __name__ == "__main__":
    unittest.main()
