"""M3 behavior checks. Install requirements-training.txt to enable PyTorch tests."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "install requirements-training.txt for M3 tests")
class MAPPOTests(unittest.TestCase):
    def setUp(self):
        from src.algorithms.mappo import MAPPO
        from src.training.config import MAPPOConfig
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = MAPPOConfig(actor_hidden_size=16, critic_hidden_size=16,
                                  rollout_steps=16, ppo_epochs=2, target_kl=1.0)
        self.agent = MAPPO(3, 5, config=self.config)

    def make_buffer(self):
        from src.algorithms.rollout_buffer import RolloutBuffer
        rng = np.random.default_rng(7)
        buffer = RolloutBuffer(16, 4, 3, 5)
        for t in range(16):
            observations = rng.normal(size=(4, 3)).astype(np.float32)
            state = rng.normal(size=5).astype(np.float32)
            _, latent, log_probs = self.agent.act(observations)
            value = float(self.agent.value(state))
            # Independent terminal transitions with diverse rewards: enough
            # variation to exercise both policy and state-value gradients.
            buffer.add(observations, state, latent, log_probs, (t % 5) - 1.0,
                       value, 0, True, False)
        return buffer

    def test_sampled_and_recomputed_likelihoods_agree(self):
        observations = torch.randn(4, 3)
        actions, latent, old = self.agent.actor.sample(observations)
        current, entropy = self.agent.actor.evaluate(observations, latent)
        torch.testing.assert_close(old, current)
        self.assertTrue(torch.all(actions.abs() <= 1))
        self.assertTrue(torch.isfinite(entropy).all())

    def test_log_probability_matches_transformed_distribution(self):
        from torch.distributions import Independent, TanhTransform, TransformedDistribution
        observations = torch.randn(4, 3)
        distribution = self.agent.actor.distribution(observations)
        latent = torch.tensor([[0.2, -0.4], [0.8, -1.2], [-0.3, 0.1], [1.1, 0.5]])
        expected = TransformedDistribution(Independent(distribution, 1), [TanhTransform(cache_size=1)])
        actual = self.agent.actor.squashed_log_prob(distribution, latent)
        torch.testing.assert_close(actual, expected.log_prob(latent.tanh()), atol=1e-5, rtol=1e-5)

    def test_saturated_actions_have_finite_log_probabilities_and_gradients(self):
        observations = torch.zeros(2, 3)
        latent = torch.tensor([[20., -20.], [50., -50.]])
        log_prob, entropy = self.agent.actor.evaluate(observations, latent)
        loss = -log_prob.mean() - 0.01 * entropy.mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in self.agent.actor.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_ppo_updates_actor_and_critic_and_reports_finite_losses(self):
        buffer = self.make_buffer()
        actor_before = [p.detach().clone() for p in self.agent.actor.parameters()]
        critic_before = [p.detach().clone() for p in self.agent.critic.parameters()]
        metrics = self.agent.update(buffer, np.random.default_rng(4))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, self.agent.actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(critic_before, self.agent.critic.parameters())))
        for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "actor_grad_norm", "critic_grad_norm"):
            self.assertTrue(np.isfinite(metrics[name]), name)
        self.assertEqual(metrics["team_transitions"], 16)
        self.assertEqual(metrics["actor_samples"], 64)
        self.assertGreater(self.agent.obs_normalizer.count.item(), 0)

    def test_actor_execution_does_not_call_critic(self):
        def fail(*args, **kwargs):
            raise AssertionError("critic must not run during action selection")
        self.agent.critic.forward = fail
        actions, _, _ = self.agent.act(np.zeros((4, 3)), deterministic=True)
        self.assertEqual(actions.shape, (4, 2))

    def test_checkpoint_roundtrip_restores_actions_values_normalizers_and_optimizers(self):
        from src.algorithms.mappo import MAPPO
        self.agent.update(self.make_buffer(), np.random.default_rng(4))
        observations = np.full((4, 3), 0.3, dtype=np.float32)
        state = np.ones(5, dtype=np.float32)
        actions = self.agent.act(observations, deterministic=True)[0]
        value = self.agent.value(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            self.agent.save_checkpoint(path, {"environment_steps": 16})
            restored, payload = MAPPO.from_checkpoint(path, load_optimizers=True)
            np.testing.assert_array_equal(restored.act(observations, deterministic=True)[0], actions)
            np.testing.assert_array_equal(restored.value(state), value)
            torch.testing.assert_close(restored.obs_normalizer.mean, self.agent.obs_normalizer.mean)
            self.assertTrue(restored.actor_optimizer.state)
            self.assertTrue(restored.critic_optimizer.state)
            self.assertEqual(payload["training"]["environment_steps"], 16)

    def test_normalization_forward_does_not_change_statistics(self):
        from src.algorithms.normalization import RunningNormalizer
        normalizer = RunningNormalizer(2)
        normalizer.update(torch.tensor([[1., 2.], [3., 6.]]))
        mean = normalizer.mean.clone()
        count = normalizer.count.clone()
        result = normalizer(torch.tensor([[1., 2.], [3., 6.]]))
        torch.testing.assert_close(result, torch.tensor([[-1., -1.], [1., 1.]]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(normalizer.mean, mean)
        torch.testing.assert_close(normalizer.count, count)


if __name__ == "__main__":
    unittest.main()
