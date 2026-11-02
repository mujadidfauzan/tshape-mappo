"""On-policy transitions: one team value/reward and one actor row per agent."""

import numpy as np

from .gae import compute_gae


class RolloutBuffer:
    def __init__(self, capacity, num_agents, observation_size, state_size, action_size=2):
        self.capacity = capacity
        self.size = 0
        self.observations = np.empty((capacity, num_agents, observation_size), dtype=np.float32)
        self.states = np.empty((capacity, state_size), dtype=np.float32)
        self.pre_tanh = np.empty((capacity, num_agents, action_size), dtype=np.float32)
        self.log_probs = np.empty((capacity, num_agents), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.values = np.empty(capacity, dtype=np.float32)
        self.next_values = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=bool)
        self.truncated = np.empty(capacity, dtype=bool)

    def add(self, observations, state, pre_tanh, log_probs, reward, value, next_value, terminated, truncated):
        if self.size >= self.capacity:
            raise ValueError("rollout buffer is full")
        items = ((self.observations, observations), (self.states, state),
                 (self.pre_tanh, pre_tanh), (self.log_probs, log_probs))
        for target, item in items:
            item = np.asarray(item, dtype=np.float32)
            if item.shape != target.shape[1:] or not np.isfinite(item).all():
                raise ValueError("invalid rollout feature shape or non-finite value")
        if not np.isfinite([reward, value, next_value]).all():
            raise ValueError("rollout scalar values must be finite")
        t = self.size
        for target, item in items:
            target[t] = item
        self.rewards[t], self.values[t], self.next_values[t] = reward, value, next_value
        self.terminated[t], self.truncated[t] = terminated, truncated
        self.size += 1

    def batch(self, gamma, gae_lambda):
        if self.size == 0:
            raise ValueError("cannot train on an empty rollout")
        selected = slice(0, self.size)
        advantages, returns = compute_gae(
            self.rewards[selected], self.values[selected], self.next_values[selected],
            self.terminated[selected], self.truncated[selected], gamma, gae_lambda,
        )
        # Normalize across team transitions, not across duplicated agent rows.
        normalized = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
        return {"observations": self.observations[selected], "states": self.states[selected],
                "pre_tanh": self.pre_tanh[selected], "old_log_probs": self.log_probs[selected],
                "old_values": self.values[selected], "advantages": normalized,
                "returns": returns}
