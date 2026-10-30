"""Team GAE with separate masks for bootstrapping and episode boundaries."""

import numpy as np


def compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.99, gae_lambda=0.95):
    arrays = [np.asarray(a) for a in (rewards, values, next_values, terminated, truncated)]
    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("GAE inputs must be one-dimensional arrays with equal length")
    rewards, values, next_values = (np.asarray(a, dtype=np.float64) for a in arrays[:3])
    if not all(np.isfinite(a).all() for a in (rewards, values, next_values)):
        raise ValueError("GAE rewards and values must be finite")
    terminated, truncated = (np.asarray(a, dtype=bool) for a in arrays[3:])
    advantages = np.zeros(len(rewards), dtype=np.float64)
    carry = 0.0
    for t in reversed(range(len(rewards))):
        # A time limit bootstraps from the FINAL observation, but does not carry
        # advantages from the next reset episode. True termination never bootstraps.
        delta = rewards[t] + gamma * (not terminated[t]) * next_values[t] - values[t]
        carry = delta + gamma * gae_lambda * (not (terminated[t] or truncated[t])) * carry
        advantages[t] = carry
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)
