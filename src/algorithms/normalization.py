"""Running input statistics, explicitly frozen throughout collection and PPO."""

import torch
from torch import nn


class RunningNormalizer(nn.Module):
    def __init__(self, size, clip=10.0, enabled=True):
        super().__init__()
        self.clip = clip
        self.enabled = enabled
        self.register_buffer("mean", torch.zeros(size, dtype=torch.float64))
        self.register_buffer("variance", torch.ones(size, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))

    def forward(self, values):
        if not self.enabled or self.count.item() == 0:
            return values
        normalized = (values - self.mean.to(values.dtype)) / torch.sqrt(self.variance.to(values.dtype) + 1e-6)
        return normalized.clamp(-self.clip, self.clip)

    @torch.no_grad()
    def update(self, values):
        if not self.enabled:
            return
        values = values.reshape(-1, self.mean.numel()).to(dtype=torch.float64, device=self.mean.device)
        if values.shape[0] == 0:
            return
        batch_count = values.shape[0]
        batch_mean = values.mean(dim=0)
        batch_variance = values.var(dim=0, unbiased=False)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        combined = (self.variance * self.count + batch_variance * batch_count
                    + delta.square() * self.count * batch_count / total)
        self.mean.add_(delta * batch_count / total)
        self.variance.copy_((combined / total).clamp_min(1e-6))
        self.count.copy_(total)
