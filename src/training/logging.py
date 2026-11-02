"""A rolling episode window and TensorBoard logging for fixed Easy."""

from collections import deque
import math
from numbers import Real
from pathlib import Path


class EpisodeStats:
    def __init__(self, window_size, records=()):
        self.records = deque(records, maxlen=window_size)

    def add(self, info):
        m = info["episode_metrics"]
        self.records.append(dict(ep_rew_mean=m["episode_return"], ep_len_mean=info["episode_step"],
                                 success_rate=float(info["success"]), stable_rate=float(info["stable"]),
                                 distance_to_goal_mean=info["distance_to_goal"],
                                 orientation_error_rad_mean=info["orientation_error_rad"],
                                 inside_goal_steps_mean=m["inside_goal_steps"],
                                 object_path_length_m_mean=m["object_path_length_m"]))

    def summary(self):
        result = {"window_episodes": len(self.records)}
        if self.records:
            for key in self.records[-1]:
                values = [r[key] for r in self.records if r[key] is not None and math.isfinite(r[key])]
                if values:
                    result[key] = sum(values)/len(values)
        return result

    def state_dict(self):
        return list(self.records)


class RunLogger:
    def __init__(self, run_dir, enabled=True, reward_interval=100):
        self.writer = None
        self.reward_interval = reward_interval
        if enabled:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(Path(run_dir)/"tensorboard"), flush_secs=10)

    def scalars(self, prefix, values, step):
        if self.writer is not None:
            for key, value in values.items():
                if isinstance(value, Real) and math.isfinite(value):
                    self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def reward(self, value, step):
        if self.writer is not None and self.reward_interval and step % self.reward_interval == 0:
            self.writer.add_scalar("reward/step", value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
