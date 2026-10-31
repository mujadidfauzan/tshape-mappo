"""Rolling episode statistics and TensorBoard scalars indexed by team steps."""

from collections import deque
import math
from numbers import Real
from pathlib import Path


class EpisodeStats:
    def __init__(self, window_size, records=()):
        self.records = deque(records, maxlen=window_size)

    def add(self, info):
        metrics = info["episode_metrics"]
        self.records.append({
            "ep_rew_mean": float(metrics["episode_return"]),
            "ep_len_mean": int(info["episode_step"]),
            "success_rate": float(info["success"]),
            "stable_rate": float(info["stable"]),
            "distance_to_goal_mean": info["distance_to_goal"],
            "object_path_length_mean_m": float(metrics["object_path_length_m"]),
            "wall_collision_events_mean": float(metrics["wall_collision_events"]),
            "control_effort_mean_N2_s": float(metrics["control_effort_N2_s"]),
            "max_penetration_mean_m": float(metrics["max_penetration_m"]),
            "completion_time_mean_s": info["completion_time_s"],
        })
        diagnostics = info.get("diagnostics", {})
        for name in ("footprint_error_m", "footprint_margin_m", "gate_clear_fraction"):
            if name in diagnostics:
                self.records[-1][name + "_mean"] = diagnostics[name]
        if diagnostics:
            for category in ("gate_blocked", "gate_not_reached", "contact_lost", "goal_not_settled",
                             "goal_footprint_outside", "physics_failure"):
                self.records[-1][category + "_rate"] = float(diagnostics.get("failure_category") == category)

    def summary(self):
        result = {"window_episodes": len(self.records)}
        if not self.records:
            return result
        for key in self.records[-1]:
            values = [r[key] for r in self.records if r.get(key) is not None and math.isfinite(r[key])]
            if values:
                result[key] = sum(values) / len(values)
        return result

    def state_dict(self):
        return list(self.records)


class RunLogger:
    def __init__(self, run_dir, enabled=True, reward_interval=100):
        self.writer = None
        self.reward_interval = reward_interval
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError("TensorBoard is missing; install requirements-training.txt "
                                   "or use --no-tensorboard") from error
            self.writer = SummaryWriter(log_dir=str(Path(run_dir) / "tensorboard"), flush_secs=10)

    def scalars(self, prefix, values, step):
        if self.writer is None:
            return
        for name, value in values.items():
            if isinstance(value, Real) and math.isfinite(value):
                self.writer.add_scalar(f"{prefix}/{name}", value, global_step=step)

    def reward(self, reward, step):
        if self.writer is not None and self.reward_interval and step % self.reward_interval == 0:
            self.writer.add_scalar("reward/step", reward, global_step=step)

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
