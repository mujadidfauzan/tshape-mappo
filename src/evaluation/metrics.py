"""Aggregate matched evaluation protocols without mixing best/latest or tasks."""

from collections import defaultdict
import numpy as np


def aggregate_benchmarks(rows):
    groups = defaultdict(list)
    keys = ("task_id", "protocol_id", "num_agents", "difficulty", "reward_mode", "mode", "checkpoint_kind")
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, members in groups.items():
        row = dict(zip(keys, key))
        row.update(policies=len(members), unique_training_seeds=len({r["training_seed"] for r in members}))
        for metric in ("success_rate", "stable_rate", "mean_return", "mean_completion_time_s",
                       "orientation_error_rad_mean",
                       "object_path_length_m_mean", "control_effort_N2_s_mean", "control_effort_per_agent_N2_s_mean"):
            values = [float(r[metric]) for r in members if r.get(metric) is not None]
            row[metric + "_mean"] = float(np.mean(values)) if values else None
            row[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        output.append(row)
    return output
