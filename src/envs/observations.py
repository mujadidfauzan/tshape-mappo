"""Fixed four-agent Easy observations: 34 actor features and 33 critic features.

Vectors use world-aligned XY axes. This is NOT a limited-range local sensor model.
Other agents appear in ascending agent index, excluding the observer.
"""

import numpy as np

# Sizes are explicit so the contract can be inspected without running MuJoCo.
ACTOR_FEATURES = (
    ("self_position", 2), ("self_velocity", 2), ("object_relative", 2),
    ("object_sin_cos_yaw", 2), ("object_velocity", 2), ("object_yaw_rate", 1),
    ("goal_relative", 2), ("goal_half_size", 2), ("other_relative_positions", 6),
    ("other_velocities", 6), ("wall_clearances", 4), ("object_contact", 1),
    ("remaining_time", 1), ("goal_hold_fraction", 1),
)
STATE_FEATURES = (
    ("object_position", 2), ("object_sin_cos_yaw", 2), ("object_velocity", 2),
    ("object_yaw_rate", 1), ("goal_center", 2), ("goal_half_size", 2),
    ("agent_positions", 8), ("agent_velocities", 8), ("arena_lower_upper", 4),
    ("remaining_time", 1), ("goal_hold_fraction", 1),
)
OBSERVATION_SIZE = sum(size for _, size in ACTOR_FEATURES)
STATE_SIZE = sum(size for _, size in STATE_FEATURES)


def build_state(physics, goal_center, goal_half_size, remaining_time, goal_hold):
    obj = physics.get_object_state()
    positions, velocities = physics.get_agents_state()
    return np.concatenate([
        obj["position"], [np.sin(obj["yaw"]), np.cos(obj["yaw"])],
        obj["velocity"], [obj["angular_velocity"]], goal_center, goal_half_size,
        positions.ravel(), velocities.ravel(), physics.lower, physics.upper,
        [remaining_time, goal_hold],
    ]).astype(np.float32)


def build_observations(physics, goal_center, goal_half_size, remaining_time, goal_hold):
    obj = physics.get_object_state()
    positions, velocities = physics.get_agents_state()
    contacts, _, _ = physics.contact_state()
    result = {}
    for i, name in enumerate(physics.agent_names):
        others = [j for j in range(physics.num_agents) if j != i]
        wall_clearances = np.r_[positions[i] - physics.lower - physics.radii[i],
                                physics.upper - positions[i] - physics.radii[i]]
        result[name] = np.concatenate([
            positions[i], velocities[i], obj["position"] - positions[i],
            [np.sin(obj["yaw"]), np.cos(obj["yaw"])], obj["velocity"],
            [obj["angular_velocity"]], goal_center - positions[i], goal_half_size,
            (positions[others] - positions[i]).ravel(), velocities[others].ravel(),
            wall_clearances, [float(contacts[i]), remaining_time, goal_hold],
        ]).astype(np.float32)
    return result
