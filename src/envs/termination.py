"""Shared geometry and speed conditions for completing transport tasks."""

import numpy as np


def object_inside_goal(corners, goal_center, goal_half_size):
    lower = np.asarray(goal_center) - goal_half_size
    upper = np.asarray(goal_center) + goal_half_size
    return bool(np.isfinite(corners).all()
                and np.all(corners >= lower - 1e-9)
                and np.all(corners <= upper + 1e-9))


def goal_is_settled(physics, goal_center, goal_half_size, linear_speed, angular_speed):
    obj = physics.get_object_state()
    return (object_inside_goal(physics.object_corners(), goal_center, goal_half_size)
            and float(np.linalg.norm(obj["velocity"])) <= linear_speed
            and abs(obj["angular_velocity"]) <= angular_speed)


def footprint_error(corners, goal_center, goal_half_size):
    """Minimum translation norm that fits this oriented footprint in the goal.

    A positive span deficit is included if this orientation cannot fit. The
    result is zero exactly when all footprint corners are inside the rectangle.
    """
    corners = np.asarray(corners)
    lower_shift = np.asarray(goal_center) - goal_half_size - corners.min(axis=0)
    upper_shift = np.asarray(goal_center) + goal_half_size - corners.max(axis=0)
    correction = np.maximum(lower_shift, 0) + np.minimum(upper_shift, 0)
    deficit = np.maximum(lower_shift - upper_shift, 0)
    return float(np.linalg.norm(correction) + np.linalg.norm(deficit))


def footprint_margin(corners, goal_center, goal_half_size):
    """Signed nearest boundary margin: negative means some geometry protrudes."""
    return float(min(np.min(corners - (np.asarray(goal_center) - goal_half_size)),
                     np.min(np.asarray(goal_center) + goal_half_size - corners)))
