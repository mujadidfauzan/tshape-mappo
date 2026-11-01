"""One reward per team transition; diagnostic components sum to that reward."""

import math


def wrapped_yaw_error(yaw, goal_yaw):
    """Signed angle difference on [-pi, pi]; the unequal end bars distinguish pi."""
    delta = yaw - goal_yaw
    return math.atan2(math.sin(delta), math.cos(delta))


def team_reward(previous_distance, distance, success, failed, config,
                previous_footprint_error=None, current_footprint_error=None, *, yaw=None):
    progress = 0.0 if failed else previous_distance - distance
    if config.reward_mode == "pose":
        # Invalid physics is an exceptional terminal penalty, not a source of
        # progress or pose reward. In particular, never read a non-finite yaw.
        components = {"distance": 0.0, "orientation": 0.0, "progress": 0.0,
                      "success": 0.0, "failure": 0.0}
        if failed:
            components["failure"] = -config.failure_penalty
        else:
            if yaw is None or not math.isfinite(yaw):
                raise ValueError("pose reward requires a finite object yaw")
            distance_cost = distance / (distance + config.distance_scale)
            angle_cost = abs(wrapped_yaw_error(yaw, config.goal_yaw)) / math.pi
            # Goal alignment matters near the target. Farther away the object
            # can still rotate while passing the fixed obstacle openings.
            proximity = math.exp(-distance / config.orientation_distance_scale)
            components.update(distance=-config.distance_weight * distance_cost,
                              orientation=-config.orientation_weight * proximity * angle_cost,
                              progress=config.progress_weight * progress,
                              success=config.success_reward if success else 0.0)
        components = {key: float(value) for key, value in components.items()}
        return float(sum(components.values())), float(progress), components

    # Historical objectives remain available for saved runs and ablations.
    shaped = progress
    if config.reward_mode == "footprint_potential":
        # Phi=-error; true terminal potentials are zero. A time-limit truncation
        # keeps its potential, consistent with GAE's continuation bootstrap.
        next_potential = 0.0 if success or failed else -current_footprint_error
        shaped = config.shaping_gamma * next_potential + previous_footprint_error
    components = {
        "progress": float(config.progress_weight * shaped),
        "time": -float(config.time_penalty),
        "success": float(config.success_reward if success else 0.0),
        "failure": -float(config.failure_penalty if failed else 0.0),
    }
    return float(sum(components.values())), float(progress), components
