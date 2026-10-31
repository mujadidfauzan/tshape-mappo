"""One reward per team transition; diagnostic components sum to that reward."""


def team_reward(previous_distance, distance, success, failed, config,
                previous_footprint_error=None, current_footprint_error=None):
    progress = 0.0 if failed else previous_distance - distance
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
