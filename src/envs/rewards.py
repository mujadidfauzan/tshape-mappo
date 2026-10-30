"""One reward per team transition; diagnostic components sum to that reward."""


def team_reward(previous_distance, distance, success, failed, config):
    progress = 0.0 if failed else previous_distance - distance
    components = {
        "progress": float(config.progress_weight * progress),
        "time": -float(config.time_penalty),
        "success": float(config.success_reward if success else 0.0),
        "failure": -float(config.failure_penalty if failed else 0.0),
    }
    return float(sum(components.values())), float(progress), components
