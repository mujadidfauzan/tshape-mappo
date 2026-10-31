"""Independent agent-count/reward experiments at a fixed team-step budget."""

from dataclasses import replace
from itertools import product
from pathlib import Path

import yaml

from src.envs.config import load_config
from .config import load_mappo_config
from .curriculum import Curriculum


def experiment_jobs(source, *, device=None, total_steps=None):
    definition = yaml.safe_load(Path(source).read_text()) if isinstance(source, (str, Path)) else source
    allowed = {"environment_config", "mappo_config", "num_agents", "seeds", "reward_modes", "curriculum", "total_steps"}
    if not isinstance(definition, dict) or set(definition) - allowed:
        raise ValueError("unknown experiment matrix fields")
    env = load_config(definition.get("environment_config"))
    algo = load_mappo_config(definition.get("mappo_config"), device=device,
                             total_steps=total_steps if total_steps is not None else definition.get("total_steps"))
    counts, seeds = definition.get("num_agents", [2, 4, 8, 16]), definition.get("seeds", [11, 22, 33, 44, 55])
    modes = definition.get("reward_modes", ["baseline"])
    for name, values in (("num_agents", counts), ("seeds", seeds), ("reward_modes", modes)):
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be a nonempty list without duplicates")
    if definition.get("curriculum"):
        algo = replace(algo, curriculum=definition["curriculum"])
    jobs = []
    for n, seed, mode in product(counts, seeds, modes):
        environment = replace(env, num_agents=n, reward_mode=mode, shaping_gamma=algo.gamma)
        config = replace(algo, seed=seed)
        if config.curriculum:
            Curriculum.load(config.curriculum).validate_configs(environment)
        jobs.append((f"agents_{n:02d}_seed_{seed}_{mode}", config, environment))
    return definition, jobs
