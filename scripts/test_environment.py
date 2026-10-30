"""Run Easy/Medium/Hard smoke episodes, API checks, and trajectory logs."""

import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.envs.config import load_config
from src.envs.scripted import scripted_actions
from src.envs.tshape_env import TShapeParallelEnv


def run_episode(env, seed=42, policy="random", trajectory_path=None):
    observations, infos = env.reset(seed=seed)
    names = env.possible_agents
    states = [env.state()] if trajectory_path is not None else []
    times = [float(env.physics.data.time)] if trajectory_path is not None else []
    observation_log = [np.stack([observations[n] for n in names])] if trajectory_path is not None else []
    actions_log, reward_log, terminated_log, truncated_log = [], [], [], []
    interrupted = False
    final_info = infos[names[0]]
    while env.agents:
        started = time.perf_counter()
        if policy == "zero":
            actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}
        elif policy == "random":
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        elif policy == "scripted":
            actions = scripted_actions(env)
        else:
            raise ValueError(f"unknown policy: {policy}")
        observations, rewards, terminated, truncated, infos = env.step(actions)
        final_info = infos[names[0]]
        if trajectory_path is not None:
            states.append(env.state())
            times.append(float(env.physics.data.time))
            observation_log.append(np.stack([observations[n] for n in names]))
            actions_log.append(np.stack([actions[n] for n in names]))
            reward_log.append(rewards[names[0]])
            terminated_log.append(terminated[names[0]])
            truncated_log.append(truncated[names[0]])
        if env.render_mode == "human":
            env.render()
            if not env.viewer.is_running():
                interrupted = True
                break
            remaining = env.physics.control_dt - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
    if trajectory_path is not None:
        trajectory_path = Path(trajectory_path)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trajectory_path, states=np.asarray(states, dtype=np.float32),
            time_s=np.asarray(times, dtype=np.float64),
            observations=np.asarray(observation_log, dtype=np.float32),
            actions=np.asarray(actions_log, dtype=np.float32),
            rewards=np.asarray(reward_log, dtype=np.float32),
            terminated=np.asarray(terminated_log, dtype=bool),
            truncated=np.asarray(truncated_log, dtype=bool), seed=np.asarray(seed),
        )
    return {"seed": seed, "policy": policy, "interrupted": interrupted,
            **final_info, "trajectory": str(trajectory_path) if trajectory_path is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="environment YAML; default configs/environment.yaml")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", choices=["random", "zero", "scripted"], default="random")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-randomize", action="store_true", help="use the fixed initial pose and spawn layout")
    parser.add_argument("--api-test", action="store_true", help="run PettingZoo Parallel API test instead of episodes")
    parser.add_argument("--json", type=Path, help="save episode results, configuration, and software versions")
    parser.add_argument("--trajectory-dir", type=Path, help="save per-episode NPZ state/observation/action trajectories")
    parser.add_argument("--require-success", action="store_true", help="exit nonzero if any episode does not succeed")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("--episodes must be positive and --seed nonnegative")
    config = load_config(args.config, max_steps=args.max_steps)
    if args.no_randomize:
        config = replace(config, randomize_reset=False)
    env = TShapeParallelEnv(config=config, render_mode="human" if args.render else None)
    reports = []
    try:
        if args.api_test:
            from pettingzoo.test import parallel_api_test
            parallel_api_test(env, num_cycles=config.max_steps + 1)
            return 0
        for episode in range(args.episodes):
            seed = args.seed + episode
            path = (args.trajectory_dir / f"{args.policy}_seed_{seed}.npz"
                    if args.trajectory_dir is not None else None)
            report = run_episode(env, seed, args.policy, path)
            reports.append(report)
            print(f"Episode {episode + 1}: return={report['episode_metrics']['episode_return']:.3f}, "
                  f"success={report['success']}, stable={report['stable']}, "
                  f"steps={report['episode_step']}, reason={report['end_reason']}")
            if report["failure_reason"] is not None:
                print(f"  failure={report['failure_reason']}, "
                      f"max_penetration={1000 * report['episode_metrics']['max_penetration_m']:.3f} mm")
                details = report.get("failure_details") or {}
                contacts = details.get("contacts", [])
                if contacts:
                    worst = contacts[0]
                    print(f"  deepest_contact={' / '.join(worst['geoms'])}, "
                          f"penetration={1000 * worst['penetration_m']:.3f} mm")
            if report["interrupted"]:
                break
        if args.json:
            payload = {
                "config": config.to_dict(), "model_xml": env.physics.model_path.read_text(),
                "model_sha256": hashlib.sha256(env.physics.model_path.read_bytes()).hexdigest(),
                "software": {"python": platform.python_version(),
                             **{name: version(name) for name in ("mujoco", "numpy", "gymnasium", "pettingzoo", "PyYAML")}},
                "reports": reports,
                "summary": {"episodes": len(reports),
                            "success_rate": sum(r["success"] for r in reports) / max(1, len(reports)),
                            "stable_rate": sum(r["stable"] for r in reports) / max(1, len(reports))},
            }
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            print(f"Report: {args.json}")
    finally:
        env.close()
    failed = any(not r["stable"] or r["interrupted"] for r in reports)
    if args.require_success:
        failed |= any(not r["success"] for r in reports)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
