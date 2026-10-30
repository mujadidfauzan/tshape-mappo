"""Train a shared-policy MAPPO baseline on the M2 Easy environment."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.config import load_mappo_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="MAPPO YAML; defaults to configs/mappo.yaml")
    parser.add_argument("--env-config", type=Path, help="environment YAML; default configs/environment.yaml")
    parser.add_argument("--run-dir", type=Path, help="new directory for logs, model snapshot and checkpoints")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--total-steps", type=int, help="total team environment decisions, including previous steps on resume")
    budget.add_argument("--total-episodes", type=int, help="cumulative completed episodes; replaces the step budget")
    parser.add_argument("--video-interval-episodes", type=int, help="record every N completed episodes; 0 disables videos")
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"])
    parser.add_argument("--resume", type=Path, help="restore training; starts a new episode and output directory")
    args = parser.parse_args()
    overrides = {"total_episodes": args.total_episodes}
    if args.total_steps is not None:
        from dataclasses import replace
        # An explicit step budget also clears an episode budget in the YAML.
        config = replace(load_mappo_config(args.config), total_episodes=None)
    else:
        config = load_mappo_config(args.config)
    config = load_mappo_config(config, total_steps=args.total_steps, **overrides,
                               video_interval_episodes=args.video_interval_episodes, rollout_steps=args.rollout_steps,
                               seed=args.seed, device=args.device)
    if args.resume:
        if args.run_dir is None:
            parser.error("resume requires --run-dir pointing to a new output directory")
        if args.seed is not None or args.rollout_steps is not None or args.config is not None:
            parser.error("resume restores MAPPO settings; only budget, device and video interval may override them")
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        from src.training.trainer import Trainer
    except ModuleNotFoundError as error:
        if error.name == "torch":
            parser.error("PyTorch is missing; install requirements-training.txt first (see README)")
        raise
    run_dir = args.run_dir or ROOT / "results" / f"mappo_seed{config.seed}"
    trainer = Trainer(run_dir, config, args.env_config, args.resume,
                      resume_episode_budget=args.total_episodes,
                      resume_step_budget=args.total_steps,
                      resume_video_interval=args.video_interval_episodes)
    print(f"Run: {trainer.run_dir}; device: {trainer.agent.device}", flush=True)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
