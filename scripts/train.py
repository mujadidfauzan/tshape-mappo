"""Train a shared-policy MAPPO baseline on the M2 Easy environment."""

import argparse
from dataclasses import replace
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
    parser.add_argument("--total-steps", "--total-timesteps", dest="total_steps", type=int,
                        help="cumulative team environment decisions, including saved steps on resume")
    parser.add_argument("--video-interval-steps", type=int, help="record every N steps at rollout boundaries; 0 disables videos")
    parser.add_argument("--eval-interval-steps", type=int)
    parser.add_argument("--checkpoint-interval-steps", type=int)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reward-log-interval-steps", type=int, help="sample raw reward every N steps; 1 logs every step; 0 disables raw samples")
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"])
    parser.add_argument("--resume", type=Path, help="restore training; starts a new episode and output directory")
    args = parser.parse_args()
    config = load_mappo_config(args.config)
    if args.total_steps is not None:
        config = replace(config, total_episodes=None)
    config = load_mappo_config(config, total_steps=args.total_steps,
                               video_interval_steps=args.video_interval_steps,
                               eval_interval_steps=args.eval_interval_steps,
                               checkpoint_interval_steps=args.checkpoint_interval_steps,
                               tensorboard=args.tensorboard, reward_log_interval_steps=args.reward_log_interval_steps,
                               rollout_steps=args.rollout_steps, seed=args.seed, device=args.device)
    if args.resume:
        if args.run_dir is None:
            parser.error("resume requires --run-dir pointing to a new output directory")
        if args.seed is not None or args.rollout_steps is not None or args.config is not None:
            parser.error("resume restores MAPPO settings; only step budget, device, logging and schedule flags may override them")
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
                      resume_step_budget=args.total_steps,
                      resume_video_interval=args.video_interval_steps,
                      resume_eval_interval=args.eval_interval_steps,
                      resume_checkpoint_interval=args.checkpoint_interval_steps,
                      resume_tensorboard=args.tensorboard,
                      resume_reward_interval=args.reward_log_interval_steps)
    print(f"Run: {trainer.run_dir}; device: {trainer.agent.device}", flush=True)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
