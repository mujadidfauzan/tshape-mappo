"""Start or resume MAPPO on the fixed Easy layout."""

import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True,help="new output directory")
    parser.add_argument("--config",type=Path,help="MAPPO YAML")
    parser.add_argument("--env-config",type=Path,help="fixed Easy YAML")
    parser.add_argument("--total-steps",type=int,help="cumulative team timestep budget")
    parser.add_argument("--device",choices=["cpu","cuda","auto"])
    parser.add_argument("--seed",type=int,help="policy/exploration seed; the layout stays fixed")
    parser.add_argument("--rollout-steps",type=int)
    parser.add_argument("--eval-interval-steps",type=int)
    parser.add_argument("--video-interval-steps",type=int)
    parser.add_argument("--tensorboard",action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument("--resume",type=Path,help="fixed Easy checkpoint; requires a larger total budget")
    args=parser.parse_args()
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL","egl")
    from src.training.config import load_mappo_config
    if args.resume:
        if args.total_steps is None:
            parser.error("resume requires --total-steps with a larger cumulative budget")
        if any(v is not None for v in [args.config,args.env_config,args.seed,args.rollout_steps]):
            parser.error("resume restores the task, seed, architecture and rollout settings")
        import torch
        from src.algorithms.mappo import CHECKPOINT_VERSION
        payload=torch.load(args.resume,map_location="cpu",weights_only=True)
        if payload.get("format_version")!=CHECKPOINT_VERSION:
            parser.error("old experiment checkpoints are unsupported; start a fresh fixed Easy run")
        base=payload["mappo_config"]
    else:
        base=args.config
    cfg=load_mappo_config(base,total_steps=args.total_steps,device=args.device,seed=args.seed,
                         rollout_steps=args.rollout_steps,eval_interval_steps=args.eval_interval_steps,
                         video_interval_steps=args.video_interval_steps,tensorboard=args.tensorboard)
    from src.training.trainer import Trainer
    trainer=Trainer(args.run_dir,cfg,args.env_config,resume=args.resume)
    print(f"Fixed Easy: {trainer.run_dir}; device={trainer.agent.device}",flush=True)
    trainer.train()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
