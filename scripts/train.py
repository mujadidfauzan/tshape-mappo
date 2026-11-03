"""Start or resume Easy MAPPO with configurable agent count and seeded spawns."""

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
    parser.add_argument("--env-config",type=Path,help="Easy YAML")
    parser.add_argument("--num-agents",type=int,choices=range(1,17),metavar="1..16",
                        help="agent count, fixed for this run; default from environment YAML (4)")
    parser.add_argument("--randomize-agents",action=argparse.BooleanOptionalAction,default=None,
                        help="randomize agent spawns near the object; enabled in default YAML")
    parser.add_argument("--total-steps",type=int,help="cumulative team timestep budget")
    parser.add_argument("--device",choices=["cpu","cuda","auto"])
    parser.add_argument("--seed",type=int,help="seed for networks, actions and agent spawns")
    parser.add_argument("--rollout-steps",type=int)
    parser.add_argument("--eval-interval-steps",type=int)
    parser.add_argument("--eval-episodes",type=int,help="seeded monitoring layouts per evaluation (default 10)")
    parser.add_argument("--video-interval-steps",type=int)
    parser.add_argument("--tensorboard",action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument("--resume",type=Path,help="Easy checkpoint; restores agent count and spawn settings")
    args=parser.parse_args()
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL","egl")
    from src.training.config import load_mappo_config
    if args.resume:
        if args.total_steps is None:
            parser.error("resume requires --total-steps with a larger cumulative budget")
        if any(v is not None for v in [args.config,args.env_config,args.seed,args.rollout_steps,
                                       args.num_agents,args.randomize_agents]):
            parser.error("resume restores the task, seed, architecture and rollout settings")
        import torch
        from src.algorithms.mappo import CHECKPOINT_VERSION
        payload=torch.load(args.resume,map_location="cpu",weights_only=True)
        if payload.get("format_version")!=CHECKPOINT_VERSION:
            parser.error("old experiment checkpoints are unsupported; start a fresh Easy run")
        base=payload["mappo_config"]
    else:
        base=args.config
    cfg=load_mappo_config(base,total_steps=args.total_steps,device=args.device,seed=args.seed,
                         rollout_steps=args.rollout_steps,eval_interval_steps=args.eval_interval_steps,
                         eval_episodes=args.eval_episodes,
                         video_interval_steps=args.video_interval_steps,tensorboard=args.tensorboard)
    from src.envs.config import load_config
    env_config = None
    if not args.resume:
        env_config = load_config(args.env_config,num_agents=args.num_agents,randomize_agents=args.randomize_agents)
    from src.training.trainer import Trainer
    trainer=Trainer(args.run_dir,cfg,env_config,resume=args.resume)
    print(f"Easy: {trainer.run_dir}; device={trainer.agent.device}; "
          f"agents={trainer.environment_config.num_agents}; "
          f"randomize_agents={trainer.environment_config.randomize_agents}",flush=True)
    trainer.train()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
