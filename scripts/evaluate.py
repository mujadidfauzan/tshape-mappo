"""Evaluate an Easy checkpoint with its saved agent count and spawn settings."""

import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("--device",choices=["cpu","cuda","auto"],default="auto")
    parser.add_argument("--episodes",type=int,default=1)
    parser.add_argument("--seed",type=int,default=1000000,help="seed for agent spawns and action sampling")
    parser.add_argument("--stochastic",action="store_true")
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--video-dir",type=Path,help="save the first episode as policy.mp4")
    parser.add_argument("--json",type=Path,help="optional single report file")
    args=parser.parse_args()
    if not args.render and sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL","egl")
    print(f"Loading Easy policy: {args.checkpoint}",flush=True)
    import torch
    torch.set_num_threads(1)
    from src.evaluation.checkpoints import loaded_policy
    from src.evaluation.evaluator import evaluate_policy
    from src.training.artifacts import write_json
    def progress(event):
        if event["event"]=="episode_start":
            print(f"Episode {event['episode']}/{event['episodes']} started",flush=True)
        elif event["event"]=="episode_progress":
            print(f"  step={event['step']} distance={event['distance']}",flush=True)
        elif event["event"]=="episode_end":
            r=event["report"]
            print(f"  return={r['episode_metrics']['episode_return']:.3f} "
                  f"success={r['success']} stable={r['stable']} steps={r['episode_step']}",flush=True)
    with loaded_policy(args.checkpoint,args.device) as (agent,config,payload):
        result=evaluate_policy(agent,config,episodes=args.episodes,seed=args.seed,
                               deterministic=not args.stochastic,render=args.render,
                               video_dir=args.video_dir,progress_callback=progress)
        result.update(checkpoint=str(args.checkpoint.resolve()),task=payload["training"]["task"],
                      model_sha256=payload["training"]["model_sha256"])
    if args.json:
        write_json(args.json,result)
    s=result["summary"]
    print(f"Success={s['success_rate']:.1%}; mean return={s['mean_return']:.3f}",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
