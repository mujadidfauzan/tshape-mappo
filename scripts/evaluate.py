"""Evaluate actor execution from a checkpoint and its frozen MuJoCo model."""

import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stochastic", action="store_true", help="sample actions; default uses tanh(mean)")
    parser.add_argument("--env-config", type=Path, help="explicit alternative environment for transfer experiments")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--video-dir", type=Path, help="save one top-view MP4 per evaluation episode")
    parser.add_argument("--video-width", type=int, default=1920)
    parser.add_argument("--video-height", type=int, default=1080)
    parser.add_argument("--video-crf", type=int, default=18)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be positive and seed nonnegative")
    if args.video_dir is not None and sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        from src.algorithms.mappo import MAPPO
        from src.envs.config import load_config
        from src.evaluation.evaluator import evaluate_policy
        from src.training.trainer import feature_schema
    except ModuleNotFoundError as error:
        if error.name == "torch":
            parser.error("PyTorch is missing; install requirements-training.txt first")
        raise
    agent, payload = MAPPO.from_checkpoint(args.checkpoint, args.device)
    training = payload["training"]
    if training.get("feature_schema") != feature_schema():
        parser.error("checkpoint feature schema does not match this environment")
    with tempfile.TemporaryDirectory(prefix="tshape-evaluation-") as temporary:
        if args.env_config:
            config = load_config(args.env_config)
        else:
            model_xml = training["model_xml"]
            if hashlib.sha256(model_xml.encode()).hexdigest() != training["model_sha256"]:
                raise ValueError("checkpoint XML hash mismatch")
            snapshot = Path(temporary) / "arena.xml"
            snapshot.write_text(model_xml)
            config = replace(load_config(training["environment_config"]), model_path=str(snapshot))
        result = evaluate_policy(agent, config, args.episodes, args.seed,
                                 deterministic=not args.stochastic, render=args.render,
                                 trajectory_dir=args.trajectory_dir, video_dir=args.video_dir,
                                 video_width=args.video_width, video_height=args.video_height,
                                 video_crf=args.video_crf)
        model_xml = config.resolved_model_path.read_text()
        result.update(checkpoint=str(args.checkpoint.resolve()),
                      environment_config=config.to_dict(), model_xml=model_xml,
                      model_sha256=hashlib.sha256(model_xml.encode()).hexdigest(),
                      software_training=training.get("software"),
                      software_evaluation={name: str(version(name)) for name in
                                           ("torch", "mujoco", "numpy", "gymnasium", "pettingzoo", "PyYAML")},
                      environment_source="override" if args.env_config else "checkpoint_snapshot")
    for report in result["reports"]:
        print(f"seed={report['seed']} success={report['success']} stable={report['stable']} "
              f"steps={report['episode_step']} reason={report['end_reason']}")
    print(json.dumps(result["summary"], indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Report: {args.json}")
    return 1 if any(r["interrupted"] or not r["stable"] for r in result["reports"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
