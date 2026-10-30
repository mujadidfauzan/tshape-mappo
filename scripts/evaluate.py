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
    parser.add_argument("--video-episodes", type=int, help="record only the first N episodes; default records all when --video-dir is set")
    parser.add_argument("--torch-threads", type=int, default=1, help="CPU threads for policy inference (default: 1)")
    parser.add_argument("--progress-interval-steps", type=int, default=100)
    parser.add_argument("--quiet", action="store_true", help="hide live progress; still print the final summary")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be positive and seed nonnegative")
    if args.torch_threads < 1 or args.progress_interval_steps < 1:
        parser.error("torch threads and progress interval must be positive")
    if args.video_episodes is not None and args.video_episodes < 0:
        parser.error("video episodes must be nonnegative")
    if args.video_episodes is not None and args.video_dir is None:
        parser.error("--video-episodes requires --video-dir")
    if args.video_dir is not None and sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    def say(message):
        if not args.quiet:
            print(message, flush=True)

    say(f"Loading checkpoint: {args.checkpoint}")
    try:
        import torch
        from src.algorithms.mappo import MAPPO
        from src.envs.config import load_config, load_checkpoint_config
        from src.evaluation.evaluator import evaluate_policy, summarize_reports
        from src.training.trainer import feature_schema
    except ModuleNotFoundError as error:
        if error.name == "torch":
            parser.error("PyTorch is missing; install requirements-training.txt first")
        raise
    torch.set_num_threads(args.torch_threads)
    agent, payload = MAPPO.from_checkpoint(args.checkpoint, args.device)
    training = payload["training"]
    if training.get("feature_schema") != feature_schema():
        parser.error("checkpoint feature schema does not match this environment")
    completed_reports = []
    metadata = {}

    def save_report(complete):
        if args.json is None:
            return
        args.json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json.with_suffix(args.json.suffix + ".tmp")
        temporary.write_text(json.dumps({**metadata, "reports": completed_reports,
            "summary": summarize_reports(completed_reports), "requested_episodes": args.episodes,
            "completed_episodes": sum(not r["interrupted"] for r in completed_reports), "complete": complete},
            indent=2, allow_nan=False) + "\n")
        temporary.replace(args.json)

    def progress(event):
        label = f"[{event['episode']}/{event['episodes']}] seed={event['seed']}"
        kind = event["event"]
        if kind == "episode_start":
            say(f"{label} starting")
        elif kind == "video_start":
            say(f"  Preparing {event['width']}x{event['height']} video: {event['path']}")
        elif kind == "episode_progress":
            distance = event["distance_to_goal"]
            distance = f"{distance:.3f}m" if distance is not None else "n/a"
            say(f"{label} step={event['step']}/{event['max_steps']} "
                f"return={event['episode_return']:.3f} distance={distance} wall={event['wall_seconds']:.1f}s")
        elif kind == "video_finalizing":
            say(f"  Finalizing video: {event['path']}")
        elif kind == "episode_end":
            report = event["report"]
            completed_reports.append(report)
            save_report(False)
            status = "interrupted" if report["interrupted"] else "done"
            say(f"{label} {status} success={report['success']} stable={report['stable']} "
                f"steps={report['episode_step']} return={report['episode_metrics']['episode_return']:.3f} "
                f"reason={report['end_reason']} wall={event['wall_seconds']:.1f}s")

    with tempfile.TemporaryDirectory(prefix="tshape-evaluation-") as temporary:
        if args.env_config:
            config = load_config(args.env_config)
        else:
            model_xml = training["model_xml"]
            if hashlib.sha256(model_xml.encode()).hexdigest() != training["model_sha256"]:
                raise ValueError("checkpoint XML hash mismatch")
            snapshot = Path(temporary) / "arena.xml"
            snapshot.write_text(model_xml)
            config = replace(load_checkpoint_config(training["environment_config"]), model_path=str(snapshot))
        model_xml = config.resolved_model_path.read_text()
        metadata.update(checkpoint=str(args.checkpoint.resolve()),
                      environment_config=config.to_dict(), model_xml=model_xml,
                      model_sha256=hashlib.sha256(model_xml.encode()).hexdigest(),
                      software_training=training.get("software"),
                      software_evaluation={name: str(version(name)) for name in
                                           ("torch", "mujoco", "numpy", "gymnasium", "pettingzoo", "PyYAML")},
                      environment_source="override" if args.env_config else "checkpoint_snapshot",
                      actual_device=str(agent.device), torch_threads=args.torch_threads)
        videos = min(args.episodes, args.video_episodes if args.video_episodes is not None else args.episodes) if args.video_dir else 0
        say(f"Evaluation: device={agent.device} mode={'stochastic' if args.stochastic else 'deterministic'} "
            f"episodes={args.episodes} seeds={args.seed}..{args.seed + args.episodes - 1} "
            f"videos={videos} torch_threads={args.torch_threads}")
        save_report(False)
        try:
            result = evaluate_policy(agent, config, args.episodes, args.seed,
                                     deterministic=not args.stochastic, render=args.render,
                                     trajectory_dir=args.trajectory_dir, video_dir=args.video_dir,
                                     video_width=args.video_width, video_height=args.video_height,
                                     video_crf=args.video_crf, video_episodes=args.video_episodes,
                                     progress_callback=progress,
                                     progress_interval_steps=args.progress_interval_steps)
        except KeyboardInterrupt:
            save_report(False)
            print(f"Evaluation interrupted; {len(completed_reports)}/{args.episodes} completed episodes retained"
                  + (f" in {args.json}" if args.json else ""), flush=True)
            return 130
        complete = len(result["reports"]) == args.episodes and not any(r["interrupted"] for r in result["reports"])
        save_report(complete)
    print(json.dumps(result["summary"], indent=2), flush=True)
    if args.json:
        print(f"Report: {args.json}", flush=True)
    return 1 if any(r["interrupted"] or not r["stable"] for r in result["reports"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
