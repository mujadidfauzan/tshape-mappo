"""Paired deterministic/stochastic checkpoint evaluation on reserved held-out seeds."""

import argparse
import csv
import json
import hashlib
from importlib.metadata import version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--env-config", type=Path, help="explicit alternative task; logged as an override")
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 200000 or args.seed + args.episodes > 300000:
        parser.error("held-out seeds must lie in the reserved range [200000, 300000)")
    import torch
    torch.set_num_threads(1)
    from src.evaluation.checkpoints import loaded_policy
    from src.evaluation.evaluator import evaluate_policy, wilson_interval
    from src.evaluation.metrics import aggregate_benchmarks
    from src.envs.model import model_with_agents
    from src.training.artifacts import write_json
    if len({str(path.resolve()) for path in args.checkpoints}) != len(args.checkpoints):
        parser.error("duplicate checkpoints would duplicate the same policy")
    args.output.mkdir(parents=True, exist_ok=False)
    specs, summaries = [], []
    with (args.output / "episodes.jsonl").open("w") as episode_log, (args.output / "summary.csv").open("w") as summary_log:
        writer = None
        for checkpoint in args.checkpoints:
            with loaded_policy(checkpoint, args.device, args.env_config) as (agent, env, payload):
                cfg = agent.config
                validation = [(cfg.eval_seed, cfg.eval_seed + cfg.eval_episodes)]
                curriculum = payload["training"].get("curriculum_definition")
                if curriculum:
                    start = curriculum.get("validation_seed", 150000)
                    validation.append((start, start + curriculum.get("validation_episodes", 20)))
                if any(args.seed < end and args.seed + args.episodes > start for start, end in validation):
                    raise ValueError("held-out evaluation overlaps this checkpoint's validation seeds")
                xml = model_with_agents(env.resolved_model_path.read_text(), env.num_agents)
                actual_hash = hashlib.sha256(xml.encode()).hexdigest()
                task_id = hashlib.sha256(json.dumps({"environment": {k:v for k,v in env.to_dict().items() if k != "model_path"},
                    "model_sha256": actual_hash}, sort_keys=True).encode()).hexdigest()[:12]
                protocol_id = hashlib.sha256(json.dumps({"mappo": {k:v for k,v in cfg.to_dict().items() if k not in ("seed","device","curriculum")},
                    "curriculum": curriculum}, sort_keys=True).encode()).hexdigest()[:12]
                spec = {"checkpoint": str(checkpoint.resolve()), "num_agents": env.num_agents,
                        "actual_device": str(agent.device), "torch_threads": 1,
                        "software_training": payload["training"].get("software"),
                        "software_evaluation": {name: str(version(name)) for name in ("torch", "numpy", "mujoco", "pettingzoo")},
                        "environment_config": {**env.to_dict(), "model_path": "<saved XML>"},
                        "model_sha256": actual_hash, "checkpoint_model_sha256": payload["training"]["model_sha256"],
                        "environment_override": str(args.env_config) if args.env_config else None,
                        "saved_reset_distribution": payload["training"]["environment_config"].get("reset_distribution")}
                specs.append(spec)
                write_json(args.output / "manifest.json", {"seed": args.seed, "episodes": args.episodes,
                    "requested_checkpoints": [str(p.resolve()) for p in args.checkpoints],
                    "modes": ["deterministic", "stochastic"], "cases": specs, "complete": False})
                for deterministic in (True, False):
                    mode = "deterministic" if deterministic else "stochastic"
                    print(f"{checkpoint}: {mode}, {args.episodes} episodes", flush=True)
                    def progress(event):
                        if event["event"] == "episode_end":
                            episode_log.write(json.dumps({"checkpoint": str(checkpoint.resolve()), "mode": mode,
                                "num_agents": env.num_agents, **event["report"]}, allow_nan=False) + "\n")
                            episode_log.flush()
                            print(f"  {event['episode']}/{args.episodes} success={event['report']['success']}", flush=True)
                    result = evaluate_policy(agent, env, args.episodes, args.seed,
                        deterministic=deterministic, progress_callback=progress)
                    s = result["summary"]
                    low, high = wilson_interval(sum(r["success"] for r in result["reports"]), len(result["reports"]))
                    s.update(success_ci95_low=low, success_ci95_high=high)
                    row = {"checkpoint": str(checkpoint.resolve()), "mode": mode, "num_agents": env.num_agents,
                           "training_seed": cfg.seed, "training_steps": payload["training"].get("environment_steps"),
                           "difficulty": env.difficulty, "reward_mode": env.reward_mode, "checkpoint_kind": checkpoint.stem,
                           "task_id": task_id, "protocol_id": protocol_id,
                           **{key: s.get(key) for key in ("episodes", "success_rate", "success_ci95_low", "success_ci95_high",
                               "stable_rate", "mean_return", "std_return", "mean_completion_time_s", "footprint_error_m_mean", "gate_clear_fraction_mean",
                               "object_path_length_m_mean", "control_effort_N2_s_mean", "control_effort_per_agent_N2_s_mean", "wall_collision_events_mean")}}
                    for category in ("success", "gate_blocked", "gate_not_reached", "contact_lost", "goal_not_settled",
                                     "goal_footprint_outside", "physics_failure"):
                        row[category + "_episodes"] = s["failure_categories"].get(category, 0)
                    if writer is None:
                        writer = csv.DictWriter(summary_log, fieldnames=list(row)); writer.writeheader()
                    writer.writerow(row); summary_log.flush(); summaries.append(row)
                    groups = aggregate_benchmarks(summaries)
                    with (args.output / "summary_by_group.csv").open("w") as grouped:
                        group_writer = csv.DictWriter(grouped, fieldnames=list(groups[0]))
                        group_writer.writeheader(); group_writer.writerows(groups)
    write_json(args.output / "manifest.json", {"seed": args.seed, "episodes": args.episodes,
        "requested_checkpoints": [str(p.resolve()) for p in args.checkpoints],
        "modes": ["deterministic", "stochastic"], "cases": specs, "complete": True})
    print(f"Saved {len(summaries)} comparisons to {args.output / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
