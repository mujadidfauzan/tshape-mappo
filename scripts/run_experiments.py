"""Run independent seed/agent-count experiments; use --dry-run to inspect the matrix."""

import argparse
import csv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new experiment directory")
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"])
    parser.add_argument("--total-steps", type=int, help="override the same team-step budget for every job")
    parser.add_argument("--job-index", type=int, help="run one zero-based index from the matrix")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from src.training.experiments import experiment_jobs
    definition, jobs = experiment_jobs(args.matrix, device=args.device, total_steps=args.total_steps)
    if args.job_index is not None:
        if not 0 <= args.job_index < len(jobs):
            parser.error("job-index outside the experiment matrix")
        selected = [(args.job_index, jobs[args.job_index])]
    else:
        selected = list(enumerate(jobs))
    for index, (name, config, env) in selected:
        print(f"[{index}] {name}: arena={env.difficulty} steps={config.total_steps} curriculum={config.curriculum}", flush=True)
    if args.dry_run:
        print(f"{len(selected)} independent runs; {sum(j[1][1].total_steps for j in selected)} team steps total", flush=True)
        return 0
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    import yaml
    from src.training.trainer import Trainer
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.yaml").write_text(yaml.safe_dump(definition))
    fields = ["job_index", "run", "num_agents", "seed", "reward_mode", "total_steps", "status",
              "best_validation_success_rate", "error"]
    rows = [{"job_index": i, "run": name, "num_agents": env.num_agents, "seed": cfg.seed,
             "reward_mode": env.reward_mode, "total_steps": cfg.total_steps, "status": "planned",
             "best_validation_success_rate": "", "error": ""} for i,(name,cfg,env) in selected]
    def save():
        path = args.output / "experiments.csv"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader();writer.writerows(rows)
        temporary.replace(path)
    save()
    for row, (_, (name, config, env)) in zip(rows, selected):
        row["status"] = "running";save()
        try:
            summary = Trainer(args.output / name, config, env).train()
            row["status"] = "complete"
            row["best_validation_success_rate"] = summary["best_eval_success_rate"]
        except BaseException as error:
            row["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            row["error"] = str(error);save()
            raise
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
