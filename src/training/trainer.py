"""Single-environment CPU-friendly MAPPO trainer with frozen model snapshots."""

from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from src.algorithms.mappo import MAPPO
from src.algorithms.rollout_buffer import RolloutBuffer
from src.envs.config import load_config
from src.envs.observations import ACTOR_FEATURES, STATE_FEATURES
from src.envs.tshape_env import TShapeParallelEnv
from src.evaluation.evaluator import evaluate_policy
from .config import load_mappo_config


def feature_schema():
    return {"actor": [[name, size] for name, size in ACTOR_FEATURES],
            "critic": [[name, size] for name, size in STATE_FEATURES]}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


class Trainer:
    def __init__(self, run_dir, config=None, environment_config=None, resume=None,
                 resume_episode_budget=None, resume_step_budget=None, resume_video_interval=None):
        cfg = self.config = load_mappo_config(config)
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.steps = self.updates = self.episode_index = 0
        self.completed_episodes = self.last_video_episode = 0
        self.best_score = (-1.0, -float("inf"))
        self.agent = None
        self.resume_source = str(Path(resume).resolve()) if resume is not None else None
        self.resume_best_score = None
        if resume is None:
            base_env = load_config(environment_config)
            model_xml = base_env.resolved_model_path.read_text()
        else:
            if environment_config is not None:
                raise ValueError("resume uses the checkpoint environment; do not supply --env-config")
            self.agent, payload = MAPPO.from_checkpoint(resume, device=cfg.device, load_optimizers=True)
            training = payload["training"]
            if training.get("feature_schema") != feature_schema():
                raise ValueError("checkpoint observation schema does not match this implementation")
            # Budget, device and recording cadence may change on resume.
            previous_cfg = load_mappo_config(payload["mappo_config"])
            episode_budget = (resume_episode_budget if resume_episode_budget is not None else
                              None if resume_step_budget is not None else
                              cfg.total_episodes if cfg.total_episodes is not None else previous_cfg.total_episodes)
            cfg = self.config = replace(previous_cfg, total_steps=cfg.total_steps, device=cfg.device,
                                       total_episodes=episode_budget,
                                       video_interval_episodes=(previous_cfg.video_interval_episodes
                                           if resume_video_interval is None else resume_video_interval))
            self.agent.config = cfg
            torch.set_num_threads(cfg.torch_threads)
            self.steps = training["environment_steps"]
            self.updates = training["updates"]
            self.completed_episodes = training.get("completed_episodes", training["episode_index"])
            self.last_video_episode = training.get("last_video_episode", 0)
            # Resume starts a fresh episode: no warm-start of partially collected
            # MuJoCo state. Use the next episode seed so the dropped start isn't reused.
            self.episode_index = training["episode_index"] + 1
            # The previous run retains its best.pt; pick a best.pt for this new
            # directory from evaluations actually made during this resumed run.
            self.resume_best_score = training.get("best_score")
            self.rng.bit_generator.state = training["minibatch_rng_state"]
            torch.set_rng_state(training["torch_rng_state"])
            if self.agent.device.type == "cuda" and training.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(training["cuda_rng_state"], self.agent.device)
            base_env = load_config(training["environment_config"])
            model_xml = training["model_xml"]
            if hashlib.sha256(model_xml.encode()).hexdigest() != training["model_sha256"]:
                raise ValueError("checkpoint XML hash mismatch")
            if self._budget_reached():
                raise ValueError("training budget must exceed the steps/episodes already in the checkpoint")
        # Require a new directory; never overwrite another experiment's results.
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.model_xml = model_xml
        self.model_hash = hashlib.sha256(model_xml.encode()).hexdigest()
        snapshot = self.run_dir / "arena.xml"
        snapshot.write_text(model_xml)
        self.environment_config = replace(base_env, model_path=str(snapshot))
        self.env = TShapeParallelEnv(config=self.environment_config)
        dimensions = {"observation_size": self.env.obs_dim, "state_size": self.env.state_space.shape[0],
                      "action_size": 2, "num_agents": len(self.env.possible_agents)}
        if self.agent is None:
            self.agent = MAPPO(**dimensions, config=cfg)
        elif self.agent.dimensions != dimensions:
            raise ValueError("checkpoint dimensions do not match the environment")
        interval = cfg.video_interval_episodes
        next_video = (self.completed_episodes // interval + 1) * interval if interval else None
        maximum_episodes = (cfg.total_episodes if cfg.total_episodes is not None else
                            self.completed_episodes + cfg.total_steps - self.steps)
        if next_video is not None and maximum_episodes >= next_video:
            from src.evaluation.video import check_video_support
            try:
                check_video_support(self.environment_config, cfg.video_width, cfg.video_height, cfg.video_crf)
            except Exception:
                self.env.close()
                raise
        self.software = {name: str(version(name)) for name in
                         ("torch", "mujoco", "numpy", "gymnasium", "pettingzoo", "PyYAML")}
        manifest = {"mappo": cfg.to_dict(), "environment": self.environment_config.to_dict(),
                    "software": self.software, "model_sha256": self.model_hash,
                    "feature_schema": feature_schema(), "resume_source": self.resume_source,
                    "previous_run_best_score": self.resume_best_score,
                    "actual_device": str(self.agent.device),
                    "completed_episodes_at_start": self.completed_episodes,
                    "resume_starts_new_episode": resume is not None}
        write_json(self.run_dir / "manifest.json", manifest)
        (self.run_dir / "mappo.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
        (self.run_dir / "environment.yaml").write_text(yaml.safe_dump(self.environment_config.to_dict()))

    def checkpoint(self, name):
        if name not in {"best.pt", "latest.pt"}:
            raise ValueError("training retains only best.pt and latest.pt")
        training = {"environment_config": self.environment_config.to_dict(),
                    "model_xml": self.model_xml, "model_sha256": self.model_hash,
                    "feature_schema": feature_schema(), "software": self.software,
                    "environment_steps": self.steps, "updates": self.updates,
                    "episode_index": self.episode_index,
                    "completed_episodes": self.completed_episodes,
                    "last_video_episode": self.last_video_episode,
                    "best_score": list(self.best_score) if self.best_score[0] >= 0 else None,
                    "minibatch_rng_state": self.rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (torch.cuda.get_rng_state(self.agent.device)
                                       if self.agent.device.type == "cuda" else None)}
        self.agent.save_checkpoint(self.run_dir / "checkpoints" / name, training)

    def _budget_reached(self):
        if self.config.total_episodes is not None:
            return self.completed_episodes >= self.config.total_episodes
        return self.steps >= self.config.total_steps

    def _record_video(self):
        episode = self.completed_episodes
        directory = self.run_dir / "videos" / f"episode_{episode:06d}"
        self.last_video_episode = episode
        self.checkpoint("latest.pt")
        result = evaluate_policy(
            self.agent, self.environment_config, episodes=1, seed=self.config.eval_seed,
            video_dir=directory, video_width=self.config.video_width,
            video_height=self.config.video_height, video_crf=self.config.video_crf)
        result.update(training_episodes=episode, environment_steps=self.steps,
                      updates=self.updates, model_sha256=self.model_hash,
                      checkpoint=str(self.run_dir / "checkpoints" / "latest.pt"),
                      checkpoint_is_overwritten=True)
        write_json(directory / "evaluation.json", result)
        print(f"  video episode={episode}: {result['reports'][0]['video']['path']}", flush=True)

    def episode_seed(self):
        """Skip the fixed validation seed range even during a very long run."""
        cfg = self.config
        first, last = cfg.eval_seed, cfg.eval_seed + cfg.eval_episodes - 1
        candidate = cfg.seed + self.episode_index
        if first <= cfg.seed <= last:
            candidate += last - cfg.seed + 1
        elif cfg.seed < first <= candidate:
            candidate += cfg.eval_episodes
        return candidate

    def train(self):
        cfg, env, agent = self.config, self.env, self.agent
        names = env.possible_agents
        observations, _ = env.reset(seed=self.episode_seed())
        started = time.perf_counter()
        try:
            with (self.run_dir / "episodes.jsonl").open("a") as episode_log, (self.run_dir / "updates.jsonl").open("a") as update_log:
                while not self._budget_reached():
                    capacity = (cfg.rollout_steps if cfg.total_episodes is not None else
                                min(cfg.rollout_steps, cfg.total_steps - self.steps))
                    buffer = RolloutBuffer(capacity, **agent.dimensions)
                    completed = []
                    for _ in range(capacity):
                        obs_array = np.stack([observations[name] for name in names])
                        state = env.state()
                        actions, latent, log_probs = agent.act(obs_array)
                        value = float(agent.value(state))
                        next_obs, rewards, terminated, truncated, infos = env.step(dict(zip(names, actions)))
                        term = terminated[names[0]]
                        trunc = truncated[names[0]]
                        # Read final state BEFORE reset, including on time-limit truncation.
                        next_value = 0.0 if term else float(agent.value(env.state()))
                        buffer.add(obs_array, state, latent, log_probs, rewards[names[0]],
                                   value, next_value, term, trunc)
                        self.steps += 1
                        observations = next_obs
                        if term or trunc:
                            info = infos[names[0]]
                            record = {"environment_steps": self.steps, "episode_index": self.episode_index,
                                      "completed_episodes": self.completed_episodes + 1,
                                      "seed": self.episode_seed(), **info}
                            episode_log.write(json.dumps(record, allow_nan=False) + "\n")
                            episode_log.flush()
                            completed.append(info)
                            self.completed_episodes += 1
                            self.episode_index += 1
                            if (cfg.video_interval_episodes and
                                    self.completed_episodes % cfg.video_interval_episodes == 0 and
                                    self.completed_episodes > self.last_video_episode):
                                self._record_video()
                            if self._budget_reached():
                                break
                            observations, _ = env.reset(seed=self.episode_seed())
                    metrics = agent.update(buffer, self.rng)
                    self.updates += 1
                    record = {"environment_steps": self.steps, "update": self.updates,
                              "total_completed_episodes": self.completed_episodes,
                              "wall_seconds": time.perf_counter() - started,
                              "completed_episodes": len(completed), **metrics}
                    if completed:
                        record.update(mean_episode_return=float(np.mean([i["episode_metrics"]["episode_return"] for i in completed])),
                                      rollout_success_rate=sum(i["success"] for i in completed) / len(completed),
                                      rollout_stable_rate=sum(i["stable"] for i in completed) / len(completed))
                    update_log.write(json.dumps(record, allow_nan=False) + "\n")
                    update_log.flush()
                    budget = (f"steps={self.steps} episodes_total={self.completed_episodes}/{cfg.total_episodes}" if cfg.total_episodes
                              else f"steps={self.steps}/{cfg.total_steps}")
                    print(f"update={self.updates} {budget} "
                          f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f} "
                          f"episodes={len(completed)}", flush=True)
                    final = self._budget_reached()
                    if self.updates % cfg.eval_interval == 0 or final:
                        evaluation = evaluate_policy(agent, self.environment_config, cfg.eval_episodes, cfg.eval_seed)
                        write_json(self.run_dir / f"evaluation_{self.updates:06d}.json", evaluation)
                        score = (evaluation["summary"]["success_rate"], evaluation["summary"]["mean_return"])
                        if score > self.best_score:
                            self.best_score = score
                            self.checkpoint("best.pt")
                        print(f"  eval success={score[0]:.1%} mean_return={score[1]:.3f}", flush=True)
                    if self.updates % cfg.checkpoint_interval == 0 or final:
                        self.checkpoint("latest.pt")
        finally:
            env.close()
        return {"environment_steps": self.steps, "updates": self.updates,
                "completed_episodes": self.completed_episodes,
                "run_dir": str(self.run_dir), "best_eval_success_rate": self.best_score[0]}
