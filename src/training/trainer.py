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
from src.envs.config import load_config, load_checkpoint_config
from src.envs.observations import feature_layout
from src.envs.model import model_with_agents
from src.envs.tshape_env import TShapeParallelEnv
from src.evaluation.evaluator import evaluate_policy
from .config import load_mappo_config
from .logging import EpisodeStats, RunLogger
from .artifacts import write_json, record_evaluation
from .curriculum import Curriculum


def feature_schema(num_agents=4):
    actor, critic = feature_layout(num_agents)
    return {"actor": [[name, size] for name, size in actor],
            "critic": [[name, size] for name, size in critic]}


class Trainer:
    def __init__(self, run_dir, config=None, environment_config=None, resume=None,
                 resume_step_budget=None, resume_video_interval=None, resume_eval_interval=None,
                 resume_checkpoint_interval=None, resume_tensorboard=None, resume_reward_interval=None,
                 init_actor=None):
        cfg = self.config = load_mappo_config(config)
        if cfg.total_episodes is not None:
            raise ValueError("Episode budgets are no longer used; set total_steps/--total-steps instead")
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.steps = self.updates = self.episode_index = 0
        self.completed_episodes = self.last_video_step = 0
        self.episode_stats = EpisodeStats(cfg.stats_window_size)
        self.best_score = (-1.0, -float("inf"))
        self.agent = None
        self.resume_source = str(Path(resume).resolve()) if resume is not None else None
        self.resume_best_score = None
        if resume is not None and init_actor is not None:
            raise ValueError("actor initialization starts a new run; do not combine it with resume")
        self.actor_initialization_source = str(Path(init_actor).resolve()) if init_actor is not None else None
        self.curriculum = None
        saved_environment = None
        if resume is None:
            base_env = load_config(environment_config)
            model_xml = base_env.resolved_model_path.read_text()
        else:
            if environment_config is not None:
                raise ValueError("resume uses the checkpoint environment; do not supply --env-config")
            self.agent, payload = MAPPO.from_checkpoint(resume, device=cfg.device, load_optimizers=True)
            training = payload["training"]
            self.actor_initialization_source = training.get("actor_initialization_source")
            if training.get("feature_schema") != feature_schema(self.agent.dimensions["num_agents"]):
                raise ValueError("checkpoint observation schema does not match this implementation")
            # Budget, device and recording cadence may change on resume.
            previous_cfg = load_mappo_config(payload["mappo_config"])
            if previous_cfg.total_episodes is not None and resume_step_budget is None:
                raise ValueError("Legacy episode-budget checkpoint: supply --total-steps explicitly to resume")
            cfg = self.config = replace(previous_cfg,
                                       total_steps=(cfg.total_steps if resume_step_budget is None else resume_step_budget),
                                       device=cfg.device,
                                       total_episodes=None,
                                       video_interval_steps=(previous_cfg.video_interval_steps
                                           if resume_video_interval is None else resume_video_interval),
                                       eval_interval_steps=(previous_cfg.eval_interval_steps
                                           if resume_eval_interval is None else resume_eval_interval),
                                       checkpoint_interval_steps=(previous_cfg.checkpoint_interval_steps
                                           if resume_checkpoint_interval is None else resume_checkpoint_interval),
                                       tensorboard=(previous_cfg.tensorboard
                                           if resume_tensorboard is None else resume_tensorboard),
                                       reward_log_interval_steps=(previous_cfg.reward_log_interval_steps
                                           if resume_reward_interval is None else resume_reward_interval))
            self.agent.config = cfg
            torch.set_num_threads(cfg.torch_threads)
            self.steps = training["environment_steps"]
            self.updates = training["updates"]
            self.completed_episodes = training.get("completed_episodes", training["episode_index"])
            self.last_video_step = training.get("last_video_step", 0)
            self.episode_stats = EpisodeStats(cfg.stats_window_size, training.get("episode_stats", ()))
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
            saved_environment = training["environment_config"]
            base_env = load_checkpoint_config(saved_environment)
            model_xml = training["model_xml"]
            if hashlib.sha256(model_xml.encode()).hexdigest() != training["model_sha256"]:
                raise ValueError("checkpoint XML hash mismatch")
            if self._budget_reached():
                raise ValueError("training budget must exceed the steps/episodes already in the checkpoint")
            if training.get("curriculum_definition") is not None:
                self.curriculum = Curriculum.load(training["curriculum_definition"], training.get("curriculum_state"))
        if resume is None and cfg.curriculum is not None:
            self.curriculum = Curriculum.load(cfg.curriculum)
        if resume is not None and cfg.curriculum is not None and self.curriculum is None:
            raise ValueError("curriculum checkpoint is missing its saved definition")
        if self.curriculum:
            start = self.curriculum.seed
            if cfg.eval_seed < start + self.curriculum.episodes and cfg.eval_seed + cfg.eval_episodes > start:
                raise ValueError("curriculum validation seeds must be separate from full-task validation")
        # Read saved markers before migration so old statistics are not reused.
        previous = saved_environment if saved_environment is not None else base_env.to_dict()
        previous_reset_distribution = previous.get("reset_distribution", "legacy")
        previous_randomize_goal = previous.get("randomize_goal", True)
        reset_distribution_changed = previous_reset_distribution != "near_object" or not base_env.randomize_reset
        goal_distribution_changed = base_env.difficulty != "easy" and previous_randomize_goal
        base_env = replace(base_env, reset_distribution="near_object", randomize_reset=True,
                           randomize_goal=previous_randomize_goal if base_env.difficulty == "easy" else False)
        if base_env.reward_mode == "footprint_potential" and base_env.shaping_gamma != cfg.gamma:
            raise ValueError("footprint potential shaping_gamma must equal MAPPO gamma")
        if resume is None:
            model_xml = model_with_agents(model_xml, base_env.num_agents)
        if self.agent is not None and (reset_distribution_changed or goal_distribution_changed):
            self.agent.obs_normalizer.reset()
            self.agent.state_normalizer.reset()
            self.episode_stats = EpisodeStats(cfg.stats_window_size)
        # Require a new directory; never overwrite another experiment's results.
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.model_xml = model_xml
        self.model_hash = hashlib.sha256(model_xml.encode()).hexdigest()
        snapshot = self.run_dir / "arena.xml"
        snapshot.write_text(model_xml)
        self.environment_config = replace(base_env, model_path=str(snapshot))
        if self.curriculum:
            self.curriculum.validate_configs(self.environment_config)
            for index in range(len(self.curriculum.definition["stages"])):
                probe = TShapeParallelEnv(config=self.curriculum.config(self.environment_config, index))
                try:
                    probe.reset(seed=self.curriculum.seed + index)
                finally:
                    probe.close()
            (self.run_dir / "curriculum.yaml").write_text(yaml.safe_dump(self.curriculum.definition))
        active_config = self.curriculum.config(self.environment_config) if self.curriculum else self.environment_config
        self.env = TShapeParallelEnv(config=active_config)
        if base_env.difficulty == "hard":
            centers = []
            for prefix in ("obstacle_0", "obstacle_1"):
                upper = self.env.physics.model.geom(f"{prefix}_upper")
                lower = self.env.physics.model.geom(f"{prefix}_lower")
                centers.append((upper.pos[1] - upper.size[1] + lower.pos[1] + lower.size[1]) / 2)
            if not np.allclose(centers, 0):
                self.env.close()
                raise ValueError("Hard training requires aligned openings at y=0. A resumed checkpoint keeps its saved XML; "
                                 "start a new run with configs/environment_hard.yaml for the corrected course")
        dimensions = {"observation_size": self.env.obs_dim, "state_size": self.env.state_space.shape[0],
                      "action_size": 2, "num_agents": len(self.env.possible_agents)}
        if self.agent is None:
            self.agent = MAPPO(**dimensions, config=cfg)
        elif self.agent.dimensions != dimensions:
            raise ValueError("checkpoint dimensions do not match the environment")
        if init_actor is not None:
            source = torch.load(init_actor, map_location="cpu", weights_only=True)
            if source["dimensions"] != dimensions or source["training"].get("feature_schema") != feature_schema(base_env.num_agents):
                raise ValueError("actor transfer requires the same agent count and feature schema")
            previous_cfg = load_mappo_config(source["mappo_config"])
            if (previous_cfg.normalize_inputs != cfg.normalize_inputs or previous_cfg.normalization_clip != cfg.normalization_clip):
                raise ValueError("actor transfer requires matching input-normalization settings")
            self.agent.actor.load_state_dict({k.removeprefix("actor."): v for k,v in source["model"].items() if k.startswith("actor.")})
            self.agent.obs_normalizer.load_state_dict({k.removeprefix("obs_normalizer."): v for k,v in source["model"].items() if k.startswith("obs_normalizer.")})
        self.next_eval_step = self._next_threshold(cfg.eval_interval_steps)
        self.next_checkpoint_step = self._next_threshold(cfg.checkpoint_interval_steps)
        self.next_video_step = self._next_threshold(cfg.video_interval_steps)
        if self.next_video_step is not None and cfg.total_steps >= self.next_video_step:
            from src.evaluation.video import check_video_support
            try:
                check_video_support(self.environment_config, cfg.video_width, cfg.video_height, cfg.video_crf)
            except Exception:
                self.env.close()
                raise
        try:
            self.logger = RunLogger(self.run_dir, cfg.tensorboard, cfg.reward_log_interval_steps)
        except Exception:
            self.env.close()
            raise
        self.software = {name: str(version(name)) for name in
                         ("torch", "mujoco", "numpy", "gymnasium", "pettingzoo", "PyYAML")}
        if cfg.tensorboard:
            self.software["tensorboard"] = str(version("tensorboard"))
        manifest = {"mappo": cfg.to_dict(), "environment": self.environment_config.to_dict(),
                    "software": self.software, "model_sha256": self.model_hash,
                    "feature_schema": feature_schema(base_env.num_agents), "resume_source": self.resume_source,
                    "actor_initialization_source": self.actor_initialization_source,
                    "curriculum_definition": self.curriculum.definition if self.curriculum else None,
                    "curriculum_state": self.curriculum.state_dict() if self.curriculum else None,
                    "previous_run_best_score": self.resume_best_score,
                    "actual_device": str(self.agent.device),
                    "previous_reset_distribution": previous_reset_distribution,
                    "reset_distribution_changed": reset_distribution_changed,
                    "previous_randomize_goal": previous_randomize_goal,
                    "goal_distribution_changed": goal_distribution_changed,
                    "normalization_reset": resume is not None and (reset_distribution_changed or goal_distribution_changed),
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
                    "feature_schema": feature_schema(self.environment_config.num_agents), "software": self.software,
                    "curriculum_definition": self.curriculum.definition if self.curriculum else None,
                    "curriculum_state": self.curriculum.state_dict() if self.curriculum else None,
                    "active_environment_config": self.env.config.to_dict(),
                    "actor_initialization_source": self.actor_initialization_source,
                    "environment_steps": self.steps, "updates": self.updates,
                    "episode_index": self.episode_index,
                    "completed_episodes": self.completed_episodes,
                    "last_video_step": self.last_video_step,
                    "episode_stats": self.episode_stats.state_dict(),
                    "best_score": list(self.best_score) if self.best_score[0] >= 0 else None,
                    "minibatch_rng_state": self.rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (torch.cuda.get_rng_state(self.agent.device)
                                       if self.agent.device.type == "cuda" else None)}
        self.agent.save_checkpoint(self.run_dir / "checkpoints" / name, training)

    def _budget_reached(self):
        return self.steps >= self.config.total_steps

    def _next_threshold(self, interval):
        return (self.steps // interval + 1) * interval if interval else None

    def _record_video(self, scheduled_step):
        directory = self.run_dir / "videos" / f"step_{self.steps:09d}"
        self.last_video_step = self.steps
        self.checkpoint("latest.pt")
        result = evaluate_policy(
            self.agent, self.environment_config, episodes=1, seed=self.config.eval_seed,
            video_dir=directory, video_width=self.config.video_width,
            video_height=self.config.video_height, video_crf=self.config.video_crf)
        result.update(training_episodes=self.completed_episodes, environment_steps=self.steps,
                      scheduled_step=scheduled_step,
                      updates=self.updates, model_sha256=self.model_hash,
                      checkpoint=str(self.run_dir / "checkpoints" / "latest.pt"),
                      checkpoint_is_overwritten=True)
        record_evaluation(self.run_dir, result, kind="video")
        print(f"  video steps={self.steps}: {result['reports'][0]['video']['path']}", flush=True)

    def episode_seed(self):
        """Skip the fixed validation seed range even during a very long run."""
        cfg = self.config
        candidate = cfg.seed + self.episode_index
        ranges = [(cfg.eval_seed, cfg.eval_seed + cfg.eval_episodes - 1), (200000, 299999)]
        curriculum = getattr(self, "curriculum", None)
        if curriculum:
            ranges.append((curriculum.seed, curriculum.seed + curriculum.episodes - 1))
        # Merge overlapping reserved ranges, then map the episode index into
        # the remaining seed sequence without collisions at either boundary.
        merged = []
        for first, last in sorted(ranges):
            if merged and first <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(last, merged[-1][1]))
            else:
                merged.append((first, last))
        for first, last in merged:
            if last < cfg.seed:
                continue
            if max(first, cfg.seed) <= candidate:
                candidate += last - max(first, cfg.seed) + 1
        return candidate

    def train(self):
        cfg, env, agent = self.config, self.env, self.agent
        names = env.possible_agents
        observations, _ = env.reset(seed=self.episode_seed())
        started = time.perf_counter()
        initial_steps = self.steps
        try:
            with (self.run_dir / "episodes.jsonl").open("a") as episode_log, (self.run_dir / "updates.jsonl").open("a") as update_log:
                while not self._budget_reached():
                    capacity = min(cfg.rollout_steps, cfg.total_steps - self.steps)
                    buffer = RolloutBuffer(capacity, **agent.dimensions)
                    completed = []
                    component_totals = {key: 0.0 for key in ("progress", "time", "success", "failure")}
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
                        self.logger.reward(rewards[names[0]], self.steps)
                        for key, value in infos[names[0]]["reward_components"].items():
                            component_totals[key] += value
                        observations = next_obs
                        if term or trunc:
                            info = infos[names[0]]
                            record = {"environment_steps": self.steps, "episode_index": self.episode_index,
                                      "completed_episodes": self.completed_episodes + 1,
                                      "seed": self.episode_seed(), **info}
                            record["curriculum_stage"] = self.curriculum.name if self.curriculum else None
                            episode_log.write(json.dumps(record, allow_nan=False) + "\n")
                            episode_log.flush()
                            completed.append(info)
                            self.episode_stats.add(info)
                            self.logger.scalars("episode", {"return": info["episode_metrics"]["episode_return"],
                                "length": info["episode_step"], "success": int(info["success"]),
                                "stable": int(info["stable"])}, self.steps)
                            self.logger.scalars("contact", {f"agent_{i}_time_s": seconds for i, seconds in
                                enumerate(info["diagnostics"]["agent_object_contact_time_s"])}, self.steps)
                            self.completed_episodes += 1
                            self.episode_index += 1
                            if self._budget_reached():
                                break
                            observations, _ = env.reset(seed=self.episode_seed())
                    metrics = agent.update(buffer, self.rng)
                    self.updates += 1
                    elapsed = time.perf_counter() - started
                    speed = (self.steps - initial_steps) / max(elapsed, 1e-9)
                    rolling = self.episode_stats.summary()
                    reward_means = {key + "_mean": value / buffer.size for key, value in component_totals.items()}
                    reward_mean = float(np.mean(buffer.rewards[:buffer.size]))
                    record = {"environment_steps": self.steps, "update": self.updates,
                              "curriculum_stage": self.curriculum.name if self.curriculum else None,
                              "total_completed_episodes": self.completed_episodes,
                              "wall_seconds": elapsed, "fps": speed,
                              "reward_mean_step": reward_mean, "reward_components_mean": reward_means,
                              "rolling_episode_stats": rolling,
                              "completed_episodes": len(completed), **metrics}
                    if completed:
                        record.update(mean_episode_return=float(np.mean([i["episode_metrics"]["episode_return"] for i in completed])),
                                      rollout_success_rate=sum(i["success"] for i in completed) / len(completed),
                                      rollout_stable_rate=sum(i["stable"] for i in completed) / len(completed))
                    update_log.write(json.dumps(record, allow_nan=False) + "\n")
                    update_log.flush()
                    self.logger.scalars("train", metrics, self.steps)
                    self.logger.scalars("rollout", rolling, self.steps)
                    self.logger.scalars("reward", {"mean_step": reward_mean, **reward_means}, self.steps)
                    self.logger.scalars("time", {"fps": speed, "elapsed_s": elapsed,
                        "total_timesteps": self.steps, "episodes": self.completed_episodes}, self.steps)
                    performance = (f"return_mean={rolling['ep_rew_mean']:.3f} "
                        f"success={rolling['success_rate']:.1%} stable={rolling['stable_rate']:.1%} "
                        f"ep_len={rolling['ep_len_mean']:.1f} window={rolling['window_episodes']}"
                        if rolling["window_episodes"] else "return_mean=n/a success=n/a stable=n/a window=0")
                    print(f"update={self.updates} steps={self.steps}/{cfg.total_steps} "
                          f"episodes_total={self.completed_episodes} {performance} "
                          f"reward_step_mean={reward_mean:.4f} fps={speed:.1f} "
                          f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f}", flush=True)
                    final = self._budget_reached()
                    if self.steps >= self.next_eval_step or final:
                        evaluation = evaluate_policy(agent, self.environment_config, cfg.eval_episodes, cfg.eval_seed)
                        evaluation.update(environment_steps=self.steps, update=self.updates,
                                          scheduled_step=(self.next_eval_step if self.steps >= self.next_eval_step else None),
                                          final=final)
                        self.logger.scalars("eval", evaluation["summary"], self.steps)
                        score = (evaluation["summary"]["success_rate"], evaluation["summary"]["mean_return"])
                        improved = score > self.best_score
                        if improved:
                            self.best_score = score
                            self.checkpoint("best.pt")
                        record_evaluation(self.run_dir, evaluation, best=improved)
                        print(f"  eval steps={self.steps} success={score[0]:.1%} mean_return={score[1]:.3f}", flush=True)
                        self.next_eval_step = self._next_threshold(cfg.eval_interval_steps)
                    if self.next_video_step is not None and self.steps >= self.next_video_step:
                        self._record_video(self.next_video_step)
                        self.next_video_step = self._next_threshold(cfg.video_interval_steps)
                    if self.curriculum and self.curriculum.due(self.steps):
                        stage_name = self.curriculum.name
                        stage_result = evaluate_policy(agent, env.config, self.curriculum.episodes, self.curriculum.seed)
                        stage_result.update(environment_steps=self.steps, update=self.updates, stage=stage_name)
                        promoted = self.curriculum.observe(stage_result["summary"]["success_rate"], self.steps)
                        stage_result.update(promoted=promoted, curriculum_state=self.curriculum.state_dict())
                        record_evaluation(self.run_dir, stage_result, kind="curriculum")
                        self.logger.scalars("curriculum", {"stage_index": self.curriculum.index,
                            "stage_success_rate": stage_result["summary"]["success_rate"],
                            "completed": int(self.curriculum.completed)}, self.steps)
                        print(f"  curriculum stage={stage_name} success={stage_result['summary']['success_rate']:.1%} "
                              f"next={self.curriculum.name} completed={self.curriculum.completed}", flush=True)
                        if promoted:
                            transition = {"environment_steps": self.steps, "from_stage": stage_name,
                                "to_stage": self.curriculum.name, "discarded_partial_episode_steps": env.steps,
                                "discarded_seed": self.episode_seed()}
                            with (self.run_dir / "curriculum_transitions.jsonl").open("a") as stream:
                                stream.write(json.dumps(transition) + "\n")
                            if env.steps:
                                self.episode_index += 1
                            env.close()
                            env = self.env = TShapeParallelEnv(config=self.curriculum.config(self.environment_config))
                            self.episode_stats = EpisodeStats(cfg.stats_window_size)
                            if not final:
                                observations, _ = env.reset(seed=self.episode_seed())
                        # Promotion state must survive even when the normal
                        # checkpoint threshold has not been crossed yet.
                        self.checkpoint("latest.pt")
                    if self.steps >= self.next_checkpoint_step or final:
                        self.checkpoint("latest.pt")
                        self.next_checkpoint_step = self._next_threshold(cfg.checkpoint_interval_steps)
                    self.logger.flush()
        finally:
            self.logger.close()
            env.close()
        return {"environment_steps": self.steps, "updates": self.updates,
                "completed_episodes": self.completed_episodes,
                "run_dir": str(self.run_dir), "best_eval_success_rate": self.best_score[0]}
