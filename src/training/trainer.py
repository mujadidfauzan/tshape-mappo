"""Train MAPPO on Easy/Medium; keep the best and latest checkpoints."""

from dataclasses import replace
import hashlib
from importlib.metadata import version
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import yaml

from src.algorithms.mappo import MAPPO
from src.algorithms.rollout_buffer import RolloutBuffer
from src.envs.config import load_config
from src.envs.model import build_model_xml
from src.envs.tshape_env import TShapeParallelEnv
from src.evaluation.evaluator import evaluate_policy
from .artifacts import append_jsonl, record_evaluation, write_json
from .config import load_mappo_config
from .logging import EpisodeStats, RunLogger


class Trainer:
    def __init__(self, run_dir, config=None, environment_config=None, resume=None):
        self.config = cfg = load_mappo_config(config)
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.steps = self.updates = self.completed_episodes = 0
        self.best_score = (-1., -float("inf"))
        self.stats = EpisodeStats(cfg.stats_window_size)
        self.agent = None
        saved = None
        self.run_dir = Path(run_dir).resolve()
        if self.run_dir.exists():
            raise FileExistsError(f"choose a new run directory: {self.run_dir}")
        if resume is not None:
            if environment_config is not None:
                raise ValueError("resume uses the task saved in the checkpoint")
            self.agent, payload = MAPPO.from_checkpoint(resume,device=cfg.device,load_optimizers=True)
            saved = payload["training"]
            if saved.get("task") not in ("fixed_easy_v1", "easy_v2", "medium_v1"):
                raise ValueError("resume requires an Easy or Medium checkpoint")
            previous = load_mappo_config(payload["mappo_config"])
            allowed = {"device","total_steps","eval_interval_steps","checkpoint_interval_steps",
                       "video_interval_steps","tensorboard","reward_log_interval_steps","eval_episodes"}
            if any(getattr(previous,k) != v for k,v in cfg.to_dict().items() if k not in allowed):
                raise ValueError("resume restores network and optimizer settings from the checkpoint")
            base = load_config(saved["environment_config"])
            if (saved["task"] == "medium_v1") != (base.task == "medium"):
                raise ValueError("checkpoint task does not match its environment configuration")
            xml = saved["model_xml"]
            self.steps, self.updates = saved["environment_steps"], saved["updates"]
            self.completed_episodes = saved["completed_episodes"]
            self.stats = EpisodeStats(cfg.stats_window_size,saved["episode_stats"])
            self.agent.config = cfg
        else:
            base = load_config(environment_config)
            xml = build_model_xml(base.resolved_model_path.read_text(), base.num_agents)
        if cfg.total_steps <= self.steps:
            raise ValueError("total_steps must exceed the checkpoint's cumulative timestep count")
        self.model_xml = xml
        self.model_hash = hashlib.sha256(xml.encode()).hexdigest()
        if saved and saved["model_sha256"] != self.model_hash:
            raise ValueError("checkpoint model hash mismatch")
        with tempfile.TemporaryDirectory(prefix="tshape-easy-start-") as directory:
            model = Path(directory)/"arena.xml"
            model.write_text(xml)
            temporary_config = replace(base,model_path=str(model))
            self.env = TShapeParallelEnv(temporary_config)
            try:
                if cfg.video_interval_steps and cfg.total_steps >= self._next(cfg.video_interval_steps):
                    from src.evaluation.video import check_video_support
                    check_video_support(temporary_config,cfg.video_width,cfg.video_height,cfg.video_crf)
            except BaseException:
                self.env.close()
                raise
        self.run_dir.mkdir(parents=True,exist_ok=False)
        snapshot = self.run_dir/"arena.xml"
        snapshot.write_text(xml)
        self.environment_config = replace(base,model_path=str(snapshot))
        self.env.config = self.environment_config
        self.env.physics.model_path = snapshot
        if self.agent is None:
            self.agent = MAPPO(**self.env.policy_dimensions,config=cfg)
        if self.agent.dimensions != self.env.policy_dimensions:
            self.env.close()
            raise ValueError("checkpoint dimensions do not match the Easy agent count")
        (self.run_dir/"environment.yaml").write_text(yaml.safe_dump(self.environment_config.to_dict()))
        (self.run_dir/"mappo.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
        self.logger = RunLogger(self.run_dir,cfg.tensorboard,cfg.reward_log_interval_steps)
        write_json(self.run_dir/"manifest.json",dict(
            task=self.environment_config.task_id,environment=self.environment_config.to_dict(),mappo=cfg.to_dict(),
            model_sha256=self.model_hash,actual_device=str(self.agent.device),
            resume_source=str(Path(resume).resolve()) if resume else None,
            previous_run_best_score=saved["best_score"] if saved else None,
            starting_steps=self.steps,reset_behavior="random_agents" if base.randomize_agents else "fixed",
            software={n:version(n) for n in ("torch","mujoco","numpy","pettingzoo")}))
        if saved:
            self.rng.bit_generator.state = saved["minibatch_rng_state"]
            torch.set_rng_state(saved["torch_rng_state"])
            if self.agent.device.type == "cuda" and saved.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(saved["cuda_rng_state"],self.agent.device)
        self._resume_environment_rng = saved.get("environment_rng_state") if saved else None
        if self._resume_environment_rng is not None:
            self.env.np_random.bit_generator.state = self._resume_environment_rng
        self.next_eval = self._next(cfg.eval_interval_steps)
        self.next_checkpoint = self._next(cfg.checkpoint_interval_steps)
        self.next_video = self._next(cfg.video_interval_steps) if cfg.video_interval_steps else None

    def _next(self, interval):
        return (self.steps//interval+1)*interval

    def checkpoint(self, name):
        self.agent.save_checkpoint(self.run_dir/"checkpoints"/name,dict(
            task=self.environment_config.task_id,environment_steps=self.steps,updates=self.updates,
            completed_episodes=self.completed_episodes,
            best_score=list(self.best_score) if self.best_score[0] >= 0 else None,
            environment_config=self.environment_config.to_dict(),model_xml=self.model_xml,
            model_sha256=self.model_hash,episode_stats=self.stats.state_dict(),
            minibatch_rng_state=self.rng.bit_generator.state,
            environment_rng_state=self.env.np_random.bit_generator.state,
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(self.agent.device)
                if self.agent.device.type == "cuda" else None))

    def _evaluate(self, video=False):
        episodes = self.config.eval_episodes if self.environment_config.randomize_agents and not video else 1
        result = evaluate_policy(self.agent,self.environment_config,episodes=episodes,seed=1000000,
                                 video_dir=self.run_dir/"videos"/f"step_{self.steps:09d}" if video else None)
        result.update(kind="video" if video else "evaluation",environment_steps=self.steps,update=self.updates)
        if video:
            record_evaluation(self.run_dir,result)
        else:
            score = (result["summary"]["success_rate"],result["summary"]["mean_return"])
            improved = score > self.best_score
            if improved:
                self.best_score = score
                self.checkpoint("best.pt")
            record_evaluation(self.run_dir,result,best=improved)
            self.logger.scalars("eval",result["summary"],self.steps)
            print(f"  eval success={score[0]:.0%} return={score[1]:.3f}",flush=True)

    def train(self):
        cfg = self.config
        names = self.env.possible_agents
        try:
            observations,_ = self.env.reset(seed=cfg.seed if self._resume_environment_rng is None else None)
            started, first_step = time.perf_counter(), self.steps
            self.checkpoint("latest.pt")
            while self.steps < cfg.total_steps:
                buffer = RolloutBuffer(min(cfg.rollout_steps,cfg.total_steps-self.steps),**self.agent.dimensions)
                components = {}
                for _ in range(buffer.capacity):
                    obs = np.stack([observations[n] for n in names])
                    state = self.env.state()
                    action,latent,log_probs = self.agent.act(obs)
                    value = float(self.agent.value(state))
                    observations,rewards,terminated,truncated,infos = self.env.step(dict(zip(names,action)))
                    term,trunc = terminated[names[0]],truncated[names[0]]
                    next_value = 0. if term else float(self.agent.value(self.env.state()))
                    buffer.add(obs,state,latent,log_probs,rewards[names[0]],value,next_value,term,trunc)
                    self.steps += 1
                    info = infos[names[0]]
                    self.logger.reward(rewards[names[0]],self.steps)
                    for key,value in info["reward_components"].items():
                        components[key] = components.get(key,0.)+value
                    if term or trunc:
                        self.completed_episodes += 1
                        self.stats.add(info)
                        append_jsonl(self.run_dir/"episodes.jsonl",dict(environment_steps=self.steps,
                                     episode=self.completed_episodes,**info))
                        self.logger.scalars("episode",{
                            "return":info["episode_metrics"]["episode_return"],
                            "length":info["episode_step"],"success":int(info["success"])},self.steps)
                        self.logger.scalars("contact",{f"agent_{i}_time_s":v for i,v in
                            enumerate(info["episode_metrics"]["agent_object_contact_time_s"])},self.steps)
                        observations,_ = self.env.reset()
                metrics = self.agent.update(buffer,self.rng)
                self.updates += 1
                elapsed = time.perf_counter()-started
                rolling = self.stats.summary()
                means = {k+"_mean":v/buffer.size for k,v in components.items()}
                reward_mean = float(buffer.rewards[:buffer.size].mean())
                fps = (self.steps-first_step)/max(elapsed,1e-9)
                append_jsonl(self.run_dir/"updates.jsonl",dict(
                    environment_steps=self.steps,update=self.updates,episodes=self.completed_episodes,
                    wall_seconds=elapsed,fps=fps,reward_mean_step=reward_mean,
                    reward_components_mean=means,rolling_episode_stats=rolling,**metrics))
                self.logger.scalars("train",metrics,self.steps)
                self.logger.scalars("rollout",rolling,self.steps)
                self.logger.scalars("reward",dict(mean_step=reward_mean,**means),self.steps)
                self.logger.scalars("time",dict(fps=fps,elapsed_s=elapsed),self.steps)
                progress = (f"return_mean={rolling['ep_rew_mean']:.3f} success={rolling['success_rate']:.1%}"
                            if rolling["window_episodes"] else "return_mean=n/a success=n/a")
                print(f"update={self.updates} steps={self.steps}/{cfg.total_steps} "
                      f"episodes={self.completed_episodes} {progress} fps={fps:.1f} "
                      f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f}",flush=True)
                final = self.steps >= cfg.total_steps
                if self.steps >= self.next_eval or final:
                    self._evaluate()
                    self.next_eval = self._next(cfg.eval_interval_steps)
                if self.next_video is not None and self.steps >= self.next_video:
                    self.checkpoint("latest.pt")
                    self._evaluate(video=True)
                    self.next_video = self._next(cfg.video_interval_steps)
                if self.steps >= self.next_checkpoint or final:
                    self.checkpoint("latest.pt")
                    self.next_checkpoint = self._next(cfg.checkpoint_interval_steps)
        finally:
            self.logger.close()
            self.env.close()
        return dict(environment_steps=self.steps,episodes=self.completed_episodes,
                    best_success_rate=self.best_score[0])
