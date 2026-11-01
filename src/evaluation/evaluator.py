"""Evaluate without learning, changing normalization, or consuming training RNG."""

from contextlib import ExitStack
from pathlib import Path
import time
import math
from collections import Counter

import numpy as np
import torch

from src.envs.tshape_env import TShapeParallelEnv


def summarize_reports(reports):
    successful_times = [r["completion_time_s"] for r in reports if r["success"] and not r["interrupted"]]
    returns = [r["episode_metrics"]["episode_return"] for r in reports]
    result = {"episodes": len(reports),
            "success_rate": sum(r["success"] and not r["interrupted"] for r in reports) / len(reports) if reports else None,
            "stable_rate": sum(r["stable"] for r in reports) / len(reports) if reports else None,
            "mean_return": float(np.mean(returns)) if returns else None,
            "std_return": float(np.std(returns)) if returns else None,
            "mean_completion_time_s": float(np.mean(successful_times)) if successful_times else None,
            "completion_time_episodes": len(successful_times)}
    valid = [r for r in reports if not r["interrupted"]]
    if valid:
        angles = [r["orientation_error_rad"] for r in valid if r.get("orientation_error_rad") is not None]
        if angles:
            result["orientation_error_rad_mean"] = float(np.mean(angles))
        result["failure_categories"] = dict(Counter(r.get("diagnostics", {}).get("failure_category", r.get("end_reason")) for r in valid))
        for name in ("footprint_error_m", "gate_clear_fraction"):
            values = [r["diagnostics"][name] for r in valid if name in r.get("diagnostics", {})]
            if values:
                result[name + "_mean"] = float(np.mean(values))
        for name in ("object_path_length_m", "control_effort_N2_s", "wall_collision_events", "max_penetration_m"):
            values = [r["episode_metrics"][name] for r in valid if name in r.get("episode_metrics", {})]
            if values:
                result[name + "_mean"] = float(np.mean(values))
        per_agent = [r["episode_metrics"]["control_effort_N2_s"] / r["num_agents"] for r in valid
                     if r.get("num_agents") and "control_effort_N2_s" in r.get("episode_metrics", {})]
        if per_agent:
            result["control_effort_per_agent_N2_s_mean"] = float(np.mean(per_agent))
    return result


def wilson_interval(successes, episodes):
    """95% binomial interval; meaningful on a predeclared held-out sample."""
    if episodes < 1 or not 0 <= successes <= episodes:
        raise ValueError("invalid binomial sample")
    z, p = 1.959963984540054, successes / episodes
    center = (p + z*z / (2*episodes)) / (1 + z*z/episodes)
    half = z * math.sqrt(p*(1-p)/episodes + z*z/(4*episodes*episodes)) / (1 + z*z/episodes)
    return max(0., center-half), min(1., center+half)


def evaluate_policy(agent, environment_config, episodes=5, seed=100000, deterministic=True,
                    render=False, trajectory_dir=None, video_dir=None,
                    video_width=1920, video_height=1080, video_crf=18,
                    progress_callback=None, progress_interval_steps=100, video_episodes=None):
    if episodes < 1 or seed < 0:
        raise ValueError("episodes must be positive and seed nonnegative")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable")
    if isinstance(progress_interval_steps, bool) or not isinstance(progress_interval_steps, int) or progress_interval_steps < 1:
        raise ValueError("progress_interval_steps must be a positive integer")
    if video_episodes is not None and (isinstance(video_episodes, bool) or not isinstance(video_episodes, int) or video_episodes < 0):
        raise ValueError("video_episodes must be a nonnegative integer or None")
    env = TShapeParallelEnv(config=environment_config, render_mode="human" if render else None)
    expected = agent.dimensions
    if (env.obs_dim != expected["observation_size"] or env.state_space.shape[0] != expected["state_size"]
            or len(env.possible_agents) != expected["num_agents"]):
        env.close()
        raise ValueError("checkpoint dimensions do not match the evaluation environment")
    names = env.possible_agents
    devices = [agent.device.index if agent.device.index is not None else torch.cuda.current_device()] if agent.device.type == "cuda" else []
    was_training = agent.training
    agent.eval()
    reports = []
    try:
        with torch.random.fork_rng(devices=devices), ExitStack() as recordings:
            # CPU/CUDA stream is local to this evaluation and restored afterwards.
            torch.random.default_generator.manual_seed(seed)
            if agent.device.type == "cuda":
                with torch.cuda.device(agent.device):
                    torch.cuda.manual_seed(seed)
            for episode in range(episodes):
                episode_started = last_progress = time.perf_counter()
                def notify(event, **details):
                    if progress_callback is not None:
                        progress_callback({"event": event, "episode": episode + 1, "episodes": episodes,
                            "seed": seed + episode, "wall_seconds": time.perf_counter() - episode_started,
                            **details})
                notify("episode_start")
                observations, infos = env.reset(seed=seed + episode)
                video = None
                if video_dir is not None and (video_episodes is None or episode < video_episodes):
                    from .video import EpisodeVideo
                    path = Path(video_dir) / f"policy_seed_{seed + episode}.mp4"
                    notify("video_start", path=str(path), width=video_width, height=video_height)
                    video = recordings.enter_context(EpisodeVideo(
                        env.physics, path, video_width, video_height, video_crf))
                    video.capture()
                states, poses, action_log, rewards_log, times = [], [], [], [], []
                terminated_log, truncated_log = [], []
                final_info = infos[names[0]]
                interrupted = False
                while env.agents:
                    started = time.perf_counter()
                    if trajectory_dir is not None:
                        states.append(env.state())
                        poses.append(env.physics.data.qpos.copy())
                        times.append(float(env.physics.data.time))
                    obs = np.stack([observations[name] for name in names])
                    action_array, _, _ = agent.act(obs, deterministic=deterministic)
                    observations, rewards, terminated, truncated, infos = env.step(dict(zip(names, action_array)))
                    final_info = infos[names[0]]
                    if video is not None:
                        video.capture()
                    now = time.perf_counter()
                    if (final_info["episode_step"] % progress_interval_steps == 0 or now - last_progress >= 5.0):
                        notify("episode_progress", step=final_info["episode_step"], max_steps=env.max_steps,
                               episode_return=final_info["episode_metrics"]["episode_return"],
                               distance_to_goal=final_info["distance_to_goal"])
                        last_progress = now
                    if trajectory_dir is not None:
                        action_log.append(action_array)
                        rewards_log.append(rewards[names[0]])
                        terminated_log.append(terminated[names[0]])
                        truncated_log.append(truncated[names[0]])
                    if render:
                        env.render()
                        if not env.viewer.is_running():
                            interrupted = True
                            break
                        remaining = env.physics.control_dt - (time.perf_counter() - started)
                        if remaining > 0:
                            time.sleep(remaining)
                if video is not None:
                    notify("video_finalizing", path=str(video.path))
                    video.close()
                trajectory = None
                if trajectory_dir is not None:
                    states.append(env.state())
                    poses.append(env.physics.data.qpos.copy())
                    times.append(float(env.physics.data.time))
                    path = Path(trajectory_dir) / f"policy_seed_{seed + episode}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, states=np.asarray(states), qpos=np.asarray(poses),
                                        actions=np.asarray(action_log), rewards=np.asarray(rewards_log),
                                        time_s=np.asarray(times), terminated=np.asarray(terminated_log),
                                        truncated=np.asarray(truncated_log),
                                        seed=np.asarray(seed + episode))
                    trajectory = str(path)
                reports.append({"seed": seed + episode, "deterministic": deterministic,
                                "interrupted": interrupted, "trajectory": trajectory,
                                "video": video.metadata() if video is not None else None, **final_info})
                notify("episode_end", report=reports[-1])
                if interrupted:
                    break
    finally:
        agent.train(was_training)
        env.close()
    return {"reports": reports, "summary": summarize_reports(reports)}
