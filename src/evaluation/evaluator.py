"""Evaluate without learning, changing normalization, or consuming training RNG."""

from contextlib import ExitStack
from pathlib import Path
import time

import numpy as np
import torch

from src.envs.tshape_env import TShapeParallelEnv


def evaluate_policy(agent, environment_config, episodes=5, seed=100000, deterministic=True,
                    render=False, trajectory_dir=None, video_dir=None,
                    video_width=1920, video_height=1080, video_crf=18):
    if episodes < 1 or seed < 0:
        raise ValueError("episodes must be positive and seed nonnegative")
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
                observations, infos = env.reset(seed=seed + episode)
                video = None
                if video_dir is not None:
                    from .video import EpisodeVideo
                    path = Path(video_dir) / f"policy_seed_{seed + episode}.mp4"
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
                if interrupted:
                    break
    finally:
        agent.train(was_training)
        env.close()
    successful_times = [r["completion_time_s"] for r in reports if r["success"] and not r["interrupted"]]
    returns = [r["episode_metrics"]["episode_return"] for r in reports]
    return {"reports": reports, "summary": {
        "episodes": len(reports), "success_rate": sum(r["success"] and not r["interrupted"] for r in reports) / len(reports),
        "stable_rate": sum(r["stable"] for r in reports) / len(reports),
        "mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
        "mean_completion_time_s": float(np.mean(successful_times)) if successful_times else None,
        "completion_time_episodes": len(successful_times),
    }}
