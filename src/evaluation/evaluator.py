"""Evaluate seeded Easy layouts without learning or changing the training RNG."""

from contextlib import ExitStack
from pathlib import Path
import time

import numpy as np
import torch

from src.envs.tshape_env import TShapeParallelEnv


def summarize_reports(reports):
    times = [r["completion_time_s"] for r in reports if r["success"] and not r["interrupted"]]
    returns = [r["episode_metrics"]["episode_return"] for r in reports]
    result = dict(episodes=len(reports),
                  success_rate=sum(r["success"] and not r["interrupted"] for r in reports)/len(reports),
                  stable_rate=sum(r["stable"] for r in reports)/len(reports),
                  mean_return=float(np.mean(returns)), std_return=float(np.std(returns)),
                  mean_completion_time_s=float(np.mean(times)) if times else None)
    for key in ("distance_to_goal", "orientation_error_rad"):
        values = [r[key] for r in reports if r[key] is not None]
        result[key+"_mean"] = float(np.mean(values)) if values else None
    return result


def evaluate_policy(agent, config, episodes=1, seed=42, deterministic=True, render=False,
                    video_dir=None, progress_callback=None):
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    env = TShapeParallelEnv(config, render_mode="human" if render else None)
    if agent.dimensions != env.policy_dimensions:
        env.close()
        raise ValueError("policy dimensions do not match the saved Easy agent count")
    devices = ([agent.device.index if agent.device.index is not None else torch.cuda.current_device()]
               if agent.device.type == "cuda" else [])
    was_training = agent.training
    reports = []
    try:
        agent.eval()
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(agent.device):
                    torch.cuda.manual_seed(seed)
            for episode in range(episodes):
                def notify(event, **details):
                    if progress_callback is not None:
                        progress_callback(dict(event=event,episode=episode+1,episodes=episodes,**details))
                notify("episode_start")
                observations, infos = env.reset(seed=seed+episode)
                final = infos["agent_0"]
                interrupted = False
                with ExitStack() as recording:
                    video = None
                    if video_dir is not None and episode == 0:
                        from .video import EpisodeVideo
                        video = recording.enter_context(EpisodeVideo(
                            env.physics,Path(video_dir)/"policy.mp4",agent.config.video_width,
                            agent.config.video_height,agent.config.video_crf))
                        video.capture()
                    if render:
                        env.render()
                    while env.agents:
                        start = time.perf_counter()
                        if render and not env.viewer.is_running():
                            interrupted = True
                            break
                        actions,_,_ = agent.act(np.stack([observations[n] for n in env.possible_agents]),
                                               deterministic=deterministic)
                        observations,_,_,_,infos = env.step(dict(zip(env.possible_agents,actions)))
                        final = infos["agent_0"]
                        if video is not None:
                            video.capture()
                        if env.steps % 100 == 0:
                            notify("episode_progress",step=env.steps,distance=final["distance_to_goal"])
                        if render:
                            env.render()
                            time.sleep(max(0.,env.physics.control_dt-(time.perf_counter()-start)))
                    metadata = video.metadata() if video is not None else None
                report = dict(seed=seed+episode,deterministic=deterministic,interrupted=interrupted,
                              video=metadata,**final)
                reports.append(report)
                notify("episode_end",report=report)
                if interrupted:
                    break
    finally:
        agent.train(was_training)
        env.close()
    return dict(reports=reports,summary=summarize_reports(reports))
