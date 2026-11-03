"""Easy Parallel environment with a fixed object/goal and configurable agents."""

from contextlib import nullcontext
from copy import deepcopy

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .config import load_config
from .observations import feature_sizes, build_observations, build_state
from .physics import TShapePhysics, finite_array
from .spawning import sample_agent_positions
from .rewards import team_reward, wrapped_yaw_error
from .termination import goal_is_settled, object_inside_goal


class TShapeParallelEnv(ParallelEnv):
    metadata = {"name": "tshape_easy_v2", "render_modes": ["human"],
                "render_fps": 20, "is_parallelizable": True}

    def __init__(self, config=None, render_mode=None):
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or human")
        self.config = cfg = load_config(config)
        self.physics = TShapePhysics(cfg.frame_skip, cfg.resolved_model_path, cfg.num_agents)
        self.np_random = np.random.default_rng(0)
        self.goal_center = np.asarray(cfg.goal_center)
        self.goal_half_size = np.asarray(cfg.goal_half_size)
        if (np.any(self.goal_center - self.goal_half_size < self.physics.lower)
                or np.any(self.goal_center + self.goal_half_size > self.physics.upper)):
            raise ValueError("the complete goal rectangle must fit inside the arena")
        self.possible_agents = self.physics.agent_names.copy()
        self.agents = []
        self.max_steps = cfg.max_steps
        self.render_mode = render_mode
        self.viewer = None
        self.metadata = {**type(self).metadata, "render_fps": round(1/self.physics.control_dt)}
        observation_size, state_size = feature_sizes(cfg.num_agents)
        self.policy_dimensions = dict(observation_size=observation_size, state_size=state_size,
                                      action_size=2, num_agents=cfg.num_agents)
        self.state_space = spaces.Box(-np.inf, np.inf, (state_size,), np.float32)
        self._observation_spaces = {n: spaces.Box(-np.inf, np.inf, (observation_size,), np.float32)
                                    for n in self.possible_agents}
        self._action_spaces = {n: spaces.Box(-1., 1., (2,), np.float32) for n in self.possible_agents}
        self._has_reset = False
        self._failed = False
        self._last_observations = self._last_state = None
        marker = self.physics.model.site("goal")
        marker.pos[:] = [*self.goal_center, .004]
        marker.size[:] = [*self.goal_half_size, .003]
        self._restore_layout()

    def _restore_layout(self):
        cfg, physics = self.config, self.physics
        physics.reset(cfg.object_pose)
        if cfg.randomize_agents:
            positions = sample_agent_positions(physics, self.np_random, cfg.spawn_gap_min, cfg.spawn_gap_max)
        elif cfg.agent_positions is not None:
            positions = cfg.agent_positions
        elif cfg.num_agents == 4:
            positions = physics.model.body_pos[physics.agent_bodies, :2]
        else:
            positions = sample_agent_positions(physics, np.random.default_rng(0), cfg.spawn_gap_min, cfg.spawn_gap_max)
        physics.reset(cfg.object_pose, positions)
        if self.physics.boundary_penetration() > 1e-9:
            raise ValueError("initial bodies must fit inside the arena")
        if any(c.dist < -1e-9 for c in self.physics.data.contact[:self.physics.data.ncon]):
            raise ValueError("initial bodies must not overlap")
        if self.object_inside_goal():
            raise ValueError("the fixed initial object must start outside the target")

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def _finite(self):
        return np.isfinite(self.physics.data.qpos).all() and np.isfinite(self.physics.data.qvel).all()

    def _fractions(self):
        return max(0., 1-self.steps/self.max_steps), min(1., self.goal_hold_count/self.config.success_hold_steps)

    def _observations(self):
        if self._failed:
            return {n: a.copy() for n, a in self._last_observations.items()}
        return build_observations(self.physics, self.goal_center, self.goal_half_size, *self._fractions())

    def state(self):
        if self._failed:
            return self._last_state.copy()
        return build_state(self.physics, self.goal_center, self.goal_half_size, *self._fractions())

    def _distance(self):
        return float(np.linalg.norm(self.physics.get_object_state()["position"] - self.goal_center))

    def object_inside_goal(self):
        return object_inside_goal(self.physics.object_corners(), self.goal_center, self.goal_half_size)

    def reset(self, seed=None, options=None):
        # Options never override the saved task or spawn distribution.
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a nonnegative integer")
            self.np_random = np.random.default_rng(seed)
            for i, name in enumerate(self.possible_agents):
                self.action_space(name).seed(seed+i)
        with self.viewer.lock() if self.viewer is not None else nullcontext():
            self._restore_layout()
            self.agents = self.possible_agents.copy()
            self.steps = self.goal_hold_count = 0
            self._failed = False
            self._failure_details = None
            self._metrics = dict(episode_return=0., object_path_length_m=0., control_effort_N2_s=0.,
                                 wall_collision_events=0, max_penetration_m=0., inside_goal_steps=0,
                                 agent_object_contact_time_s=[0.]*self.config.num_agents)
            obj = self.physics.get_object_state()
            self._initial_state = dict(object_position=obj["position"].tolist(), object_yaw=obj["yaw"],
                                       agent_positions=self.physics.get_agents_state()[0].tolist(),
                                       goal_center=self.goal_center.tolist(), goal_yaw=self.config.goal_yaw)
            obs = self._observations()
            self._last_observations = {n: a.copy() for n, a in obs.items()}
            self._last_state = self.state()
            self._has_reset = True
            info = self._info(False, {}, None, None)
        return obs, {n: deepcopy(info) for n in self.agents}

    def _info(self, success, components, reason, failure_reason):
        finite = self._finite()
        obj = self.physics.get_object_state() if finite else None
        return dict(
            success=bool(success), stable=not self._failed, episode_step=self.steps,
            elapsed_time_s=float(self.physics.data.time) if np.isfinite(self.physics.data.time) else None,
            completion_time_s=float(self.physics.data.time) if success else None,
            distance_to_goal=self._distance() if finite else None,
            orientation_error_rad=abs(wrapped_yaw_error(obj["yaw"], self.config.goal_yaw)) if finite else None,
            goal_hold_steps=self.goal_hold_count, end_reason=reason, failure_reason=failure_reason,
            reward_components=components.copy(), episode_metrics=deepcopy(self._metrics),
            initial_state=deepcopy(self._initial_state) if self.steps == 0 or reason else None,
            failure_details=deepcopy(self._failure_details),
        )

    def step(self, actions):
        if not self._has_reset:
            raise RuntimeError("call reset() before step()")
        if not self.agents:
            return {}, {}, {}, {}, {}
        names = self.agents.copy()
        if set(actions) != set(names):
            raise ValueError("provide one action for each live agent")
        array = finite_array([actions[n] for n in names], (self.config.num_agents, 2), "actions")
        previous = self._distance() if self._finite() else 0.
        cfg = self.config
        with self.viewer.lock() if self.viewer is not None else nullcontext():
            report = self.physics.step(array, max_penetration=cfg.max_penetration,
                                       max_linear_speed=cfg.max_linear_speed,
                                       max_angular_speed=cfg.max_angular_speed)
            self.steps += 1
            self._failed = not report["stable"]
            self._failure_details = report["failure_details"]
            settled = (not self._failed and goal_is_settled(
                self.physics, self.goal_center, self.goal_half_size,
                cfg.success_linear_speed, cfg.success_angular_speed))
            self.goal_hold_count = self.goal_hold_count + 1 if settled else 0
            success = not self._failed and self.goal_hold_count >= cfg.success_hold_steps
            terminated = success or self._failed
            truncated = self.steps >= self.max_steps and not terminated
            distance = self._distance() if self._finite() else previous
            yaw = self.physics.get_object_state()["yaw"] if not self._failed else 0.
            reward, parts = team_reward(previous, distance, yaw, success, self._failed, cfg)
            self._metrics["episode_return"] += reward
            for key in ("object_path_length_m", "control_effort_N2_s", "wall_collision_events"):
                self._metrics[key] += report[key]
            self._metrics["max_penetration_m"] = max(self._metrics["max_penetration_m"], report["max_penetration_m"])
            self._metrics["agent_object_contact_time_s"] = (
                np.asarray(self._metrics["agent_object_contact_time_s"]) + report["agent_object_contact_time_s"]).tolist()
            if not self._failed:
                self._metrics["inside_goal_steps"] += int(self.object_inside_goal())
            obs = self._observations()
            if not self._failed:
                self._last_observations = {n: a.copy() for n, a in obs.items()}
                self._last_state = self.state()
            reason = "success" if success else "physics_failure" if self._failed else "time_limit" if truncated else None
            info = self._info(success, parts, reason, report["failure_reason"])
            if terminated or truncated:
                self.agents = []
        return (obs, dict.fromkeys(names, reward), dict.fromkeys(names, bool(terminated)),
                dict.fromkeys(names, bool(truncated)), {n: deepcopy(info) for n in names})

    def render(self):
        if self.render_mode != "human":
            return None
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.physics.model, self.physics.data)
            self.viewer.cam.lookat[:] = [0, 0, 0]
            self.viewer.cam.distance = 8
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -90
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
