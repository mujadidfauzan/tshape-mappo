"""Fixed four-agent Easy task using the PettingZoo Parallel API."""

from contextlib import nullcontext
from copy import deepcopy

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from .config import load_config
from .observations import OBSERVATION_SIZE, STATE_SIZE, build_observations, build_state
from .physics import TShapePhysics, finite_array
from .rewards import team_reward
from .termination import goal_is_settled, object_inside_goal


class TShapeParallelEnv(ParallelEnv):
    metadata = {"name": "tshape_easy_v1", "render_modes": ["human"],
                "render_fps": 20, "is_parallelizable": True}

    def __init__(self, max_steps=None, frame_skip=None, render_mode=None, config=None):
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        self.config = cfg = load_config(config, max_steps=max_steps, frame_skip=frame_skip)
        self.physics = TShapePhysics(cfg.frame_skip, cfg.resolved_model_path)
        self.possible_agents = self.physics.agent_names.copy()
        self.agents = []
        self.max_steps = cfg.max_steps
        self.steps = 0
        self.goal_hold_count = 0
        self.goal_center = np.asarray(cfg.goal_center, dtype=np.float64)
        self.goal_half_size = np.asarray(cfg.goal_half_size, dtype=np.float64)
        if (np.any(self.goal_center - self.goal_half_size < self.physics.lower)
                or np.any(self.goal_center + self.goal_half_size > self.physics.upper)):
            raise ValueError("goal rectangle must be entirely inside the arena")
        self.render_mode = render_mode
        self.viewer = None
        self.metadata = {**type(self).metadata, "render_fps": max(1, round(1 / self.physics.control_dt))}
        self.obs_dim = OBSERVATION_SIZE
        self.state_space = spaces.Box(-np.inf, np.inf, shape=(STATE_SIZE,), dtype=np.float32)
        self._action_spaces = {name: spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
                               for name in self.possible_agents}
        self._observation_spaces = {name: spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
                                    for name in self.possible_agents}
        self._last_state = None
        self._last_observations = None
        self._physics_failed = False
        self._has_reset = False
        self._metrics = {}
        self._failure_details = None
        # The same goal rectangle is now visible in the viewer; sites do not collide.
        import mujoco
        marker = self.physics.model.site("goal")
        self.physics.model.site_type[marker.id] = mujoco.mjtGeom.mjGEOM_BOX
        marker.pos[:] = [*self.goal_center, 0.004]
        marker.size[:] = [*self.goal_half_size, 0.003]
        marker.rgba[:] = [0.1, 0.9, 0.2, 0.2]
        mujoco.mj_forward(self.physics.model, self.physics.data)

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def _fractions(self):
        return (max(0.0, 1.0 - self.steps / self.max_steps),
                min(1.0, self.goal_hold_count / self.config.success_hold_steps))

    def state(self):
        if self._last_state is not None and (self._physics_failed or not self._finite_physics()):
            return self._last_state.copy()
        remaining, hold = self._fractions()
        return build_state(self.physics, self.goal_center, self.goal_half_size, remaining, hold)

    def _finite_physics(self):
        return all(np.isfinite(a).all() for a in (self.physics.data.qpos, self.physics.data.qvel))

    def _observations(self):
        if self._physics_failed or not self._finite_physics():
            return {name: self._last_observations[name].copy() for name in self.agents}
        remaining, hold = self._fractions()
        observations = build_observations(self.physics, self.goal_center, self.goal_half_size,
                                          remaining, hold)
        return {name: observations[name] for name in self.agents}

    def _distance_to_goal(self):
        return float(np.linalg.norm(self.physics.get_object_state()["position"] - self.goal_center))

    def _is_success(self):
        return self.goal_hold_count >= self.config.success_hold_steps

    def _spawn_positions(self, pose, randomize):
        p = self.physics
        top = p.model.geom("t_horizontal")
        stem = p.model.geom("t_vertical")
        gap = self.config.spawn_clearance
        # Two agents below the large bar and two beside the stem: all four
        # start near usable contacts, independently of the scripted controller.
        local = np.array([
            [-0.6 * top.size[0], top.pos[1] - top.size[1] - p.radii[0] - gap],
            [0.6 * top.size[0], top.pos[1] - top.size[1] - p.radii[1] - gap],
            [stem.pos[0] - stem.size[0] - p.radii[2] - gap, stem.pos[1] - 0.5 * stem.size[1]],
            [stem.pos[0] - stem.size[0] - p.radii[3] - gap, stem.pos[1] + 0.5 * stem.size[1]],
        ])
        if randomize:
            local += self.np_random.uniform(-self.config.agent_position_noise,
                                            self.config.agent_position_noise, (4, 2))
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return local @ np.array([[c, s], [-s, c]]) + pose[:2]

    def reset(self, seed=None, options=None):
        if seed is not None or not hasattr(self, "np_random"):
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        # Seed action spaces too, so random-policy episodes can be replayed.
        for name in self.possible_agents:
            self.action_space(name).seed(int(self.np_random.integers(0, 2**32 - 1)))
        options = {} if options is None else dict(options)
        randomize = options.get("randomize", self.config.randomize_reset)
        if not isinstance(randomize, bool):
            raise ValueError("reset option 'randomize' must be a boolean")
        explicit_pose = options.get("object_pose")
        explicit_agents = options.get("agent_positions")
        if explicit_pose is not None:
            explicit_pose = finite_array(explicit_pose, (3,), "object_pose")
        if explicit_agents is not None:
            explicit_agents = finite_array(explicit_agents, (4, 2), "agent_positions")
        self._has_reset = False
        self.agents = []
        lock = self.viewer.lock() if self.viewer is not None else nullcontext()
        with lock:
            for _ in range(self.config.reset_attempts):
                if explicit_pose is not None:
                    pose = explicit_pose.copy()
                else:
                    pose = np.r_[self.physics.model.body_pos[self.physics.object_body, :2], 0.0]
                    if randomize:
                        noise = np.asarray(self.config.object_position_noise)
                        pose[:2] += self.np_random.uniform(-noise, noise)
                        pose[2] += self.np_random.uniform(-self.config.object_yaw_noise,
                                                         self.config.object_yaw_noise)
                positions = explicit_agents if explicit_agents is not None else self._spawn_positions(pose, randomize)
                self.physics.reset(pose, positions)
                # Reserve a small positive margin even when noise is disabled.
                if self.physics.spawn_is_valid(clearance=0.005):
                    break
                if not randomize or (explicit_pose is not None and explicit_agents is not None):
                    raise ValueError("requested reset overlaps an agent, object, or wall")
            else:
                raise ValueError("could not sample a collision-free reset; reduce noise or adjust spawn_clearance")
            self.steps = 0
            self.goal_hold_count = 0
            self.agents = self.possible_agents.copy()
            self._metrics = {"episode_return": 0.0, "object_path_length_m": 0.0,
                             "control_effort_N2_s": 0.0, "wall_collision_events": 0,
                             "wall_contact_physics_steps": 0, "max_penetration_m": 0.0}
            self._last_state = None
            self._last_observations = None
            self._physics_failed = False
            self._failure_details = None
            observations = self._observations()
            self._last_observations = {name: obs.copy() for name, obs in observations.items()}
            self._last_state = self.state()
            self._has_reset = True
            info = self._info(False, True, 0.0, {}, None, None)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()  # sync obtains its own lock
        return observations, {name: self._copy_info(info) for name in self.agents}

    @staticmethod
    def _copy_info(info):
        return {**info, "reward_components": info["reward_components"].copy(),
                "failure_details": deepcopy(info["failure_details"]),
                "episode_metrics": info["episode_metrics"].copy()}

    def _info(self, success, stable, progress, components, end_reason, failure_reason):
        distance = self._distance_to_goal() if self._finite_physics() else None
        if distance is not None and not np.isfinite(distance):
            distance = None
        elapsed = float(self.physics.data.time) if np.isfinite(self.physics.data.time) else None
        return {"success": bool(success), "stable": bool(stable), "distance_to_goal": distance,
                "progress": float(progress), "episode_step": self.steps,
                "elapsed_time_s": elapsed, "completion_time_s": elapsed if success else None,
                "goal_hold_steps": self.goal_hold_count, "end_reason": end_reason,
                "failure_reason": failure_reason, "reward_components": components,
                "failure_details": self._failure_details,
                "episode_metrics": self._metrics.copy()}

    def step(self, actions):
        if not self._has_reset:
            raise RuntimeError("call reset() before step()")
        if not self.agents:
            return {}, {}, {}, {}, {}
        current_agents = self.agents.copy()
        if set(actions) != set(current_agents):
            raise ValueError("actions must be provided for all and only live agents")
        action_array = finite_array([actions[name] for name in current_agents], (4, 2), "actions")
        previous_distance = self._distance_to_goal() if self._finite_physics() else 0.0
        with self.viewer.lock() if self.viewer is not None else nullcontext():
            report = self.physics.step(
                action_array, max_penetration=self.config.max_penetration,
                max_linear_speed=self.config.max_linear_speed,
                max_angular_speed=self.config.max_angular_speed,
            )
            self.steps += 1
            stable = report["stable"]
            self._failure_details = report["failure_details"]
            self._physics_failed = not stable
            if stable and goal_is_settled(self.physics, self.goal_center, self.goal_half_size,
                                          self.config.success_linear_speed, self.config.success_angular_speed):
                self.goal_hold_count += 1
            else:
                self.goal_hold_count = 0
            success = stable and self._is_success()
            terminated = success or not stable
            truncated = self.steps >= self.max_steps and not terminated
            distance = self._distance_to_goal() if self._finite_physics() else previous_distance
            reward, progress, components = team_reward(previous_distance, distance, success,
                                                       not stable, self.config)
            self._metrics["episode_return"] += reward
            for key in ("object_path_length_m", "control_effort_N2_s", "wall_collision_events",
                        "wall_contact_physics_steps"):
                self._metrics[key] += report[key]
            self._metrics["max_penetration_m"] = max(self._metrics["max_penetration_m"], report["max_penetration_m"])
            observations = self._observations()
            if stable:
                self._last_observations = {name: obs.copy() for name, obs in observations.items()}
                self._last_state = self.state()
            reason = "success" if success else "physics_failure" if not stable else "time_limit" if truncated else None
            info = self._info(success, stable, progress, components, reason, report["failure_reason"])
            rewards = {name: reward for name in current_agents}
            terminations = {name: bool(terminated) for name in current_agents}
            truncations = {name: bool(truncated) for name in current_agents}
            infos = {name: self._copy_info(info) for name in current_agents}
            if terminated or truncated:
                self.agents = []
        return observations, rewards, terminations, truncations, infos

    def object_inside_goal(self):
        return object_inside_goal(self.physics.object_corners(), self.goal_center, self.goal_half_size)

    def render(self):
        if self.render_mode != "human":
            return None
        import mujoco
        import mujoco.viewer as mj_viewer
        if self.viewer is None:
            self.viewer = mj_viewer.launch_passive(self.physics.model, self.physics.data)
            with self.viewer.lock():
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = self.physics.model.camera("top_view").id
        if self.viewer.is_running():
            self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
