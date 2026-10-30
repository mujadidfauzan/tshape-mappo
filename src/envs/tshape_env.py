"""Four-agent Easy, Medium and Hard transport using the Parallel API."""

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
        self._rooms = (self.physics.transport_room_bounds(cfg.difficulty)
                       if cfg.difficulty != "easy" else None)
        if cfg.difficulty == "easy" and self.physics.obstacle_ids:
            raise ValueError("internal obstacles require difficulty: medium or hard")
        self.possible_agents = self.physics.agent_names.copy()
        self.agents = []
        self.max_steps = cfg.max_steps
        self.steps = 0
        self.goal_hold_count = 0
        self.goal_center = np.asarray(cfg.goal_center, dtype=np.float64)
        self.goal_half_size = np.asarray(cfg.goal_half_size, dtype=np.float64)
        self._object_local_corners = (self.physics.object_corners()
            - self.physics.data.xpos[self.physics.object_body, :2]) @ self.physics.rotation()
        if (np.any(self.goal_center - self.goal_half_size < self.physics.lower)
                or np.any(self.goal_center + self.goal_half_size > self.physics.upper)):
            raise ValueError("goal rectangle must be entirely inside the arena")
        if cfg.reset_distribution == "arena_wide" and np.any(
                2 * (self.goal_half_size + cfg.arena_spawn_margin) >= self.physics.upper - self.physics.lower):
            raise ValueError("goal rectangle and spawn margin leave no room for target randomization")
        if not self.physics.goal_clear_of_obstacles(self.goal_center, self.goal_half_size, .005):
            raise ValueError("goal rectangle must be clear of internal obstacles")
        if self._rooms is not None:
            for lower, upper in (self._rooms[0], self._rooms[-1]):
                if np.any(2 * (self.goal_half_size + cfg.arena_spawn_margin) >= upper - lower):
                    raise ValueError("goal rectangle and spawn margin must fit in either end room")
            self._start_room = None
            if not cfg.randomize_goal:
                for side in (0, len(self._rooms) - 1):
                    lower, upper = self._rooms[side]
                    if (np.all(self.goal_center - self.goal_half_size >= lower + cfg.arena_spawn_margin)
                            and np.all(self.goal_center + self.goal_half_size <= upper - cfg.arena_spawn_margin)):
                        self._start_room = len(self._rooms) - 1 - side
                        break
                if self._start_room is None:
                    raise ValueError("fixed goal must fit entirely in an end room with the spawn margin")
        self.render_mode = render_mode
        self.viewer = None
        self.metadata = {**type(self).metadata, "name": f"tshape_{cfg.difficulty}_v1",
                         "render_fps": max(1, round(1 / self.physics.control_dt))}
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
        if randomize and self.config.reset_distribution == "arena_wide":
            margin = self.config.arena_spawn_margin + p.radii[:, None]
            return self.np_random.uniform(p.lower + margin, p.upper - margin)
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

    def _sample_object_pose(self, room=None):
        p = self.physics
        yaw = self.np_random.uniform(-np.pi, np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        corners = self._object_local_corners @ np.array([[c, s], [-s, c]])
        bounds = (p.lower, p.upper) if room is None else self._rooms[room]
        lower = bounds[0] + self.config.arena_spawn_margin - corners.min(axis=0)
        upper = bounds[1] - self.config.arena_spawn_margin - corners.max(axis=0)
        if np.any(lower > upper):
            raise ValueError("transported object cannot fit within the arena spawn margin")
        return np.r_[self.np_random.uniform(lower, upper), yaw]

    def _sample_goal(self, room=None):
        margin = self.goal_half_size + self.config.arena_spawn_margin
        bounds = (self.physics.lower, self.physics.upper) if room is None else self._rooms[room]
        return self.np_random.uniform(bounds[0] + margin, bounds[1] - margin)

    def _obstacle_task_is_valid(self, clearance):
        """Start and goal stay in opposite end rooms, crossing every divider."""
        if self._rooms is None:
            return True
        corners = self.physics.object_corners()
        for side in (0, len(self._rooms) - 1):
            lower, upper = self._rooms[side]
            if np.all(corners >= lower + clearance) and np.all(corners <= upper - clearance):
                goal_lower, goal_upper = self._rooms[len(self._rooms) - 1 - side]
                return bool(np.all(self.goal_center - self.goal_half_size >= goal_lower + clearance)
                            and np.all(self.goal_center + self.goal_half_size <= goal_upper - clearance))
        return False

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
                wide = randomize and self.config.reset_distribution == "arena_wide"
                room = None
                if wide and self._rooms is not None:
                    if self.config.randomize_goal:
                        side = (int(explicit_pose[0] >= 0) if explicit_pose is not None
                                else int(self.np_random.integers(0, 2)))
                        room = side * (len(self._rooms) - 1)
                    else:
                        room = self._start_room
                if wide and self.config.randomize_goal:
                    goal_room = None if room is None else len(self._rooms) - 1 - room
                    self.goal_center[:] = self._sample_goal(goal_room)
                else:
                    self.goal_center[:] = self.config.goal_center
                self.physics.model.site("goal").pos[:2] = self.goal_center
                if explicit_pose is not None:
                    pose = explicit_pose.copy()
                elif wide:
                    pose = self._sample_object_pose(room)
                else:
                    pose = np.r_[self.physics.model.body_pos[self.physics.object_body, :2], 0.0]
                    if randomize:
                        noise = np.asarray(self.config.object_position_noise)
                        pose[:2] += self.np_random.uniform(-noise, noise)
                        pose[2] += self.np_random.uniform(-self.config.object_yaw_noise,
                                                         self.config.object_yaw_noise)
                positions = explicit_agents if explicit_agents is not None else self._spawn_positions(pose, randomize)
                self.physics.reset(pose, positions)
                if wide and (self._distance_to_goal() < self.config.minimum_goal_distance or self.object_inside_goal()):
                    continue
                # Reserve a small positive margin even when noise is disabled.
                clearance = max(0.005, self.config.spawn_clearance) if wide else 0.005
                if (self._obstacle_task_is_valid(clearance)
                        and self.physics.goal_clear_of_obstacles(self.goal_center, self.goal_half_size, clearance)
                        and self.physics.spawn_is_valid(clearance=clearance)):
                    break
                if not randomize or (explicit_pose is not None and explicit_agents is not None):
                    raise ValueError("requested reset overlaps a wall or body, or violates opposite-room placement")
            else:
                raise ValueError("could not sample a valid reset; check geometry, margins and goal separation")
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
            positions, _ = self.physics.get_agents_state()
            obj = self.physics.get_object_state()
            self._initial_state = {"goal_center": self.goal_center.tolist(),
                "object_position": obj["position"].tolist(), "object_yaw": obj["yaw"],
                "agent_positions": positions.tolist(), "reset_distribution": self.config.reset_distribution,
                "randomized": randomize, "goal_randomized": bool(wide and self.config.randomize_goal),
                "difficulty": self.config.difficulty}
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
                "initial_state": deepcopy(info.get("initial_state")),
                "episode_metrics": info["episode_metrics"].copy()}

    def _info(self, success, stable, progress, components, end_reason, failure_reason):
        distance = self._distance_to_goal() if self._finite_physics() else None
        if distance is not None and not np.isfinite(distance):
            distance = None
        elapsed = float(self.physics.data.time) if np.isfinite(self.physics.data.time) else None
        return {"success": bool(success), "stable": bool(stable), "distance_to_goal": distance,
                "difficulty": self.config.difficulty,
                "progress": float(progress), "episode_step": self.steps,
                "elapsed_time_s": elapsed, "completion_time_s": elapsed if success else None,
                "goal_hold_steps": self.goal_hold_count, "end_reason": end_reason,
                "failure_reason": failure_reason, "reward_components": components,
                "failure_details": self._failure_details,
                "goal_center": self.goal_center.tolist(),
                "initial_state": self._initial_state if self.steps == 0 or end_reason is not None else None,
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
