"""Validated, serializable configuration for four-agent transport tasks."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "environment.yaml"


@dataclass(frozen=True)
class EnvironmentConfig:
    difficulty: str = "easy"
    model_path: str = "assets/mujoco/arena.xml"
    frame_skip: int = 50
    max_steps: int = 500
    goal_center: tuple = (1.8, 1.5)
    goal_half_size: tuple = (0.75, 0.65)
    success_linear_speed: float = 0.05
    success_angular_speed: float = 0.1
    success_hold_steps: int = 5
    randomize_reset: bool = True
    # Easy randomizes goals; obstacle tasks use a fixed target.
    randomize_goal: bool | None = None
    reset_distribution: str = "near_object"
    arena_spawn_margin: float = 0.15
    minimum_goal_distance: float = 1.5
    agent_spawn_gap: tuple = (0.04, 0.15)
    spawn_clearance: float = 0.04
    reset_attempts: int = 100
    progress_weight: float = 2.0
    success_reward: float = 20.0
    time_penalty: float = 0.001
    failure_penalty: float = 20.0
    max_penetration: float = 0.005
    max_linear_speed: float = 5.0
    max_angular_speed: float = 10.0

    def __post_init__(self):
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError("difficulty must be easy, medium or hard")
        if self.randomize_goal is None:
            object.__setattr__(self, "randomize_goal", self.difficulty == "easy")
        if not isinstance(self.randomize_goal, bool):
            raise ValueError("randomize_goal must be a boolean")
        if self.difficulty != "easy" and self.randomize_goal:
            raise ValueError("Medium and Hard require a fixed target")
        for name in ("frame_skip", "max_steps", "success_hold_steps", "reset_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if not isinstance(self.randomize_reset, bool):
            raise ValueError("randomize_reset must be a boolean")
        if self.reset_distribution != "near_object":
            raise ValueError("reset_distribution must be near_object; obsolete reset modes were removed")
        for name in ("goal_center", "goal_half_size", "agent_spawn_gap"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain two finite numbers")
            if name == "goal_half_size" and np.any(value <= 0):
                raise ValueError("goal_half_size must be positive")
            if name == "agent_spawn_gap" and (value[0] < 0 or value[1] < value[0]):
                raise ValueError("agent_spawn_gap must be an ordered nonnegative [minimum, maximum]")
            object.__setattr__(self, name, tuple(float(x) for x in value))
        scalar_names = (
            "success_linear_speed", "success_angular_speed", "spawn_clearance", "progress_weight",
            "success_reward", "time_penalty", "failure_penalty", "max_penetration",
            "max_linear_speed", "max_angular_speed",
            "arena_spawn_margin", "minimum_goal_distance",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.agent_spawn_gap[0] < self.spawn_clearance:
            raise ValueError("agent_spawn_gap minimum must be at least spawn_clearance")
        if self.max_linear_speed <= 0 or self.max_angular_speed <= 0:
            raise ValueError("safety speed limits must be positive")
        if not isinstance(self.model_path, str) or not self.model_path:
            raise ValueError("model_path must be a nonempty path string")

    @property
    def resolved_model_path(self):
        path = Path(self.model_path).expanduser()
        return path if path.is_absolute() else ROOT / path

    def to_dict(self):
        return asdict(self)


def load_config(config=None, **overrides):
    """Accept a YAML path, mapping or config object; paths inside YAML use repo root."""
    if isinstance(config, EnvironmentConfig):
        result = config
    else:
        if config is None:
            config = DEFAULT_CONFIG_PATH
        if isinstance(config, (str, Path)):
            with Path(config).open(encoding="utf-8") as stream:
                values = yaml.safe_load(stream)
        else:
            values = config
        if not isinstance(values, Mapping):
            raise ValueError("environment configuration must be a mapping")
        unknown = set(values) - {field.name for field in fields(EnvironmentConfig)}
        if unknown:
            raise ValueError(f"unknown environment configuration keys: {sorted(unknown)}")
        result = EnvironmentConfig(**values)
    return replace(result, **{key: value for key, value in overrides.items() if value is not None})


def load_checkpoint_config(values):
    """Migrate saved task settings to the current reset contract, keeping XML separate.

    Old sampling implementations are deliberately not retained in the runtime.
    Training/evaluation record the changed reset distribution in their metadata.
    """
    values = dict(values)
    for removed in ("object_position_noise", "object_yaw_noise", "agent_position_noise"):
        values.pop(removed, None)
    values["reset_distribution"] = "near_object"
    if values.get("difficulty", "easy") != "easy":
        values["randomize_goal"] = False
    return load_config(values)
