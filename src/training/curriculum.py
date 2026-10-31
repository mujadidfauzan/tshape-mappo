"""Validated curriculum definitions and checkpointable promotion state."""

from dataclasses import replace
from pathlib import Path

import yaml

from src.envs.config import ROOT

STAGE_FIELDS = {"max_steps", "goal_center", "object_yaw_range", "object_goal_distance_range"}


class Curriculum:
    def __init__(self, definition, state=None):
        allowed = {"difficulty", "validation_interval_steps", "validation_episodes", "validation_seed",
                   "success_threshold", "consecutive_checks", "minimum_stage_steps", "stages"}
        if not isinstance(definition, dict) or set(definition) - allowed:
            raise ValueError("unknown curriculum definition fields")
        self.definition = definition
        self.interval = definition.get("validation_interval_steps", 50000)
        self.episodes = definition.get("validation_episodes", 20)
        self.seed = definition.get("validation_seed", 150000)
        self.threshold = definition.get("success_threshold", .7)
        self.required = definition.get("consecutive_checks", 2)
        self.minimum_steps = definition.get("minimum_stage_steps", 50000)
        for value in (self.interval, self.episodes, self.required, self.minimum_steps):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("curriculum intervals, episodes and checks must be positive integers")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("curriculum validation seed must be nonnegative")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (float, int)) or not 0 < self.threshold <= 1:
            raise ValueError("curriculum success_threshold must lie in (0,1]")
        stages = definition.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("curriculum needs a nonempty stages list")
        names = []
        for stage in stages:
            if not isinstance(stage, dict) or set(stage) - STAGE_FIELDS - {"name"}:
                raise ValueError("unknown curriculum stage fields")
            if not isinstance(stage.get("name"), str) or not stage["name"]:
                raise ValueError("each curriculum stage needs a name")
            names.append(stage["name"])
        if len(set(names)) != len(names):
            raise ValueError("curriculum stage names must be unique")
        state = state or {}
        self.index = state.get("index", 0)
        self.streak = state.get("streak", 0)
        self.stage_start = state.get("stage_start", 0)
        self.completed = state.get("completed", False)
        self.last_validation_step = state.get("last_validation_step", 0)
        if (isinstance(self.index, bool) or not isinstance(self.index, int) or not 0 <= self.index < len(stages)
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                       for v in (self.streak, self.stage_start, self.last_validation_step))
                or not isinstance(self.completed, bool)):
            raise ValueError("invalid saved curriculum state")

    @classmethod
    def load(cls, source, state=None):
        if isinstance(source, dict):
            return cls(source, state)
        path = Path(source)
        if not path.is_absolute():
            path = ROOT / path
        return cls(yaml.safe_load(path.read_text()), state)

    @property
    def name(self):
        return self.definition["stages"][self.index]["name"]

    def config(self, base, index=None):
        if self.definition.get("difficulty", base.difficulty) != base.difficulty:
            raise ValueError("curriculum difficulty does not match the arena")
        stage = self.definition["stages"][self.index if index is None else index]
        values = {k: v for k, v in stage.items() if k != "name"}
        band = values.get("object_goal_distance_range", base.object_goal_distance_range)
        if band is not None:
            values["minimum_goal_distance"] = band[0]
        return replace(base, **values)

    def validate_configs(self, base):
        for i in range(len(self.definition["stages"])):
            self.config(base, i)
        final = self.config(base, len(self.definition["stages"]) - 1)
        if final.to_dict() != base.to_dict():
            raise ValueError("the final curriculum stage must restore the full baseline task")

    def due(self, step):
        return not self.completed and step // self.interval > self.last_validation_step // self.interval

    def observe(self, success_rate, step):
        self.last_validation_step = step
        self.streak = self.streak + 1 if success_rate >= self.threshold else 0
        if self.streak < self.required or step - self.stage_start < self.minimum_steps:
            return False
        self.streak = 0
        if self.index == len(self.definition["stages"]) - 1:
            self.completed = True
            return False
        self.index += 1
        self.stage_start = step
        return True

    def state_dict(self):
        return {"index": self.index, "streak": self.streak, "stage_start": self.stage_start,
                "completed": self.completed, "last_validation_step": self.last_validation_step}
