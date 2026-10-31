"""Episode-level task phases; these measurements do not enter policy inputs."""

import numpy as np

from .termination import footprint_error, footprint_margin


class EpisodeDiagnostics:
    def __init__(self, physics, goal, half):
        self.goal, self.half = np.asarray(goal).copy(), np.asarray(half).copy()
        self.contact_time = np.zeros(physics.num_agents)
        self.inside_steps = self.unsettled_steps = 0
        self.ever_contact = self.last_contact = False
        self.min_error = np.inf
        self.best_margin = -np.inf
        self.direction = 1 if goal[0] >= physics.get_object_state()["position"][0] else -1
        boxes = {(round(float(lo[0]), 8), round(float(hi[0]), 8)) for lo, hi in physics.obstacle_boxes}
        self.gates = sorted(boxes, reverse=self.direction < 0)
        self.cleared = [False] * len(self.gates)
        self.center_crossed = [False] * len(self.gates)
        self.first_clear_step = [None] * len(self.gates)
        self.reached = False
        self.update(physics, 0, False, False)

    def update(self, physics, step, inside, settled, contact_time=None):
        corners = physics.object_corners()
        self.error = footprint_error(corners, self.goal, self.half)
        self.margin = footprint_margin(corners, self.goal, self.half)
        self.min_error, self.best_margin = min(self.min_error, self.error), max(self.best_margin, self.margin)
        flags, _, _ = physics.contact_state()
        self.last_contact = bool(flags.any())
        self.ever_contact |= self.last_contact
        if contact_time is not None:
            self.contact_time += contact_time
            self.ever_contact |= bool(np.any(contact_time > 0))
        self.inside_steps += int(inside and step > 0)
        self.unsettled_steps += int(inside and not settled and step > 0)
        x = physics.get_object_state()["position"][0]
        for i, (left, right) in enumerate(self.gates):
            self.reached |= bool(corners[:, 0].min() <= right + .1 and corners[:, 0].max() >= left - .1)
            self.center_crossed[i] |= bool(x > right if self.direction > 0 else x < left)
            full = corners[:, 0].min() > right if self.direction > 0 else corners[:, 0].max() < left
            if full and not self.cleared[i]:
                self.cleared[i], self.first_clear_step[i] = True, step

    def report(self, success=False, stable=True, terminal=False):
        category = None
        if terminal:
            if not stable:
                category = "physics_failure"
            elif success:
                category = "success"
            elif self.gates and not all(self.cleared):
                category = "gate_blocked" if self.reached else "gate_not_reached"
            elif self.inside_steps:
                category = "goal_not_settled"
            elif self.ever_contact and not self.last_contact:
                category = "contact_lost"
            else:
                category = "goal_footprint_outside"
        return {"failure_category": category, "footprint_error_m": self.error,
                "footprint_margin_m": self.margin, "min_footprint_error_m": self.min_error,
                "best_footprint_margin_m": self.best_margin, "inside_goal_steps": self.inside_steps,
                "inside_but_unsettled_steps": self.unsettled_steps,
                "gates_center_crossed": list(self.center_crossed), "gates_fully_cleared": list(self.cleared),
                "gate_first_clear_steps": list(self.first_clear_step),
                "gate_clear_fraction": sum(self.cleared)/len(self.cleared) if self.cleared else 1.0,
                "agent_object_contact_time_s": self.contact_time.tolist(),
                "contact_lost_at_end": bool(self.ever_contact and not self.last_contact)}
