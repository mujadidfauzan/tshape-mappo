"""Privileged four-agent translation controller for debugging Easy episodes.

This reads global geometry/state and is not a learned or decentralized policy.
It is a mechanics baseline, not a guarantee of task success or a path planner.
"""

import numpy as np


def contact_target_depth(radius, penetration_limit):
    """Small position bias for contact, below the environment's failure limit.

    This is a controller target, not a guarantee about the actual contact
    penetration produced by the solver or by another agent.
    """
    return min(0.001, 0.05 * radius, 0.2 * penetration_limit)


def scripted_actions(env, push_force=2.0):
    if not np.isfinite(push_force) or push_force <= 0:
        raise ValueError("push_force must be finite and positive")
    p = env.physics
    if p.num_agents != 4:
        raise ValueError("The scripted mechanics diagnostic is defined for four agents only; use random or learned policies")
    result = np.zeros((4, 2), dtype=np.float64)
    if env.object_inside_goal():
        return {name: result[i].astype(np.float32) for i, name in enumerate(p.agent_names)}
    obj = p.get_object_state()
    rotation = p.rotation()
    goal = rotation.T @ (env.goal_center - obj["position"])
    direction = goal / max(np.linalg.norm(goal), 1e-9)
    top = p.model.geom("t_horizontal")
    bottom = p.model.geom("t_bottom")
    stem = p.model.geom("t_vertical")
    # A symmetric pair supplies Y force; another pair supplies X force.
    # The paired lever arms reduce unwanted yaw without imposing object motion.
    y_sign = 1.0 if direction[1] >= 0 else -1.0
    y_bar = top if y_sign > 0 else bottom
    y_surface = y_bar.pos[1] - y_sign * y_bar.size[1]
    x_sign = 1.0 if direction[0] >= 0 else -1.0
    x_surface = stem.pos[0] - x_sign * stem.size[0]
    surfaces = [np.array([-0.6 * y_bar.size[0], y_surface]),
                np.array([0.6 * y_bar.size[0], y_surface]),
                np.array([x_surface, stem.pos[1] - 0.5 * stem.size[1]]),
                np.array([x_surface, stem.pos[1] + 0.5 * stem.size[1]])]
    normals = [np.array([0., y_sign]), np.array([0., y_sign]),
               np.array([x_sign, 0.]), np.array([x_sign, 0.])]
    positions, velocities = p.get_agents_state()
    origin = p.data.xpos[p.object_body, :2]
    for i in range(4):
        depth = contact_target_depth(p.radii[i], env.config.max_penetration)
        local = surfaces[i] - (p.radii[i] - depth) * normals[i]
        offset = rotation @ local
        target = origin + offset
        com_offset = target - obj["position"]
        target_velocity = obj["velocity"] + obj["angular_velocity"] * np.array([-com_offset[1], com_offset[0]])
        component = abs(direction[1] if i < 2 else direction[0])
        normal = rotation @ normals[i]
        tangent = np.array([-normal[1], normal[0]])
        error = target - positions[i]
        relative_velocity = velocities[i] - target_velocity
        # Do not add inward feed-forward while still sliding towards the
        # assigned point: that can drive an agent into an edge or the stem.
        alignment = max(0.0, 1.0 - abs(float(error @ tangent)) / p.radii[i])
        normal_force = (30.0 * float(error @ normal)
                        - 4.0 * float(relative_velocity @ normal)
                        + push_force * component * alignment)
        tangent_force = 30.0 * float(error @ tangent) - 2.0 * float(relative_velocity @ tangent)
        force = normal_force * normal + tangent_force * tangent
        # Invert the zero-centered action-to-force mapping in TShapePhysics.
        low, high = p.motor_limits[i, :, 0], p.motor_limits[i, :, 1]
        result[i] = np.clip(np.where(force >= 0, force / high, force / (-low)), -1, 1)
    return {name: result[i].astype(np.float32) for i, name in enumerate(p.agent_names)}
