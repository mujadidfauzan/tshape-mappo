"""Reference diagonal push for the shipped fixed layout; never used by training."""

import numpy as np


def scripted_actions(env):
    if env.config.num_agents != 4 or env.config.randomize_agents:
        raise ValueError("the reference controller requires the fixed four-agent demo layout")
    if env.object_inside_goal():
        return {name: np.zeros(2, np.float32) for name in env.possible_agents}
    p = env.physics
    obj = p.get_object_state()
    positions, velocities = p.get_agents_state()
    rotation = p.rotation()
    direction = rotation.T @ (env.goal_center-obj["position"])
    direction /= max(np.linalg.norm(direction),1e-9)
    top = p.model.geom("t_horizontal")
    stem = p.model.geom("t_vertical")
    surfaces = [np.array([-.6*top.size[0],top.pos[1]-top.size[1]]),
                np.array([.6*top.size[0],top.pos[1]-top.size[1]]),
                np.array([stem.pos[0]-stem.size[0],-.5*stem.size[1]]),
                np.array([stem.pos[0]-stem.size[0],.5*stem.size[1]])]
    actions = {}
    for i, name in enumerate(p.agent_names):
        local_normal = np.array([0.,1.]) if i<2 else np.array([1.,0.])
        normal = rotation @ local_normal
        tangent = np.array([-normal[1],normal[0]])
        local = surfaces[i]-(p.radii[i]-.0005)*local_normal
        offset = rotation @ local
        target = p.data.xpos[p.object_body, :2] + offset
        desired_velocity = obj["velocity"] + obj["angular_velocity"]*np.array([-offset[1],offset[0]])
        error = target-positions[i]
        speed = velocities[i]-desired_velocity
        alignment = max(0.,1.-abs(float(error@tangent))/p.radii[i])
        push = 2.*max(float(direction@local_normal),0.)*alignment
        force = (30*float(error@normal)-4*float(speed@normal)+push)*normal
        force += (30*float(error@tangent)-2*float(speed@tangent))*tangent
        low,high = p.motor_limits[i,:,0],p.motor_limits[i,:,1]
        actions[name] = np.clip(np.where(force>=0,force/high,force/(-low)),-1,1).astype(np.float32)
    return actions
