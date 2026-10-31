"""Materialize a self-contained agent-count variant of an arena XML."""

from copy import deepcopy
import colorsys
import re
import xml.etree.ElementTree as ET

import numpy as np


def model_with_agents(xml, num_agents):
    if isinstance(num_agents, bool) or not isinstance(num_agents, int) or not 1 <= num_agents <= 16:
        raise ValueError("num_agents must be an integer between 1 and 16")
    root = ET.fromstring(xml)
    world, actuator = root.find("worldbody"), root.find("actuator")
    bodies = [b for b in world.findall("body") if re.fullmatch(r"agent_\d+", b.get("name", ""))]
    if len(bodies) == num_agents:
        return xml  # Preserve historical four-agent XML bytes and hashes.
    if not bodies or actuator is None:
        raise ValueError("arena must contain an agent template and actuators")
    template = next(b for b in bodies if b.get("name") == "agent_0")
    motors = [next(m for m in actuator if m.get("name") == f"agent_0_f{axis}") for axis in ("x", "y")]
    for body in bodies:
        world.remove(body)
    for motor in list(actuator):
        if re.fullmatch(r"agent_\d+_f[xy]", motor.get("name", "")):
            actuator.remove(motor)
    origin = np.fromstring(world.find("body[@name='t_shape']").get("pos"), sep=" ")[:2]
    for i in range(num_agents):
        body = deepcopy(template)
        for element in body.iter():
            for attr, value in list(element.attrib.items()):
                element.set(attr, value.replace("agent_0", f"agent_{i}"))
        theta = 2 * np.pi * i / num_agents
        position = origin + .95 * np.array([np.cos(theta), np.sin(theta)])
        height = np.fromstring(template.get("pos"), sep=" ")[2]
        body.set("pos", f"{position[0]} {position[1]} {height}")
        color = colorsys.hsv_to_rgb((i * .61803398875) % 1, .75, .85)
        body.find("geom").set("rgba", " ".join(map(str, (*color, 1))))
        world.append(body)
        for motor in motors:
            clone = deepcopy(motor)
            for attr, value in list(clone.attrib.items()):
                clone.set(attr, value.replace("agent_0", f"agent_{i}"))
            actuator.append(clone)
    return ET.tostring(root, encoding="unicode")
