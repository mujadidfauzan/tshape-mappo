"""Deterministic M1 physics diagnostics; all object motion comes from contact.

Run the full suite with: python scripts/test_physics.py --headless --scenario all
"""

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.envs.physics import TShapePhysics
from src.envs.scripted import contact_target_depth

MODEL_PATH = ROOT / "assets" / "mujoco" / "arena.xml"
DURATIONS = {
    "idle": 3.0, "translation": 3.0, "rotation": 3.0, "wall": 6.0,
    "release": 12.0, "reset": 2.0, "stability": 60.0,
}
PUSH_SECONDS = 2.0
MAX_PENETRATION = 0.005  # m; soft contacts allow small penetration
MAX_LINEAR_SPEED = 5.0  # m/s; diagnostic safety bounds, not engine limits
MAX_ANGULAR_SPEED = 10.0  # rad/s
MAX_COAST_DISTANCE = 0.25  # m after the default scripted push is released
MAX_SETTLING_TIME = 1.5  # s until linear speed falls below 1 mm/s


@dataclass(frozen=True)
class Attachment:
    agent: int
    surface: np.ndarray  # contact point in object coordinates
    direction: np.ndarray  # inward push direction in object coordinates


class Physics(TShapePhysics):
    """M1 scenario controllers using the same mechanics as the M2 environment."""

    def __init__(self, path=MODEL_PATH):
        super().__init__(frame_skip=1, model_path=path)
        self.reset("idle")

    def attachments(self, scenario):
        bar = self.model.geom("t_horizontal")
        lever = 0.6 * bar.size[0]
        bottom, top = bar.pos[1] - bar.size[1], bar.pos[1] + bar.size[1]
        if scenario == "idle":
            return []
        if scenario == "rotation":
            return [Attachment(0, np.array([lever, bottom]), np.array([0., 1.])),
                    Attachment(1, np.array([-lever, top]), np.array([0., -1.]))]
        return [Attachment(i, np.array([x, bottom]), np.array([0., 1.]))
                for i, x in enumerate((-lever, lever))]

    def reset(self, scenario):
        m, d = self.model, self.data
        TShapePhysics.reset(self)
        if scenario == "release":
            # Leave enough free travel to measure damping without a wall stop.
            d.qpos[self.object_qpos[1]] = -1.0
        if scenario == "wall":
            bar = m.geom("t_horizontal")
            d.qpos[self.object_qpos[1]] = self.upper[1] - bar.pos[1] - bar.size[1] - 0.35
        mujoco.mj_forward(m, d)
        origin = d.xpos[self.object_body, :2].copy()
        for a in self.attachments(scenario):
            local = a.surface - (self.radii[a.agent] + 0.04) * a.direction
            position = origin + self.rotation() @ local
            d.qpos[self.agent_qpos[a.agent]] = position - m.body_pos[self.agent_bodies[a.agent], :2]
        mujoco.mj_forward(m, d)
        if d.ncon:
            raise ValueError(f"Scenario {scenario} starts with unexpected contacts")

    def controller(self, scenario, force):
        m, d = self.model, self.data
        controls = np.zeros(m.nu)
        if scenario == "idle" or (scenario == "release" and d.time >= PUSH_SECONDS):
            return controls
        origin = d.xpos[self.object_body, :2]
        rotation = self.rotation()
        linear = d.qvel[self.object_dofs[:2]]
        angular = d.qvel[self.object_dofs[2]]
        for a in self.attachments(scenario):
            # Match the shallow M2 contact bias; keep well below the diagnostic
            # penetration threshold. Controls only drive agent motors.
            depth = contact_target_depth(self.radii[a.agent], MAX_PENETRATION)
            local = a.surface - (self.radii[a.agent] - depth) * a.direction
            offset = rotation @ local
            target = origin + offset
            target_velocity = linear + angular * np.array([-offset[1], offset[0]])
            position = d.xpos[self.agent_bodies[a.agent], :2]
            velocity = d.qvel[self.agent_dofs[a.agent]]
            effort = 30.0 * (target - position) + 2.0 * (target_velocity - velocity)
            effort += force * (rotation @ a.direction)
            indices = self.motors[a.agent]
            controls[indices] = np.clip(effort, m.actuator_ctrlrange[indices, 0],
                                       m.actuator_ctrlrange[indices, 1])
        return controls

    def state(self):
        # Report center of mass in world XY and unwrapped joint yaw.
        return np.r_[self.data.xipos[self.object_body, :2],
                     self.data.qpos[self.object_qpos[2]]].copy()

    def integration_snapshot(self):
        d = self.data
        return np.concatenate(([d.time], d.qpos, d.qvel, d.act, d.ctrl,
                               d.qacc_warmstart, d.qfrc_applied,
                               d.xfrc_applied.ravel(), d.userdata))


def run_scenario(physics, scenario, duration, force, viewer=None):
    m, d = physics.model, physics.data
    with viewer.lock() if viewer is not None else nullcontext():
        physics.reset(scenario)
    initial_snapshot = physics.integration_snapshot()
    initial = physics.state()
    initial_qpos = d.qpos.copy()
    report = {
        "scenario": scenario, "requested_duration_s": duration,
        "initial_state": initial.tolist(), "object_contact_steps": 0,
        "object_wall_contact_steps": 0, "wall_contact_steps": 0,
        "max_penetration_m": 0.0, "max_contact_force_N": 0.0,
        "max_object_speed_m_s": 0.0, "max_agent_speed_m_s": 0.0,
        "max_object_yaw_speed_rad_s": 0.0, "max_yaw_change_rad": 0.0,
        "max_control_N_per_axis": 0.0, "max_idle_drift": 0.0,
        "release_speed_m_s": None, "floor_contact_steps": 0,
        "coast_distance_m": None, "settling_time_s": None,
    }
    release_position = None
    release_time = None
    failures = []
    force_vector = np.zeros(6)
    steps = math.ceil(duration / m.opt.timestep)
    completed = 0
    for step in range(steps):
        if viewer is not None and not viewer.is_running():
            failures.append("viewer closed before scenario completed")
            break
        started = time.perf_counter()
        if scenario == "release" and d.time >= PUSH_SECONDS and report["release_speed_m_s"] is None:
            report["release_speed_m_s"] = float(np.linalg.norm(d.qvel[physics.object_dofs[:2]]))
            release_position = physics.state()[:2].copy()
            release_time = d.time
        with viewer.lock() if viewer is not None else nullcontext():
            d.ctrl[:] = physics.controller(scenario, force)
            report["max_control_N_per_axis"] = max(report["max_control_N_per_axis"], float(np.max(np.abs(d.ctrl))))
            old_time = d.time
            mujoco.mj_step(m, d)
            # Refresh world poses/contact forces at the integrated state before measuring.
            mujoco.mj_forward(m, d)
        completed += 1
        if not all(np.isfinite(a).all() for a in (d.qpos, d.qvel, d.qacc, d.ctrl)):
            failures.append("non-finite simulation state")
            break
        if not np.isclose(d.time, old_time + m.opt.timestep, rtol=0, atol=1e-9):
            failures.append("simulation time jumped (possible automatic reset)")
            break
        if np.any(d.warning.number):
            failures.append("MuJoCo warning")
            break
        state = physics.state()
        speed = float(np.linalg.norm(d.qvel[physics.object_dofs[:2]]))
        agent_speed = float(np.max(np.linalg.norm(d.qvel[physics.agent_dofs], axis=1)))
        yaw_speed = float(abs(d.qvel[physics.object_dofs[2]]))
        report["max_object_speed_m_s"] = max(report["max_object_speed_m_s"], speed)
        report["max_agent_speed_m_s"] = max(report["max_agent_speed_m_s"], agent_speed)
        report["max_object_yaw_speed_rad_s"] = max(report["max_object_yaw_speed_rad_s"], yaw_speed)
        report["max_yaw_change_rad"] = max(report["max_yaw_change_rad"], float(abs(state[2] - initial[2])))
        if release_position is not None:
            coast_distance = float(np.linalg.norm(state[:2] - release_position))
            report["coast_distance_m"] = max(report["coast_distance_m"] or 0.0, coast_distance)
            # Reset the timer if speed subsequently rises above the threshold.
            if speed >= 0.001:
                report["settling_time_s"] = None
            elif report["settling_time_s"] is None:
                report["settling_time_s"] = float(d.time - release_time)
        report["max_idle_drift"] = max(report["max_idle_drift"], float(np.max(np.abs(d.qpos - initial_qpos))))
        penetration = physics.boundary_penetration()
        object_contact = object_wall = wall_contact = floor_contact = False
        for index in range(d.ncon):
            contact = d.contact[index]
            if contact.efc_address < 0:
                continue
            pair = {int(contact.geom1), int(contact.geom2)}
            object_contact |= bool(pair & physics.object_geoms and pair & physics.agent_geoms)
            object_wall |= bool(pair & physics.object_geoms and pair & physics.walls)
            wall_contact |= bool(pair & physics.walls)
            floor_contact |= physics.floor in pair
            penetration = max(penetration, float(-contact.dist))
            mujoco.mj_contactForce(m, d, index, force_vector)
            report["max_contact_force_N"] = max(report["max_contact_force_N"], float(np.linalg.norm(force_vector[:3])))
        report["object_contact_steps"] += int(object_contact)
        report["object_wall_contact_steps"] += int(object_wall)
        report["wall_contact_steps"] += int(wall_contact)
        report["floor_contact_steps"] += int(floor_contact)
        report["max_penetration_m"] = max(report["max_penetration_m"], penetration)
        if penetration > MAX_PENETRATION:
            failures.append("penetration exceeds 5 mm tolerance")
            break
        if max(speed, agent_speed) > MAX_LINEAR_SPEED or yaw_speed > MAX_ANGULAR_SPEED:
            failures.append("speed exceeds diagnostic safety bounds")
            break
        if viewer is not None:
            viewer.sync()
            remaining = m.opt.timestep - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
    final = physics.state()
    report.update({
        "steps": completed, "simulated_duration_s": float(d.time),
        "final_state": final.tolist(),
        "displacement_m": float(np.linalg.norm(final[:2] - initial[:2])),
        "final_speed_m_s": float(np.linalg.norm(d.qvel[physics.object_dofs[:2]])),
        "warnings": {mujoco.mjtWarning(i).name: int(w.number)
                     for i, w in enumerate(d.warning) if w.number},
    })
    if report["floor_contact_steps"]:
        failures.append("visual floor unexpectedly participates in collisions")
    if scenario == "idle":
        if report["max_idle_drift"] > 1e-9:
            failures.append("zero-action state drift exceeds 1e-9")
    else:
        if not report["object_contact_steps"]:
            failures.append("no agent-object contact detected")
    if scenario in ("translation", "reset", "release"):
        if final[1] - initial[1] < 0.15:
            failures.append("forward displacement below 0.15 m")
        if abs(final[0] - initial[0]) > 0.05 or report["max_yaw_change_rad"] > 0.1:
            failures.append("symmetric push has excessive lateral drift or rotation")
    if scenario == "rotation" and report["max_yaw_change_rad"] < 0.35:
        failures.append("rotation below 0.35 rad")
    if scenario == "wall" and not report["object_wall_contact_steps"]:
        failures.append("object never reached the wall")
    if scenario == "release":
        release_speed = report["release_speed_m_s"]
        if release_speed is None or release_speed < 0.05:
            failures.append("object was not moving at force release")
        elif report["final_speed_m_s"] > 0.1 * release_speed:
            failures.append("speed did not fall below 10% of release speed")
        if report["wall_contact_steps"]:
            failures.append("wall contact confounds the damping measurement")
        if report["coast_distance_m"] is not None and report["coast_distance_m"] > MAX_COAST_DISTANCE:
            failures.append("post-release travel exceeds 0.25 m")
        if report["settling_time_s"] is None or report["settling_time_s"] > MAX_SETTLING_TIME:
            failures.append("object did not settle below 1 mm/s within 1.5 s of release")
    report["failures"] = failures
    report["passed"] = not failures
    return report, initial_snapshot, physics.integration_snapshot()


def execute(physics, scenarios, args, viewer=None):
    reports = []
    for scenario in scenarios:
        duration = args.duration if args.duration is not None else DURATIONS[scenario]
        repeats = 3 if scenario == "reset" else 1
        reference_initial = reference_final = None
        for repeat in range(repeats):
            if viewer is not None and not viewer.is_running():
                return reports
            report, initial, final = run_scenario(physics, scenario, duration, args.force, viewer)
            if scenario == "reset":
                report["repeat"] = repeat + 1
                if reference_initial is None:
                    reference_initial, reference_final = initial, final
                else:
                    error = float(max(np.max(np.abs(initial - reference_initial)),
                                      np.max(np.abs(final - reference_final))))
                    report["reset_replay_error"] = error
                    if error > 1e-12:
                        report["failures"].append("reset did not reproduce the initial state and trajectory")
                        report["passed"] = False
            reports.append(report)
            status = "PASS" if report["passed"] else "FAIL"
            print(f"[{status}] {scenario:11s} time={report['simulated_duration_s']:6.2f}s "
                  f"distance={report['displacement_m']:.4f}m "
                  f"yaw={report['max_yaw_change_rad']:.4f}rad "
                  f"penetration={1000 * report['max_penetration_m']:.3f}mm")
            for failure in report["failures"]:
                print(f"       {failure}")
    return reports


def positive_number(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without importing the GUI viewer")
    parser.add_argument("--scenario", choices=[*DURATIONS, "all"], default="translation")
    parser.add_argument("--duration", type=positive_number, help="override simulated seconds per scenario")
    parser.add_argument("--force", type=positive_number, default=1.0,
                        help="feed-forward push per active agent in N (default: 1; final motor force includes PD)")
    parser.add_argument("--json", type=Path, help="write measurements and reproducibility metadata")
    args = parser.parse_args()
    scenarios = list(DURATIONS) if args.scenario == "all" else [args.scenario]
    if "release" in scenarios and args.duration is not None and args.duration < 8:
        parser.error("release requires at least 8 seconds (2 s pushing, then coasting)")
    physics = Physics()
    limits = physics.model.actuator_ctrlrange[physics.motors]
    if args.force > float(np.min(np.minimum(-limits[..., 0], limits[..., 1]))):
        parser.error("--force exceeds an XML motor force limit")
    print(f"MuJoCo {mujoco.__version__}: {len(physics.agent_names)} agents, "
          f"{physics.model.nu} motors, dt={physics.model.opt.timestep}s")
    if args.headless:
        reports = execute(physics, scenarios, args)
    else:
        import mujoco.viewer as mj_viewer
        with mj_viewer.launch_passive(physics.model, physics.data) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = physics.model.camera("top_view").id
            reports = execute(physics, scenarios, args, viewer)
    expected = sum(3 if s == "reset" else 1 for s in scenarios)
    passed = len(reports) == expected and all(r["passed"] for r in reports)
    if args.json:
        payload = {
            "passed": passed, "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
            "model_xml": MODEL_PATH.read_text(),
            "software": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                         "numpy": np.__version__, "platform": platform.platform()},
            "settings": {"agents": physics.agent_names, "timestep_s": physics.model.opt.timestep,
                         "force_N": args.force, "max_penetration_m": MAX_PENETRATION,
                         "max_linear_speed_m_s": MAX_LINEAR_SPEED,
                         "max_angular_speed_rad_s": MAX_ANGULAR_SPEED,
                         "max_coast_distance_m": MAX_COAST_DISTANCE,
                         "max_settling_time_s": MAX_SETTLING_TIME,
                         "dof_damping": physics.model.dof_damping.tolist(),
                         "dof_frictionloss": physics.model.dof_frictionloss.tolist()},
            "reports": reports,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(f"Report: {args.json}")
    print(f"{'PASS' if passed else 'FAIL'}: {sum(r['passed'] for r in reports)}/{expected} checks")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
