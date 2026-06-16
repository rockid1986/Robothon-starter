from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import imageio.v3 as iio
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised by users before install
    raise SystemExit(
        "Missing dependency. Install from the repository root with:\n"
        "  python -m pip install -r requirements.txt\n\n"
        f"Original error: {exc}"
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "scene.xml"
OUTPUT_DIR = PROJECT_DIR / "outputs"

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_BASE_X = {
    "thumb": -0.18,
    "index": -0.06,
    "middle": 0.06,
    "ring": 0.18,
    "pinky": 0.24,
}
BUTTON_X = {
    "a": -0.18,
    "b": -0.06,
    "c": 0.06,
    "d": 0.18,
}
BUTTON_SENSORS = {
    "a": "button_a_depth",
    "b": "button_b_depth",
    "c": "button_c_depth",
    "d": "button_d_depth",
}
DIAL_SENSOR = "dial_angle"
PRESS_THRESHOLD = 0.005


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if value <= edge0:
        return 0.0
    if value >= edge1:
        return 1.0
    x = (value - edge0) / (edge1 - edge0)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, alpha: float) -> float:
    return float(a + (b - a) * alpha)


def round_list(values: Iterable[float], digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


def actuator_name(finger: str, axis: str) -> str:
    return f"{finger}_{axis}_act"


def home_pose() -> dict[str, float]:
    pose: dict[str, float] = {}
    for finger in FINGERS:
        pose[actuator_name(finger, "x")] = 0.0
        pose[actuator_name(finger, "y")] = 0.035
        pose[actuator_name(finger, "z")] = -0.040
    return pose


def place_finger(
    pose: dict[str, float],
    finger: str,
    *,
    world_x: float,
    y: float,
    z: float,
) -> None:
    pose[actuator_name(finger, "x")] = float(np.clip(world_x - FINGER_BASE_X[finger], -0.075, 0.075))
    pose[actuator_name(finger, "y")] = float(np.clip(y, -0.020, 0.220))
    pose[actuator_name(finger, "z")] = float(np.clip(z, -0.155, 0.020))


def press_pose(finger: str, button: str, *, z: float = -0.146) -> dict[str, float]:
    pose = home_pose()
    place_finger(pose, finger, world_x=BUTTON_X[button], y=0.120, z=z)
    return pose


def chord_pose(*pairs: tuple[str, str], z: float = -0.146) -> dict[str, float]:
    pose = home_pose()
    for finger, button in pairs:
        place_finger(pose, finger, world_x=BUTTON_X[button], y=0.120, z=z)
    return pose


def dial_pose(y: float, *, z: float = -0.125) -> dict[str, float]:
    pose = home_pose()
    # Push the knob handle near the positive x side through a tangential sweep.
    place_finger(pose, "pinky", world_x=0.248, y=y, z=z)
    return pose


def sensor_slice(model: mujoco.MjModel, sensor_name: str) -> slice:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
        raise KeyError(f"Missing sensor: {sensor_name}")
    adr = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])
    return slice(adr, adr + dim)


def joint_value(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(f"Missing joint: {joint_name}")
    return float(data.qpos[model.jnt_qposadr[joint_id]])


@dataclass(frozen=True)
class Waypoint:
    time_s: float
    pose: dict[str, float]
    label: str


class ScriptedPolicy:
    """Deterministic multi-finger task script used by demo and data collection."""

    def __init__(self, jitter: float = 0.0, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self.waypoints = [
            Waypoint(0.0, home_pose(), "home"),
            Waypoint(0.7, chord_pose(("thumb", "a"), ("index", "b"), ("middle", "c"), ("ring", "d"), z=-0.075), "hover over keys"),
            Waypoint(1.1, press_pose("thumb", "a"), "press red key"),
            Waypoint(1.7, press_pose("thumb", "a"), "hold red key"),
            Waypoint(2.1, home_pose(), "release red key"),
            Waypoint(2.5, press_pose("index", "b"), "press blue key"),
            Waypoint(3.1, press_pose("index", "b"), "hold blue key"),
            Waypoint(3.5, home_pose(), "release blue key"),
            Waypoint(3.9, chord_pose(("middle", "c"), ("ring", "d"), z=-0.138), "prepare green/gold chord"),
            Waypoint(4.5, chord_pose(("middle", "c"), ("ring", "d"), z=-0.151), "press green/gold chord"),
            Waypoint(5.0, home_pose(), "release chord"),
            Waypoint(5.4, dial_pose(0.010, z=-0.118), "touch dial handle"),
            Waypoint(6.2, dial_pose(0.205, z=-0.124), "sweep dial"),
            Waypoint(6.9, dial_pose(0.205, z=-0.118), "hold dial"),
            Waypoint(7.5, home_pose(), "return home"),
        ]
        if jitter:
            self._apply_command_jitter(rng, jitter)

    def _apply_command_jitter(self, rng: np.random.Generator, jitter: float) -> None:
        # Small domain randomization for data collection. It perturbs commanded
        # fingertip x positions while leaving the fixed MuJoCo scene unchanged.
        for waypoint in self.waypoints:
            for finger in FINGERS:
                y_key = actuator_name(finger, "y")
                x_key = actuator_name(finger, "x")
                if waypoint.pose[y_key] > 0.08:
                    waypoint.pose[x_key] = float(
                        np.clip(waypoint.pose[x_key] + rng.uniform(-jitter, jitter), -0.075, 0.075)
                    )

    def target(self, time_s: float) -> dict[str, float]:
        previous = self.waypoints[0]
        for waypoint in self.waypoints[1:]:
            if time_s <= waypoint.time_s:
                alpha = smoothstep(previous.time_s, waypoint.time_s, time_s)
                return {
                    name: lerp(previous.pose[name], waypoint.pose[name], alpha)
                    for name in previous.pose
                }
            previous = waypoint
        return dict(self.waypoints[-1].pose)

    def label(self, time_s: float) -> str:
        current = self.waypoints[0].label
        for waypoint in self.waypoints:
            if time_s >= waypoint.time_s:
                current = waypoint.label
            else:
                break
        return current


def actuator_ids(model: mujoco.MjModel) -> dict[str, int]:
    ids: dict[str, int] = {}
    for finger in FINGERS:
        for axis in ("x", "y", "z"):
            name = actuator_name(finger, axis)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise KeyError(f"Missing actuator: {name}")
            ids[name] = actuator_id
    return ids


def apply_controls(model: mujoco.MjModel, data: mujoco.MjData, controls: dict[str, float]) -> None:
    ids = actuator_ids(model)
    for name, value in controls.items():
        data.ctrl[ids[name]] = value


def read_sensors(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float | list[float]]:
    values: dict[str, float | list[float]] = {}
    for button, sensor_name in BUTTON_SENSORS.items():
        values[f"button_{button}_depth"] = round(float(data.sensordata[sensor_slice(model, sensor_name)][0]), 5)
    values["dial_angle"] = round(float(data.sensordata[sensor_slice(model, DIAL_SENSOR)][0]), 5)
    for button in BUTTON_SENSORS:
        sensor_name = f"button_{button}_touch"
        values[sensor_name] = round(float(data.sensordata[sensor_slice(model, sensor_name)][0]), 5)
    for finger in FINGERS:
        sensor_name = f"{finger}_tip_pos"
        values[sensor_name] = round_list(data.sensordata[sensor_slice(model, sensor_name)], 4)
    return values


def run_episode(
    *,
    model_path: Path = MODEL_PATH,
    duration_s: float = 8.0,
    fps: int = 30,
    width: int = 960,
    height: int = 544,
    video_path: Path | None = None,
    trajectory_path: Path | None = None,
    sample_hz: int = 50,
    seed: int | None = None,
    jitter: float = 0.0,
    camera: str = "overview",
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height) if video_path else None
    policy = ScriptedPolicy(jitter=jitter, seed=seed)
    ids = actuator_ids(model)

    timestep = float(model.opt.timestep)
    total_steps = int(math.ceil(duration_s / timestep))
    render_interval = max(1, int(round(1.0 / (fps * timestep))))
    sample_interval = max(1, int(round(1.0 / (sample_hz * timestep))))

    frames: list[np.ndarray] = []
    samples: list[dict] = []
    max_button_depth = {button: 0.0 for button in BUTTON_SENSORS}
    max_touch = {button: 0.0 for button in BUTTON_SENSORS}
    max_abs_dial = 0.0

    for step in range(total_steps):
        time_s = float(data.time)
        controls = policy.target(time_s)
        for name, value in controls.items():
            data.ctrl[ids[name]] = value
        mujoco.mj_step(model, data)

        sensor_values = read_sensors(model, data)
        for button in BUTTON_SENSORS:
            depth = float(sensor_values[f"button_{button}_depth"])
            max_button_depth[button] = max(max_button_depth[button], depth)
            max_touch[button] = max(max_touch[button], float(sensor_values[f"button_{button}_touch"]))
        max_abs_dial = max(max_abs_dial, abs(float(sensor_values["dial_angle"])))

        if renderer and step % render_interval == 0:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())

        if step % sample_interval == 0:
            samples.append(
                {
                    "time_s": round(time_s, 4),
                    "phase": policy.label(time_s),
                    "controls": {name: round(float(value), 4) for name, value in controls.items()},
                    "sensors": sensor_values,
                    "qpos_head": round_list(data.qpos[: min(12, data.qpos.size)], 5),
                    "qvel_head": round_list(data.qvel[: min(12, data.qvel.size)], 5),
                    "contact_count": int(data.ncon),
                }
            )

    pressed = {button: depth >= PRESS_THRESHOLD for button, depth in max_button_depth.items()}
    score = (
        0.20 * sum(pressed.values())
        + 0.12 * min(1.0, max_abs_dial / 0.35)
        + 0.08 * min(1.0, max(max_touch.values()) / 0.5)
    )
    summary = {
        "project": "Dexterous Data Bench",
        "task": "Five fingertip actuators press a four-key sequence, execute a two-finger chord, sweep a rotary dial, and log control/sensor streams.",
        "model": str(model_path),
        "duration_s": duration_s,
        "fps": fps,
        "sample_hz": sample_hz,
        "seed": seed,
        "pressed_buttons": pressed,
        "max_button_depth_m": {k: round(v, 5) for k, v in max_button_depth.items()},
        "max_touch": {k: round(v, 5) for k, v in max_touch.items()},
        "max_abs_dial_rad": round(max_abs_dial, 5),
        "task_score_proxy": round(score, 4),
        "success": all(pressed.values()),
        "samples": samples,
    }

    if video_path and renderer:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            iio.imwrite(video_path, np.asarray(frames), fps=fps, codec="libx264")
            summary["video"] = str(video_path)
        except Exception as exc:
            fallback = video_path.with_suffix(".gif")
            iio.imwrite(fallback, np.asarray(frames), fps=fps)
            summary["video"] = str(fallback)
            summary["video_fallback_reason"] = str(exc)
        finally:
            renderer.close()

    if trajectory_path:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["trajectory"] = str(trajectory_path)

    return summary
