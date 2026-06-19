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
    from PIL import Image, ImageDraw, ImageFont
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
AXES = ("x", "y", "z")
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
EPISODE_DURATION_S = 8.0
CONTROL_LIMITS = {
    "x": (-0.075, 0.075),
    "y": (-0.020, 0.220),
    "z": (-0.155, 0.020),
}


def ordered_actuator_names() -> tuple[str, ...]:
    return tuple(actuator_name(finger, axis) for finger in FINGERS for axis in AXES)


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


class LearnedImitationPolicy:
    """RBF imitation policy trained from expert trajectories at launch time."""

    def __init__(
        self,
        *,
        weights: np.ndarray,
        feature_centers: np.ndarray,
        feature_sigma: float,
        duration_s: float,
    ) -> None:
        self.weights = weights
        self.feature_centers = feature_centers
        self.feature_sigma = feature_sigma
        self.duration_s = duration_s
        self.expert_clock = ScriptedPolicy()

    def features(self, time_s: float, sensors: dict[str, float | list[float]] | None = None) -> np.ndarray:
        x = float(np.clip(time_s / max(self.duration_s, 1e-6), 0.0, 1.0))
        values: list[float] = [1.0, x, x * x, x * x * x]
        for freq in (1.0, 2.0, 3.0, 4.0):
            values.append(math.sin(2.0 * math.pi * freq * x))
            values.append(math.cos(2.0 * math.pi * freq * x))
        rbf = np.exp(-0.5 * ((time_s - self.feature_centers) / self.feature_sigma) ** 2)
        values.extend(float(v) for v in rbf)

        if sensors:
            values.extend(float(sensors[f"button_{button}_depth"]) / 0.026 for button in BUTTON_SENSORS)
            values.extend(min(1.0, float(sensors[f"button_{button}_touch"]) / 6.0) for button in BUTTON_SENSORS)
            values.append(float(sensors["dial_angle"]) / 1.4)
        else:
            values.extend([0.0] * 9)
        return np.asarray(values, dtype=float)

    def target(self, time_s: float, sensors: dict[str, float | list[float]] | None = None) -> dict[str, float]:
        raw = self.features(time_s, sensors) @ self.weights
        controls = {
            name: float(np.clip(value, *CONTROL_LIMITS[name.rsplit("_", 2)[1]]))
            for name, value in zip(ordered_actuator_names(), raw)
        }

        # Closed-loop correction learned policies often need for contact tasks:
        # if contact depth lags during a phase, bias only the active fingertip.
        if sensors:
            corrections = [
                (1.00, 1.95, "thumb", "a"),
                (2.40, 3.25, "index", "b"),
                (4.05, 4.85, "middle", "c"),
                (4.05, 4.85, "ring", "d"),
            ]
            for start, end, finger, button in corrections:
                if start <= time_s <= end:
                    depth = float(sensors[f"button_{button}_depth"])
                    controls[actuator_name(finger, "z")] -= max(0.0, PRESS_THRESHOLD - depth) * 3.5
            if 5.45 <= time_s <= 6.95 and abs(float(sensors["dial_angle"])) < 0.05:
                controls[actuator_name("pinky", "z")] -= 0.006

        for name, value in list(controls.items()):
            axis = name.rsplit("_", 2)[1]
            controls[name] = float(np.clip(value, *CONTROL_LIMITS[axis]))
        return controls

    def label(self, time_s: float) -> str:
        return "learned: " + self.expert_clock.label(time_s)


def train_imitation_policy(
    *,
    duration_s: float = EPISODE_DURATION_S,
    samples_hz: int = 120,
    ridge: float = 1e-5,
) -> LearnedImitationPolicy:
    expert = ScriptedPolicy()
    centers = np.linspace(0.0, duration_s, 32)
    sigma = duration_s / 30.0
    template = LearnedImitationPolicy(
        weights=np.zeros((4 + 8 + len(centers) + 9, len(ordered_actuator_names()))),
        feature_centers=centers,
        feature_sigma=sigma,
        duration_s=duration_s,
    )
    times = np.arange(0.0, duration_s, 1.0 / samples_hz)
    features = np.vstack([template.features(float(time_s), None) for time_s in times])
    targets = np.vstack(
        [
            [expert.target(float(time_s))[name] for name in ordered_actuator_names()]
            for time_s in times
        ]
    )
    regularizer = ridge * np.eye(features.shape[1])
    weights = np.linalg.solve(features.T @ features + regularizer, features.T @ targets)
    return LearnedImitationPolicy(
        weights=weights,
        feature_centers=centers,
        feature_sigma=sigma,
        duration_s=duration_s,
    )


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


def annotate_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    try:
        image = Image.fromarray(frame)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()
        line_h = 16
        pad = 10
        box_h = pad * 2 + line_h * len(lines)
        draw.rectangle((10, 10, image.size[0] - 10, 10 + box_h), fill=(0, 0, 0, 145))
        for idx, line in enumerate(lines):
            draw.text((20, 20 + idx * line_h), line, fill=(235, 245, 255, 255), font=font)
        return np.asarray(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))
    except Exception:
        return frame


def run_episode(
    *,
    model_path: Path = MODEL_PATH,
    duration_s: float = EPISODE_DURATION_S,
    fps: int = 30,
    width: int = 960,
    height: int = 544,
    video_path: Path | None = None,
    trajectory_path: Path | None = None,
    sample_hz: int = 50,
    seed: int | None = None,
    jitter: float = 0.0,
    camera: str = "overview",
    controller: str = "learned",
    overlay: bool = True,
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height) if video_path else None
    if controller == "scripted":
        policy = ScriptedPolicy(jitter=jitter, seed=seed)
        controller_name = "scripted expert"
    elif controller == "learned":
        policy = train_imitation_policy(duration_s=min(duration_s, EPISODE_DURATION_S))
        controller_name = "learned imitation + sensor feedback"
    else:
        raise ValueError(f"Unknown controller: {controller}")
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
        sensor_feedback = read_sensors(model, data)
        controls = policy.target(min(time_s, EPISODE_DURATION_S), sensor_feedback)
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
            frame = renderer.render().copy()
            if overlay:
                frame = annotate_frame(
                    frame,
                    [
                        "Dexterous Data Bench",
                        f"controller: {controller_name}",
                        f"phase: {policy.label(min(time_s, EPISODE_DURATION_S))}",
                        (
                            "depths m: "
                            + ", ".join(f"{b}={float(sensor_values[f'button_{b}_depth']):.3f}" for b in BUTTON_SENSORS)
                            + f" | dial={float(sensor_values['dial_angle']):.3f} rad"
                        ),
                    ],
                )
            frames.append(frame)

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
        "controller": controller_name,
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
