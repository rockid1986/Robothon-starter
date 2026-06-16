from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from dexterous_data_bench import MODEL_PATH, actuator_ids, home_pose, press_pose, dial_pose


PRESETS = {
    " ": ("home", home_pose()),
    "1": ("press red key", press_pose("thumb", "a")),
    "2": ("press blue key", press_pose("index", "b")),
    "3": ("press green key", press_pose("middle", "c")),
    "4": ("press gold key", press_pose("ring", "d")),
    "d": ("dial start", dial_pose(0.010)),
    "f": ("dial finish", dial_pose(0.205)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyboard teleoperation for Dexterous Data Bench.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    ids = actuator_ids(model)
    active = {"label": "home", "pose": home_pose()}

    def on_key(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if key in PRESETS:
            label, pose = PRESETS[key]
            active["label"] = label
            active["pose"] = pose
            print(f"preset: {label}")

    print("Controls: space=home, 1/2/3/4=press keys, d/f=turn dial, Esc=quit viewer")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            for name, value in active["pose"].items():
                data.ctrl[ids[name]] = value
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(float(model.opt.timestep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
