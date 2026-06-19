from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dexterous_data_bench import MODEL_PATH, OUTPUT_DIR, PROJECT_DIR, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Dexterous Data Bench MuJoCo demo and trajectory log."
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "demo.mp4")
    parser.add_argument("--trajectory", type=Path, default=OUTPUT_DIR / "dexterous_data_bench_trajectory.json")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--sample-hz", type=int, default=50)
    parser.add_argument("--camera", choices=("overview", "topdown"), default="overview")
    parser.add_argument("--controller", choices=("learned", "scripted"), default="learned")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_episode(
        model_path=args.model,
        video_path=args.output,
        trajectory_path=args.trajectory,
        duration_s=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        sample_hz=args.sample_hz,
        camera=args.camera,
        controller=args.controller,
    )
    printable = {key: value for key, value in summary.items() if key != "samples"}
    printable["sample_count"] = len(summary["samples"])
    print(json.dumps(printable, indent=2))
    return 0 if summary["success"] else 2


if __name__ == "__main__":
    sys.exit(main())
