from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dexterous_data_bench import MODEL_PATH, OUTPUT_DIR, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect deterministic and lightly randomized MuJoCo trajectories."
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR / "dataset")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--sample-hz", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jitter", type=float, default=0.006)
    parser.add_argument("--controller", choices=("learned", "scripted"), default="learned")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.outdir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for episode in range(args.episodes):
            trajectory_path = args.outdir / f"episode_{episode:03d}.json"
            summary = run_episode(
                model_path=args.model,
                trajectory_path=trajectory_path,
                duration_s=args.duration,
                sample_hz=args.sample_hz,
                seed=args.seed + episode,
                jitter=args.jitter,
                controller=args.controller,
            )
            row = {
                "episode": episode,
                "trajectory": str(trajectory_path),
                "success": summary["success"],
                "task_score_proxy": summary["task_score_proxy"],
                "pressed_buttons": summary["pressed_buttons"],
                "max_abs_dial_rad": summary["max_abs_dial_rad"],
                "sample_count": len(summary["samples"]),
            }
            manifest.write(json.dumps(row) + "\n")
            print(json.dumps(row, indent=2))

    print(f"Wrote dataset manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
