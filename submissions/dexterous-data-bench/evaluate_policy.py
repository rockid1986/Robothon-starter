from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dexterous_data_bench import MODEL_PATH, OUTPUT_DIR, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the learned Dexterous Data Bench policy across repeated rollouts."
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "learned_policy_eval.json")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--duration", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = []
    for episode in range(args.episodes):
        summary = run_episode(
            model_path=args.model,
            duration_s=args.duration,
            seed=episode,
            controller="learned",
            overlay=False,
        )
        results.append(
            {
                "episode": episode,
                "success": summary["success"],
                "task_score_proxy": summary["task_score_proxy"],
                "pressed_buttons": summary["pressed_buttons"],
                "max_abs_dial_rad": summary["max_abs_dial_rad"],
                "max_button_depth_m": summary["max_button_depth_m"],
            }
        )

    success_count = sum(1 for row in results if row["success"])
    report = {
        "project": "Dexterous Data Bench",
        "controller": "learned imitation + sensor feedback",
        "episodes": args.episodes,
        "success_count": success_count,
        "success_rate": round(success_count / max(1, args.episodes), 4),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if success_count == args.episodes else 2


if __name__ == "__main__":
    sys.exit(main())
