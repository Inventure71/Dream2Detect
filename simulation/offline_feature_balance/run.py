from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dream2detect.simulation import run_feature_balance_simulation, write_feature_balance_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline feature-balance simulation without calling any model API.")
    parser.add_argument("--total", type=int, default=1000, help="Total number of simulated prompts")
    parser.add_argument("--seed", type=int, default=7, help="Sampler seed for reproducible output")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "simulation" / "offline_feature_balance" / "outputs"),
        help="Directory where the CSVs and HTML report will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_feature_balance_simulation(total_prompts=args.total, seed=args.seed)
    outputs = write_feature_balance_outputs(
        result=result,
        output_dir=Path(args.output_dir),
    )
    print(f"Simulated {result.total_assignments} prompt assignments without OpenAI calls.")
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
