from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def build_fast_tuning_runs() -> list[dict[str, object]]:
    return [
        {"name": "lb120_base", "lookback": 120, "batch_size": 256, "dropout": 0.2, "learning_rate": 0.001, "lstm_units": [64, 32]},
        {"name": "lb168_bs128", "lookback": 168, "batch_size": 128, "dropout": 0.2, "learning_rate": 0.001, "lstm_units": [64, 32]},
        {"name": "lb168_do10", "lookback": 168, "batch_size": 256, "dropout": 0.1, "learning_rate": 0.001, "lstm_units": [64, 32]},
        {"name": "lb240_base", "lookback": 240, "batch_size": 256, "dropout": 0.2, "learning_rate": 0.001, "lstm_units": [64, 32]},
        {"name": "lb120_lr5e4", "lookback": 120, "batch_size": 256, "dropout": 0.2, "learning_rate": 0.0005, "lstm_units": [96, 48]},
    ]


def format_tuning_output_tag(run: dict[str, object]) -> str:
    return f"tune_{run['name']}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small separate LSTM tuning sweep.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(run: dict[str, object], args: argparse.Namespace) -> list[str]:
    train_script = WORKSPACE_ROOT / "src" / "train_lstm_48h.py"
    return [
        sys.executable,
        str(train_script),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--seeds",
        args.seeds,
        "--threads",
        str(args.threads),
        "--lookback",
        str(run["lookback"]),
        "--batch-size",
        str(run["batch_size"]),
        "--dropout",
        str(run["dropout"]),
        "--learning-rate",
        str(run["learning_rate"]),
        "--lstm-units",
        ",".join(str(unit) for unit in run["lstm_units"]),
        "--output-tag",
        format_tuning_output_tag(run),
    ]


def main() -> None:
    args = parse_args()
    runs = build_fast_tuning_runs()
    for run in runs:
        command = build_command(run, args)
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=WORKSPACE_ROOT)


if __name__ == "__main__":
    main()
