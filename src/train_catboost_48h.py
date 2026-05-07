from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.training.config import DEFAULT_TARGET_COLUMN, OUTPUTS_TRAINING_DIR
from src.training.data import chronological_split, load_prepared_dataset, select_feature_columns
from src.training.evaluation import compute_metrics, save_run_outputs, summarize_run_metrics
from src.training.feature_sets import FEATURE_GROUPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost benchmark models for 48-hour soil moisture forecasting.")
    parser.add_argument(
        "--feature-group",
        default="full_multiscale",
        choices=sorted(FEATURE_GROUPS.keys()),
    )
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--output-tag", default=None)
    return parser.parse_args()


def parse_seeds(seeds_text: str) -> list[int]:
    return [int(item.strip()) for item in seeds_text.split(",") if item.strip()]


def resolve_output_dir_name(feature_group: str, output_tag: str | None) -> str:
    if output_tag:
        return f"{feature_group}__{output_tag}"
    return feature_group


def build_xy(frame: pd.DataFrame, feature_columns: list[str], target_column: str):
    x_values = frame[feature_columns]
    y_values = frame[target_column]
    metadata = frame[["station", "timestamp"]].copy()
    return x_values, y_values, metadata


def train_one_run(args: argparse.Namespace, feature_columns: list[str], seed: int):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise SystemExit("catboost is not installed. Install it before running this script.") from exc

    dataset = load_prepared_dataset()
    train_df, val_df, test_df = chronological_split(
        dataset,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )

    x_train, y_train, _ = build_xy(train_df, feature_columns, DEFAULT_TARGET_COLUMN)
    x_val, y_val, meta_val = build_xy(val_df, feature_columns, DEFAULT_TARGET_COLUMN)
    x_test, y_test, meta_test = build_xy(test_df, feature_columns, DEFAULT_TARGET_COLUMN)

    model = CatBoostRegressor(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
    )
    model.fit(x_train, y_train)

    rows = []
    prediction_frames = []
    for split_name, x_split, y_split, meta_split in [
        ("val", x_val, y_val, meta_val),
        ("test", x_test, y_test, meta_test),
    ]:
        predictions = model.predict(x_split)
        metric_row = {"seed": seed, "split": split_name}
        metric_row.update(compute_metrics(y_split, predictions))
        rows.append(metric_row)

        prediction_frame = meta_split.copy()
        prediction_frame["seed"] = seed
        prediction_frame["split"] = split_name
        prediction_frame["y_true"] = y_split.to_numpy()
        prediction_frame["y_pred"] = predictions
        prediction_frames.append(prediction_frame)

    return rows, prediction_frames


def main() -> Path:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    dataset = load_prepared_dataset()
    feature_columns = select_feature_columns(dataset, args.feature_group, DEFAULT_TARGET_COLUMN)

    run_rows = []
    prediction_frames = []
    for seed in seeds:
        rows, frames = train_one_run(args, feature_columns, seed)
        run_rows.extend(rows)
        prediction_frames.extend(frames)

    run_metrics = pd.DataFrame(run_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_run_metrics(
        run_metrics.loc[run_metrics["split"] == "test", ["seed", "rmse", "mae", "r2"]].rename(columns={"seed": "run_id"})
    )

    output_dir = OUTPUTS_TRAINING_DIR / "catboost" / resolve_output_dir_name(args.feature_group, args.output_tag)
    metadata = {
        "model": "catboost",
        "feature_group": args.feature_group,
        "output_tag": args.output_tag,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "seeds": seeds,
        "target_column": DEFAULT_TARGET_COLUMN,
    }
    save_run_outputs(output_dir, run_metrics, predictions, summary, metadata)
    print(output_dir)
    return output_dir


if __name__ == "__main__":
    main()
