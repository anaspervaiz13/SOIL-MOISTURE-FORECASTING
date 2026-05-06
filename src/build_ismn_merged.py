from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "ismn" / "TERENO"
OUTPUT_DIR = ROOT / "data" / "processed"
SUMMARY_DIR = ROOT / "outputs" / "ismn"
LOG_PATH = ROOT / "logs" / "ismn_processing_log.md"

KEEP_VARIABLES = {"sm", "ts", "ta", "p"}


def parse_header(header_line: str) -> dict[str, object]:
    parts = header_line.strip().split()
    sensor = header_line.split("'")
    sensor_name = sensor[1] if len(sensor) >= 3 else "unknown"
    return {
        "network": parts[0],
        "network_abbr": parts[1],
        "station": parts[2],
        "latitude": float(parts[3]),
        "longitude": float(parts[4]),
        "elevation": float(parts[5]),
        "depth_from_header": float(parts[6]),
        "depth_to_header": float(parts[7]),
        "sensor_name": sensor_name,
    }


def parse_filename(path: Path) -> dict[str, object]:
    parts = path.stem.split("_")
    return {
        "network": parts[0],
        "network_abbr": parts[1],
        "station": parts[2],
        "variable": parts[3],
        "depth_from": float(parts[4]),
        "depth_to": float(parts[5]),
        "sensor_model": parts[6],
        "sensor_group": parts[7],
        "sensor_replicate": parts[8],
        "date_start": parts[9],
        "date_end": parts[10],
    }


def read_stm_file(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header_line = handle.readline().strip()

    if not header_line:
        return pd.DataFrame()

    header_meta = parse_header(header_line)
    file_meta = parse_filename(path)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["date", "time", "value", "quality_flag", "provider_flag"],
        usecols=[0, 1, 2, 3, 4],
        na_values=["nan", "NaN"],
        keep_default_na=True,
        engine="python",
    )
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%Y/%m/%d %H:%M",
        errors="coerce",
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float32")
    df = df.drop(columns=["date", "time"])

    for key, value in {**header_meta, **file_meta}.items():
        df[key] = value
    df["source_file"] = path.name
    df = df.dropna(subset=["timestamp", "value"])
    df = df[df["quality_flag"] == "G"].copy()
    return df


def column_name(variable: str, depth_from: float, sensor_name: str) -> str:
    if variable in {"sm", "ts"}:
        return f"{variable}_{depth_from:.2f}m"
    if variable == "ta":
        return f"{variable}_{abs(depth_from):.2f}m"
    safe_sensor = sensor_name.lower().replace(" ", "_").replace("-", "_")
    return f"p_{safe_sensor}"


def append_log(text: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(text)


def build_feature_name(df: pd.DataFrame) -> pd.Series:
    depth_label = df["depth_from"].map(lambda depth: f"{depth:.2f}m")
    ta_label = df["depth_from"].abs().map(lambda depth: f"{depth:.2f}m")
    sensor_label = (
        df["sensor_name"]
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    feature_name = pd.Series(index=df.index, dtype="object")
    feature_name[df["variable"].isin(["sm", "ts"])] = (
        df.loc[df["variable"].isin(["sm", "ts"]), "variable"] + "_" + depth_label[df["variable"].isin(["sm", "ts"])]
    )
    feature_name[df["variable"] == "ta"] = "ta_" + ta_label[df["variable"] == "ta"]
    feature_name[df["variable"] == "p"] = "p_" + sensor_label[df["variable"] == "p"]
    return feature_name


def load_all_files(paths: list[Path], workers: int) -> pd.DataFrame:
    frames = []
    total = len(paths)
    done = 0
    started = time.time()

    print(f"Parsing {total} files with {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(read_stm_file, path): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            done += 1
            try:
                df = future.result()
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                print(f"[{done}/{total}] failed: {path.name} -> {exc}")
                raise

            if done == total or done % 5 == 0:
                elapsed = time.time() - started
                print(f"[{done}/{total}] parsed | {elapsed:.1f}s")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    paths = []
    for path in sorted(RAW_DIR.rglob("*.stm")):
        if parse_filename(path)["variable"] in KEEP_VARIABLES:
            paths.append(path)

    workers = min(len(paths), os.cpu_count() or 1)
    raw_df = load_all_files(paths, workers)
    if raw_df.empty:
        raise RuntimeError("No rows parsed from ISMN files.")

    append_log("\n## Step 7: Parser executed\n\n")
    append_log(f"- Parsed `{raw_df['source_file'].nunique()}` raw `.stm` files.\n")
    append_log(f"- Parsed `{len(raw_df):,}` timestamped observations after in-parser QC filtering.\n")
    append_log(f"- Parser workers used: `{workers}`.\n")

    append_log("\n## Step 8: Quality filtering applied\n\n")
    append_log(f"- Retained `{len(raw_df):,}` observations with quality flag `G`.\n")
    append_log("- Non-`G` observations were dropped inside per-file parsing for speed.\n")

    print("Building feature names...")
    raw_df["feature_name"] = build_feature_name(raw_df)

    group_cols = [
        "station",
        "timestamp",
        "feature_name",
        "variable",
        "depth_from",
        "latitude",
        "longitude",
        "elevation",
    ]
    print("Aggregating replicate sensors...")
    agg_df = (
        raw_df.groupby(group_cols, dropna=False, sort=False)
        .agg(value=("value", "median"), sensor_count=("source_file", "nunique"))
        .reset_index()
    )

    append_log("\n## Step 9: Replicate aggregation applied\n\n")
    append_log("- Aggregated replicate sensors by median at each station/timestamp/variable/depth combination.\n")
    append_log(f"- Produced `{len(agg_df):,}` aggregated observations.\n")

    print("Pivoting merged table...")
    wide_df = (
        agg_df.pivot_table(
            index=["station", "timestamp", "latitude", "longitude", "elevation"],
            columns="feature_name",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["station", "timestamp"])
    )
    wide_df.columns.name = None

    output_path = OUTPUT_DIR / "ismn_merged_hourly.csv"
    wide_df.to_csv(output_path, index=False)

    summary = {
        "raw_files_parsed": int(raw_df["source_file"].nunique()),
        "observations_after_qc": int(len(raw_df)),
        "aggregated_observations": int(len(agg_df)),
        "merged_rows": int(len(wide_df)),
        "merged_columns": int(len(wide_df.columns)),
        "stations": sorted(wide_df["station"].dropna().unique().tolist()),
        "columns": list(wide_df.columns),
        "time_range": {
            "start": str(wide_df["timestamp"].min()),
            "end": str(wide_df["timestamp"].max()),
        },
    }

    with (SUMMARY_DIR / "merge_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    append_log("\n## Step 10: Merged output saved\n\n")
    append_log(f"- Saved merged hourly CSV to `{output_path.as_posix()}`.\n")
    append_log(f"- Saved summary JSON to `{(SUMMARY_DIR / 'merge_summary.json').as_posix()}`.\n")
    append_log(f"- Final merged shape: `{wide_df.shape[0]:,}` rows x `{wide_df.shape[1]}` columns.\n")

    print(f"Saved merged CSV: {output_path}")
    print(f"Merged shape: {wide_df.shape}")
    print(f"Columns: {list(wide_df.columns)}")


if __name__ == "__main__":
    main()
