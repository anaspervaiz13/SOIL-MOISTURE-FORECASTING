from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


FILENAME_PATTERN = re.compile(
    r"^(?P<network>[^_]+)_(?P<subnetwork>[^_]+)_(?P<station>.+?)_"
    r"(?P<variable>sm|ts|ta|p)_(?P<depth_from>-?\d+\.\d+)_(?P<depth_to>-?\d+\.\d+)_"
    r"(?P<sensor>.+)_(?P<sensor_block>\d+)_(?P<replicate>\d+)_(?P<start>\d{8})_(?P<end>\d{8})\.stm$"
)

STATIC_COLUMNS = ["station", "timestamp", "latitude", "longitude", "elevation"]
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_ROOT = WORKSPACE_ROOT / "data" / "original dataset"
PROCESSED_DATA_DIR = WORKSPACE_ROOT / "data" / "processed"
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs" / "ismn"
MERGED_OUTPUT_PATH = PROCESSED_DATA_DIR / "ismn_merged_hourly.csv"
SUMMARY_OUTPUT_PATH = OUTPUTS_DIR / "merge_summary.json"
METADATA_PATH = RAW_DATA_ROOT / "Metadata.json"
ALLOWED_QUALITY_FLAG = "G"
PREFERRED_FEATURE_ORDER = [
    "p_ecotech_rain_gauge",
    "p_ott_pluvio2s_amount",
    "p_ott_pluvio2s_volume",
    "p_vaisala_wxt510",
    "sm_0.05m",
    "sm_0.20m",
    "sm_0.50m",
    "ta_2.00m",
    "ts_0.05m",
    "ts_0.20m",
    "ts_0.50m",
]


def normalize_sensor_name(sensor: str) -> str:
    cleaned = sensor.replace("'", "").strip().lower()
    cleaned = cleaned.replace("-", "_").replace(" ", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def format_depth_label(depth_value: float) -> str:
    return f"{abs(depth_value):.2f}m"


def build_feature_name(variable: str, sensor: str, depth_from: float) -> str:
    if variable == "p":
        return f"p_{normalize_sensor_name(sensor)}"
    return f"{variable}_{format_depth_label(depth_from)}"


def parse_file_metadata(file_path: Path) -> dict[str, object]:
    match = FILENAME_PATTERN.match(file_path.name)
    if not match:
        raise ValueError(f"Unrecognized ISMN filename format: {file_path.name}")

    metadata = match.groupdict()
    depth_from = float(metadata["depth_from"])
    sensor = metadata["sensor"]

    return {
        "network": metadata["network"],
        "subnetwork": metadata["subnetwork"],
        "station": metadata["station"],
        "variable": metadata["variable"],
        "depth_from": depth_from,
        "depth_to": float(metadata["depth_to"]),
        "sensor": sensor,
        "feature_name": build_feature_name(metadata["variable"], sensor, depth_from),
    }


def read_stm_file(file_path: Path, allowed_quality: str = "G") -> pd.DataFrame:
    metadata = parse_file_metadata(file_path)

    frame = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["date", "time", "value", "quality_flag", "provider_flag"],
        engine="python",
    )

    frame = frame.loc[frame["quality_flag"] == allowed_quality].copy()
    frame["timestamp"] = pd.to_datetime(frame["date"] + " " + frame["time"], format="%Y/%m/%d %H:%M")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    with file_path.open("r", encoding="utf-8") as handle:
        header_parts = handle.readline().strip().split()

    frame["station"] = metadata["station"]
    frame["latitude"] = float(header_parts[3])
    frame["longitude"] = float(header_parts[4])
    frame["elevation"] = float(header_parts[5])
    frame["feature_name"] = metadata["feature_name"]

    return frame[
        [
            "station",
            "timestamp",
            "latitude",
            "longitude",
            "elevation",
            "feature_name",
            "value",
            "quality_flag",
            "provider_flag",
        ]
    ].reset_index(drop=True)


def aggregate_replicates(long_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        long_frame.groupby(STATIC_COLUMNS + ["feature_name"], as_index=False)["value"]
        .median()
        .sort_values(["station", "timestamp", "feature_name"])
        .reset_index(drop=True)
    )


def build_merged_hourly(aggregated_frame: pd.DataFrame) -> pd.DataFrame:
    wide_frame = (
        aggregated_frame.pivot_table(
            index=STATIC_COLUMNS,
            columns="feature_name",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["station", "timestamp"])
        .reset_index(drop=True)
    )

    wide_frame.columns.name = None
    return wide_frame


def summarize_merged_dataset(merged_frame: pd.DataFrame) -> dict[str, object]:
    feature_columns = [column for column in merged_frame.columns if column not in STATIC_COLUMNS]
    return {
        "rows": int(len(merged_frame)),
        "columns": int(len(merged_frame.columns)),
        "stations": sorted(merged_frame["station"].dropna().unique().tolist()),
        "rows_per_station": merged_frame["station"].value_counts().sort_index().to_dict(),
        "time_start": None if merged_frame.empty else merged_frame["timestamp"].min().isoformat(),
        "time_end": None if merged_frame.empty else merged_frame["timestamp"].max().isoformat(),
        "missingness": merged_frame[feature_columns].isna().mean().round(6).to_dict(),
    }


def save_summary(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_download_metadata(metadata_path: Path = METADATA_PATH) -> dict[str, object]:
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def collect_stm_files(raw_data_root: Path = RAW_DATA_ROOT) -> list[Path]:
    return sorted(raw_data_root.rglob("*.stm"))


def parse_all_stm_files(stm_files: list[Path], allowed_quality: str = ALLOWED_QUALITY_FLAG) -> pd.DataFrame:
    frames = []
    for index, file_path in enumerate(stm_files, start=1):
        print(f"[{index}/{len(stm_files)}] Reading {file_path.name}")
        frames.append(read_stm_file(file_path, allowed_quality=allowed_quality))
    if not frames:
        raise ValueError("No .stm files were found in the raw data directory.")
    return pd.concat(frames, ignore_index=True)


def order_merged_columns(merged_frame: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [column for column in merged_frame.columns if column not in STATIC_COLUMNS]
    ordered_features = [column for column in PREFERRED_FEATURE_ORDER if column in feature_columns]
    remaining_features = sorted(column for column in feature_columns if column not in ordered_features)
    return merged_frame[STATIC_COLUMNS + ordered_features + remaining_features]


def build_and_save_merged_dataset() -> dict[str, object]:
    metadata = load_download_metadata()
    download_flag = metadata.get("download_info", {}).get("g_flag_only")
    if download_flag is not True:
        print("Download metadata does not enforce G-only filtering; applying local G-only rule for preprocessing.")

    stm_files = collect_stm_files()
    print(f"Found {len(stm_files)} raw .stm files")

    long_frame = parse_all_stm_files(stm_files, allowed_quality=ALLOWED_QUALITY_FLAG)
    print(f"Retained {len(long_frame)} observations with quality flag {ALLOWED_QUALITY_FLAG}")

    aggregated = aggregate_replicates(long_frame)
    print(f"Aggregated to {len(aggregated)} station/timestamp/feature rows after replicate median")

    merged = order_merged_columns(build_merged_hourly(aggregated))

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    merged.to_csv(MERGED_OUTPUT_PATH, index=False)
    summary = summarize_merged_dataset(merged)
    save_summary(summary, SUMMARY_OUTPUT_PATH)

    print(f"Saved merged dataset to {MERGED_OUTPUT_PATH}")
    print(f"Saved summary JSON to {SUMMARY_OUTPUT_PATH}")
    print(f"Merged shape: {merged.shape[0]} rows x {merged.shape[1]} columns")

    return summary


def main() -> None:
    build_and_save_merged_dataset()


if __name__ == "__main__":
    main()
