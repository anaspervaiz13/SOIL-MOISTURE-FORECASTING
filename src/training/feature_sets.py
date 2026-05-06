from __future__ import annotations

from fnmatch import fnmatch

from .config import BASE_DYNAMIC_COLUMNS, CALENDAR_COLUMNS


FEATURE_GROUPS = {
    "current_only": {
        "direct_columns": BASE_DYNAMIC_COLUMNS + CALENDAR_COLUMNS,
        "patterns": [],
    },
    "baseline_limited": {
        "direct_columns": BASE_DYNAMIC_COLUMNS + CALENDAR_COLUMNS,
        "patterns": [
            "*_lag_1h",
            "*_lag_24h",
            "sm_0.05m_roll_mean_24h",
            "precipitation_roll_sum_24h",
        ],
    },
    "short_lags": {
        "direct_columns": BASE_DYNAMIC_COLUMNS + CALENDAR_COLUMNS,
        "patterns": [
            "*_lag_1h",
            "*_lag_3h",
            "*_lag_6h",
            "*_lag_12h",
            "*_lag_24h",
            "*_roll_*_24h",
        ],
    },
    "short_plus_medium": {
        "direct_columns": BASE_DYNAMIC_COLUMNS + CALENDAR_COLUMNS,
        "patterns": [
            "*_lag_1h",
            "*_lag_3h",
            "*_lag_6h",
            "*_lag_12h",
            "*_lag_24h",
            "*_lag_48h",
            "*_lag_72h",
            "*_roll_*_24h",
            "*_roll_*_48h",
        ],
    },
    "short_plus_weekly": {
        "direct_columns": BASE_DYNAMIC_COLUMNS + CALENDAR_COLUMNS,
        "patterns": [
            "*_lag_1h",
            "*_lag_3h",
            "*_lag_6h",
            "*_lag_12h",
            "*_lag_24h",
            "*_lag_168h",
            "*_roll_*_24h",
            "*_roll_*_168h",
        ],
    },
    "full_multiscale_no_rolling": {
        "direct_columns": BASE_DYNAMIC_COLUMNS + CALENDAR_COLUMNS,
        "patterns": [
            "*_lag_1h",
            "*_lag_3h",
            "*_lag_6h",
            "*_lag_12h",
            "*_lag_24h",
            "*_lag_48h",
            "*_lag_72h",
            "*_lag_168h",
        ],
    },
    "full_multiscale": {
        "direct_columns": BASE_DYNAMIC_COLUMNS + CALENDAR_COLUMNS,
        "patterns": [
            "*_lag_1h",
            "*_lag_3h",
            "*_lag_6h",
            "*_lag_12h",
            "*_lag_24h",
            "*_lag_48h",
            "*_lag_72h",
            "*_lag_168h",
            "*_roll_*_6h",
            "*_roll_*_24h",
            "*_roll_*_48h",
            "*_roll_*_168h",
        ],
    },
}


def get_feature_group_columns(all_columns: list[str], group_name: str) -> list[str]:
    if group_name not in FEATURE_GROUPS:
        raise KeyError(f"Unknown feature group: {group_name}")

    definition = FEATURE_GROUPS[group_name]
    selected = []

    for column in definition["direct_columns"]:
        if column in all_columns and column not in selected:
            selected.append(column)

    for pattern in definition["patterns"]:
        matches = sorted(column for column in all_columns if fnmatch(column, pattern))
        for column in matches:
            if column not in selected:
                selected.append(column)

    return selected
