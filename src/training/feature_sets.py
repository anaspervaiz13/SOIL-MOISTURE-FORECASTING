from __future__ import annotations

from .config import CALENDAR_FEATURES, CORE_BASE_FEATURES, DEPTH_FEATURES, LAG_HOURS, ROLL_WINDOWS, WEATHER_TEMP_FEATURES


def _lag_features(var_names: list[str], lag_hours: list[int]) -> list[str]:
    return [f"{var}_lag_{lag}h" for var in var_names for lag in lag_hours]


def _roll_features() -> list[str]:
    features: list[str] = []
    for window in ROLL_WINDOWS:
        features.append(f"sm_0.05m_roll_mean_{window}h")
        features.append(f"sm_0.05m_roll_std_{window}h")
        features.append(f"precipitation_roll_sum_{window}h")

    for var in ["sm_0.20m", "sm_0.50m", "ta_2.00m", "ts_0.05m", "ts_0.20m", "ts_0.50m"]:
        for window in [24, 48, 168]:
            features.append(f"{var}_roll_mean_{window}h")
    return features


FEATURE_GROUPS = {
    "surface_only": ["sm_0.05m"] + _lag_features(["sm_0.05m"], LAG_HOURS) + CALENDAR_FEATURES,
    "surface_depth": ["sm_0.05m"] + DEPTH_FEATURES + _lag_features(["sm_0.05m", "sm_0.20m", "sm_0.50m"], LAG_HOURS) + CALENDAR_FEATURES,
    "weather_depth_temporal": CORE_BASE_FEATURES + CALENDAR_FEATURES + _lag_features(CORE_BASE_FEATURES, LAG_HOURS) + _roll_features(),
}


def get_feature_group(name: str) -> list[str]:
    if name not in FEATURE_GROUPS:
        valid = ", ".join(sorted(FEATURE_GROUPS))
        raise ValueError(f"Unknown feature group '{name}'. Valid groups: {valid}")
    return FEATURE_GROUPS[name]
