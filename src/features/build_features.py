from __future__ import annotations

import pandas as pd


TARGET_HINTS = [
    "irrigation",
    "water_requirement",
    "water_required",
    "recommended_water",
    "predicted",
    "target",
    "label",
]


def build_features(df: pd.DataFrame, date_column: str | None = None) -> pd.DataFrame:
    features = df.copy()
    if {"temperature_max", "temperature_min"}.issubset(features.columns):
        features["temperature_range"] = features["temperature_max"] - features["temperature_min"]
    if "rainfall" in features.columns:
        features["rainfall_indicator"] = (features["rainfall"].fillna(0) > 0).astype(int)
        if date_column and date_column in features.columns:
            ordered = features.sort_values(date_column)
            features.loc[ordered.index, "rainfall_3d"] = ordered["rainfall"].rolling(3, min_periods=1).sum().shift(1)
            features.loc[ordered.index, "rainfall_7d"] = ordered["rainfall"].rolling(7, min_periods=1).sum().shift(1)
    if "temperature_mean" in features.columns and date_column and date_column in features.columns:
        ordered = features.sort_values(date_column)
        features.loc[ordered.index, "temperature_mean_3d"] = ordered["temperature_mean"].rolling(3, min_periods=1).mean().shift(1)
        features.loc[ordered.index, "temperature_mean_7d"] = ordered["temperature_mean"].rolling(7, min_periods=1).mean().shift(1)
    if "soil_moisture" in features.columns and "temperature_mean" in features.columns:
        features["moisture_temperature_interaction"] = features["soil_moisture"] * features["temperature_mean"]
    if "soil_humidity" in features.columns and "temperature_mean" in features.columns:
        features["soil_humidity_temperature_interaction"] = features["soil_humidity"] * features["temperature_mean"]
    if date_column and date_column in features.columns:
        parsed = pd.to_datetime(features[date_column], errors="coerce")
        features["month"] = parsed.dt.month
        features["day_of_year"] = parsed.dt.dayofyear
    return features


def select_feature_columns(
    df: pd.DataFrame,
    regression_target: str,
    date_column: str | None,
) -> list[str]:
    blocked = {regression_target}
    if date_column:
        blocked.add(date_column)
    blocked.add("date")

    selected: list[str] = []
    for column in df.columns:
        lower = column.lower()
        if column in blocked:
            continue
        if any(hint in lower for hint in TARGET_HINTS):
            continue
        selected.append(column)
    return selected


def assert_no_target_leakage(
    feature_columns: list[str],
    regression_target: str,
) -> None:
    blocked = {regression_target.lower()}
    for column in feature_columns:
        lower = column.lower()
        if lower in blocked:
            raise ValueError(f"Target leakage detected: {column} is selected as a feature.")
        if "irrigation" in lower or "predicted" in lower or lower in {"target", "label"}:
            raise ValueError(f"Potential target leakage detected in feature column: {column}")
