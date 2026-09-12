from __future__ import annotations

import pandas as pd

from src.data.load_sugarcane import find_first_existing_column

DATE_CANDIDATES = ["date", "datetime", "timestamp", "sample_date", "observation_date"]
LAT_CANDIDATES = ["latitude", "lat"]
LON_CANDIDATES = ["longitude", "lon", "lng"]


def infer_date_column(df: pd.DataFrame, configured: str | None = None) -> str | None:
    if configured and configured in df.columns:
        return configured
    return find_first_existing_column(df, DATE_CANDIDATES)


def merge_sugarcane_with_weather(
    sugarcane: pd.DataFrame,
    weather: pd.DataFrame,
    date_column: str | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    report: dict[str, object] = {"join_strategy": "", "join_success_rate": None}
    crop = sugarcane.copy()
    met = weather.copy()
    met["date"] = pd.to_datetime(met["date"]).dt.normalize()

    if date_column:
        crop[date_column] = pd.to_datetime(crop[date_column], errors="coerce").dt.normalize()
        before = len(crop)
        merged = crop.merge(met, left_on=date_column, right_on="date", how="left", suffixes=("", "_nasa"))
        weather_cols = [column for column in met.columns if column != "date"]
        matched = merged[weather_cols].notna().any(axis=1).sum() if weather_cols else 0
        report["join_strategy"] = f"daily_date_join:{date_column}"
        report["join_success_rate"] = float(matched / before) if before else 0.0
        return merged, report

    weather_summary = met.drop(columns=["date"]).mean(numeric_only=True).to_frame().T
    for column in weather_summary.columns:
        crop[column] = weather_summary.iloc[0][column]
    report["join_strategy"] = "configured_period_weather_mean_no_crop_date"
    report["join_success_rate"] = 1.0 if len(weather_summary.columns) else 0.0
    report["limitation"] = (
        "The sugarcane dataset has no usable date column. NASA POWER daily data was "
        "aggregated over the configured period and joined to every row, so row-level "
        "weather timing effects are not identifiable."
    )
    return crop, report
