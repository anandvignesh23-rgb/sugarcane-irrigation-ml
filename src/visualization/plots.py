from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_missing_values(df: pd.DataFrame, output_dir: str | Path) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing.values, y=missing.index)
    plt.title("Missing Values by Column")
    plt.xlabel("Missing count")
    _save(Path(output_dir) / "missing_values.png")


def plot_feature_distributions(df: pd.DataFrame, output_dir: str | Path, max_columns: int = 12) -> None:
    numeric = df.select_dtypes(include="number").columns[:max_columns]
    if len(numeric) == 0:
        return
    axes = df[numeric].hist(figsize=(12, 8), bins=25)
    for row in axes:
        for ax in row:
            ax.tick_params(labelsize=8)
    _save(Path(output_dir) / "feature_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str | Path) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return
    plt.figure(figsize=(11, 8))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0)
    plt.title("Numeric Feature Correlation")
    _save(Path(output_dir) / "correlation_heatmap.png")


def plot_weather_over_time(df: pd.DataFrame, date_column: str | None, output_dir: str | Path) -> None:
    if not date_column or date_column not in df.columns:
        return
    plot_df = df.copy()
    plot_df[date_column] = pd.to_datetime(plot_df[date_column], errors="coerce")
    plot_df = plot_df.sort_values(date_column)
    if "rainfall" in plot_df.columns:
        plt.figure(figsize=(11, 4))
        plt.plot(plot_df[date_column], plot_df["rainfall"])
        plt.title("Rainfall Over Time")
        plt.xlabel("Date")
        plt.ylabel("Rainfall")
        _save(Path(output_dir) / "rainfall_over_time.png")
    if "temperature_mean" in plot_df.columns:
        plt.figure(figsize=(11, 4))
        plt.plot(plot_df[date_column], plot_df["temperature_mean"], label="Mean")
        if "temperature_min" in plot_df.columns:
            plt.plot(plot_df[date_column], plot_df["temperature_min"], alpha=0.6, label="Min")
        if "temperature_max" in plot_df.columns:
            plt.plot(plot_df[date_column], plot_df["temperature_max"], alpha=0.6, label="Max")
        plt.title("Temperature Over Time")
        plt.xlabel("Date")
        plt.ylabel("Temperature")
        plt.legend()
        _save(Path(output_dir) / "temperature_over_time.png")


def plot_soil_moisture_vs_target(df: pd.DataFrame, regression_target: str, output_dir: str | Path) -> None:
    moisture = "soil_moisture" if "soil_moisture" in df.columns else "soil_humidity" if "soil_humidity" in df.columns else None
    if not moisture:
        return
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=moisture, y=regression_target)
    plt.title("Soil Moisture vs Irrigation Requirement")
    _save(Path(output_dir) / "soil_moisture_vs_irrigation.png")


def plot_actual_vs_predicted(y_true: pd.Series, y_pred, output_dir: str | Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Irrigation Quantity")
    _save(Path(output_dir) / "actual_vs_predicted.png")


def plot_residuals(y_true: pd.Series, y_pred, output_dir: str | Path) -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    _save(Path(output_dir) / "residual_plot.png")


def get_feature_names(pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []


def plot_feature_importance(artifact: dict[str, Any], output_dir: str | Path, filename: str) -> list[dict[str, Any]]:
    model = artifact["pipeline"].named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return []
    names = get_feature_names(artifact["pipeline"])
    importances = model.feature_importances_
    if len(names) != len(importances):
        names = [f"feature_{i}" for i in range(len(importances))]
    table = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(10)
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=table, x="importance", y="feature")
    plt.title(filename.replace("_", " ").replace(".png", "").title())
    _save(Path(output_dir) / filename)
    return table.to_dict(orient="records")


def plot_model_comparisons(regressor_rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    if regressor_rows:
        reg = pd.DataFrame(regressor_rows)
        plt.figure(figsize=(9, 5))
        sns.barplot(data=reg, x="model", y="rmse")
        plt.xticks(rotation=25, ha="right")
        plt.title("Regressor RMSE Comparison")
        _save(Path(output_dir) / "regressor_comparison.png")
