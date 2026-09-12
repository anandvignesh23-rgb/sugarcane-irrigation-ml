from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config, require_config, resolve_path
from src.data.acquire_sugarcane import describe_primary_dataset, ensure_sidss_sugarcane_dataset
from src.data.load_sugarcane import inspect_schema, load_sugarcane_dataset
from src.data.merge_data import infer_date_column, merge_sugarcane_with_weather
from src.data.nasa_power import fetch_nasa_power
from src.data.preprocess import clean_and_prepare_targets
from src.features.build_features import assert_no_target_leakage, build_features, select_feature_columns
from src.models.common import make_holdout_splits
from src.models.train_regressors import train_regressors
from src.visualization.plots import (
    plot_actual_vs_predicted,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_feature_importance,
    plot_missing_values,
    plot_model_comparisons,
    plot_residuals,
    plot_soil_moisture_vs_target,
    plot_weather_over_time,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def infer_nasa_dates(df: pd.DataFrame, date_column: str | None, config: dict) -> tuple[str, str, dict[str, str]]:
    configured = config["nasa_power"]
    start = configured.get("start_date")
    end = configured.get("end_date")
    if start and end:
        return str(start), str(end), {
            "date_source": "config",
            "date_range_rationale": str(configured.get("date_range_rationale", "")),
        }
    if date_column:
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if not dates.empty:
            return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d"), {
                "date_source": f"crop_dataset:{date_column}",
                "date_range_rationale": "NASA POWER range inferred from crop observation dates.",
            }
    return (
        require_config(start, "nasa_power.start_date"),
        require_config(end, "nasa_power.end_date"),
        {"date_source": "missing"},
    )


def main() -> None:
    cfg = load_config()
    metrics_dir = resolve_path("reports/metrics")
    figures_dir = resolve_path("reports/figures")
    regressor_dir = resolve_path("models/regressors")

    sugarcane_path = ensure_sidss_sugarcane_dataset(
        resolve_path(cfg["data"]["sugarcane_path"]),
        cfg["data"]["primary_dataset"],
    )
    sugarcane = load_sugarcane_dataset(sugarcane_path)
    schema_report = inspect_schema(sugarcane)
    schema_report["primary_dataset"] = cfg["data"]["primary_dataset"]
    print(f"Primary dataset: {describe_primary_dataset(cfg['data']['primary_dataset'])}")
    date_column = infer_date_column(sugarcane, cfg.get("schema", {}).get("date_column"))
    start_date, end_date, nasa_assumptions = infer_nasa_dates(sugarcane, date_column, cfg)
    latitude = float(require_config(cfg["nasa_power"].get("latitude"), "nasa_power.latitude"))
    longitude = float(require_config(cfg["nasa_power"].get("longitude"), "nasa_power.longitude"))
    nasa_assumptions["latitude"] = latitude
    nasa_assumptions["longitude"] = longitude
    nasa_assumptions["location_rationale"] = str(cfg["nasa_power"].get("location_rationale", ""))
    weather = fetch_nasa_power(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        parameters=cfg["nasa_power"].get("parameters", []),
        cache_path=resolve_path(cfg["data"]["nasa_cache_path"]),
    )

    merged, join_report = merge_sugarcane_with_weather(sugarcane, weather, date_column)
    prepared = clean_and_prepare_targets(
        merged,
        regression_target=cfg["targets"]["regression_target"],
    )
    featured = build_features(prepared.frame, date_column)
    feature_columns = select_feature_columns(
        featured,
        prepared.regression_target,
        date_column,
    )
    assert_no_target_leakage(feature_columns, prepared.regression_target)
    X = featured[feature_columns]
    y_reg = featured[prepared.regression_target]
    temporal_order = pd.to_datetime(featured[date_column], errors="coerce") if date_column else None

    Path(resolve_path(cfg["data"]["processed_path"])).parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(resolve_path(cfg["data"]["processed_path"]), index=False)

    preprocessing_report = {**prepared.report, **join_report, "nasa_power_assumptions": nasa_assumptions}
    write_json(metrics_dir / "schema_report.json", schema_report)
    write_json(metrics_dir / "preprocessing_report.json", preprocessing_report)

    plot_missing_values(featured, figures_dir)
    plot_feature_distributions(featured, figures_dir)
    plot_correlation_heatmap(featured, figures_dir)
    plot_weather_over_time(featured, date_column, figures_dir)
    plot_soil_moisture_vs_target(featured, prepared.regression_target, figures_dir)

    train_cfg = cfg["training"]
    reg_splits = make_holdout_splits(
        X,
        y_reg,
        test_size=float(train_cfg["test_size"]),
        validation_size=float(train_cfg["validation_size"]),
        random_seed=int(cfg["random_seed"]),
        temporal_order=temporal_order,
    )
    reg_results, best_regressor = train_regressors(
        X,
        y_reg,
        reg_splits,
        random_seed=int(cfg["random_seed"]),
        cv_splits=int(train_cfg["cv_splits"]),
        n_iter=int(train_cfg["random_search_iter"]),
        output_dir=regressor_dir,
        feature_columns=feature_columns,
        water_unit=cfg["targets"].get("water_unit"),
        temporal_cv=temporal_order is not None,
    )
    best_reg_artifact = joblib.load(regressor_dir / "best_regressor.joblib")
    y_pred = best_reg_artifact["pipeline"].predict(reg_splits.X_test)
    plot_actual_vs_predicted(reg_splits.y_test, y_pred, figures_dir)
    plot_residuals(reg_splits.y_test, y_pred, figures_dir)
    top_reg_features = plot_feature_importance(best_reg_artifact, figures_dir, "gradient_or_best_regressor_feature_importance.png")

    regressor_rows = [
        {"model": name, **result["test_metrics"]}
        for name, result in reg_results.items()
    ]
    plot_model_comparisons(regressor_rows, figures_dir)

    write_json(metrics_dir / "regressor_results.json", reg_results)
    pd.DataFrame(regressor_rows).to_csv(metrics_dir / "regressor_comparison.csv", index=False)
    write_json(
        metrics_dir / "explainability_summary.json",
        {
            "top_regressor_features": top_reg_features,
            "interpretation_note": (
                "Feature importance ranks predictive contribution for this fitted model; "
                "it should not be interpreted as causal evidence."
            ),
        },
    )

    print(f"Best regressor: {best_regressor}")
    print(f"Metrics saved to: {metrics_dir}")
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
