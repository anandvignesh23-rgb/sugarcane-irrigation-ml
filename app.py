from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
BEST_MODEL_PATH = ROOT / "models" / "regressors" / "best_regressor.joblib"
METRICS_PATH = ROOT / "reports" / "metrics" / "regressor_comparison.csv"

NUMERIC_FEATURES = [
    "stomataopenclose",
    "nitrogen",
    "chloropyll",
    "phosphorus",
    "leafhumidity",
    "leafevaporation",
    "evapotranspiration",
    "soilph",
    "soilhumidity",
    "microclimatictemp",
    "microclimaticwindspeed",
    "yield_kg_per_plant",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "relative_humidity",
    "rainfall",
    "wind_speed",
    "solar_radiation",
    "surface_pressure",
    "temperature_range",
    "rainfall_indicator",
]

DEFAULT_SAMPLE = {
    "inputleaf": "leaf_1",
    "stomataopenclose": 0.437086107,
    "nitrogen": 2.557245355,
    "chloropyll": 46.90811977,
    "phosphorus": 0.439719715,
    "leafhumidity": 82.68899339,
    "leafevaporation": 1.585756265,
    "evapotranspiration": 5.654312046,
    "stages": "germination",
    "soilph": 6.890814297,
    "soilhumidity": 29.9523232,
    "microclimatictemp": 38.41781865,
    "microclimaticseason": "monsoon",
    "microclimaticwindspeed": 1.430180515,
    "yield_kg_per_plant": 2.123620357,
    "temperature_mean": 26.025205479452055,
    "temperature_min": 20.50358904109589,
    "temperature_max": 32.42471232876712,
    "relative_humidity": 59.86923287671233,
    "rainfall": 2.9466027397260275,
    "wind_speed": 2.1113698630136986,
    "solar_radiation": 15.839616438356163,
    "surface_pressure": 100.09013698630136,
    "temperature_range": 11.921123287671232,
    "rainfall_indicator": 1,
}


def load_model_artifact() -> dict[str, Any]:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained best regressor artifact is missing. Expected "
            f"{BEST_MODEL_PATH.as_posix()}."
        )
    artifact = joblib.load(BEST_MODEL_PATH)
    required = {"pipeline", "feature_columns", "model_name", "water_unit"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Model artifact is missing required keys: {sorted(missing)}")
    return artifact


ARTIFACT = load_model_artifact()
FEATURE_COLUMNS = ARTIFACT["feature_columns"]
_base_model_name = str(ARTIFACT.get("model_name", "regressor")).replace("_", " ").title()
_gb_impl = ARTIFACT.get("gradient_boosting_implementation")
MODEL_NAME = f"{_base_model_name} / XGBoost" if _gb_impl == "xgboost" else _base_model_name
UNIT = ARTIFACT.get("water_unit") or "liters/day"


def metrics_markdown() -> str:
    if not METRICS_PATH.exists():
        return "Metrics file not found. Run `python main.py` to regenerate evaluation results."
    rows = pd.read_csv(METRICS_PATH)
    best = rows.loc[rows["rmse"].idxmin()]
    return (
        f"**Best model:** {best['model'].replace('_', ' ').title()}  \n"
        f"**MAE:** {best['mae']:,.2f}  \n"
        f"**RMSE:** {best['rmse']:,.2f}  \n"
        f"**R2:** {best['r2']:.3f}"
    )


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [feature for feature in FEATURE_COLUMNS if feature not in payload]
    if missing:
        raise gr.Error(f"Missing required model features: {missing}")
    clean: dict[str, Any] = {}
    for feature in FEATURE_COLUMNS:
        value = payload[feature]
        if feature in NUMERIC_FEATURES:
            try:
                clean[feature] = float(value)
            except (TypeError, ValueError) as exc:
                raise gr.Error(f"{feature} must be a valid number.") from exc
        else:
            if value in (None, ""):
                raise gr.Error(f"{feature} is required.")
            clean[feature] = str(value)
    return clean


def predict(*values: Any) -> str:
    payload = dict(zip(FEATURE_COLUMNS, values, strict=True))
    clean = validate_payload(payload)
    prediction = float(ARTIFACT["pipeline"].predict(pd.DataFrame([clean], columns=FEATURE_COLUMNS))[0])
    return (
        f"Predicted irrigation requirement: {prediction:,.2f} {UNIT}\n"
        f"Model: {MODEL_NAME}\n"
        "Note: this is a regression estimate, not a trained irrigation/no-irrigation classification."
    )


def build_inputs() -> list[gr.components.Component]:
    return [
        gr.Textbox(label="Input leaf identifier", value=DEFAULT_SAMPLE["inputleaf"]),
        gr.Number(label="Stomata open/close", value=DEFAULT_SAMPLE["stomataopenclose"]),
        gr.Number(label="Nitrogen", value=DEFAULT_SAMPLE["nitrogen"]),
        gr.Number(label="Chlorophyll", value=DEFAULT_SAMPLE["chloropyll"]),
        gr.Number(label="Phosphorus", value=DEFAULT_SAMPLE["phosphorus"]),
        gr.Number(label="Leaf humidity", value=DEFAULT_SAMPLE["leafhumidity"]),
        gr.Number(label="Leaf evaporation", value=DEFAULT_SAMPLE["leafevaporation"]),
        gr.Number(label="Evapotranspiration", value=DEFAULT_SAMPLE["evapotranspiration"]),
        gr.Dropdown(
            label="Crop stage",
            choices=["germination", "tillering", "grandgrowth", "ripening", "maturity"],
            value=DEFAULT_SAMPLE["stages"],
            allow_custom_value=False,
        ),
        gr.Number(label="Soil pH", value=DEFAULT_SAMPLE["soilph"]),
        gr.Number(label="Soil humidity", value=DEFAULT_SAMPLE["soilhumidity"]),
        gr.Number(label="Microclimatic temperature", value=DEFAULT_SAMPLE["microclimatictemp"]),
        gr.Dropdown(
            label="Microclimatic season",
            choices=["summer", "winter", "monsoon"],
            value=DEFAULT_SAMPLE["microclimaticseason"],
            allow_custom_value=False,
        ),
        gr.Number(label="Microclimatic wind speed", value=DEFAULT_SAMPLE["microclimaticwindspeed"]),
        gr.Number(label="Yield kg per plant", value=DEFAULT_SAMPLE["yield_kg_per_plant"]),
        gr.Number(label="NASA POWER mean temperature", value=DEFAULT_SAMPLE["temperature_mean"]),
        gr.Number(label="NASA POWER minimum temperature", value=DEFAULT_SAMPLE["temperature_min"]),
        gr.Number(label="NASA POWER maximum temperature", value=DEFAULT_SAMPLE["temperature_max"]),
        gr.Number(label="NASA POWER relative humidity", value=DEFAULT_SAMPLE["relative_humidity"]),
        gr.Number(label="NASA POWER rainfall", value=DEFAULT_SAMPLE["rainfall"]),
        gr.Number(label="NASA POWER wind speed", value=DEFAULT_SAMPLE["wind_speed"]),
        gr.Number(label="NASA POWER solar radiation", value=DEFAULT_SAMPLE["solar_radiation"]),
        gr.Number(label="NASA POWER surface pressure", value=DEFAULT_SAMPLE["surface_pressure"]),
        gr.Number(label="Temperature range", value=DEFAULT_SAMPLE["temperature_range"]),
        gr.Number(label="Rainfall indicator", value=DEFAULT_SAMPLE["rainfall_indicator"]),
    ]


demo = gr.Interface(
    fn=predict,
    inputs=build_inputs(),
    outputs=gr.Textbox(label="Prediction"),
    title="Sugarcane Irrigation Advisory",
    description=(
        "Regression demo for predicting sugarcane irrigation water requirement using "
        "NIT Patna/SIDSS crop features and NASA POWER weather features."
    ),
    article=(
        metrics_markdown()
        + "\n\nThis academic decision-support demo is not a substitute for professional agronomic advice."
    ),
    examples=[[DEFAULT_SAMPLE[feature] for feature in FEATURE_COLUMNS]],
)


if __name__ == "__main__":
    demo.launch()
