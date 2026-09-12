from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def _load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if "pipeline" not in artifact or "feature_columns" not in artifact:
        raise ValueError(f"Invalid model artifact: {path}")
    return artifact


def predict_irrigation_requirement(
    sample: dict[str, Any],
    regressor_path: str | Path = "models/regressors/best_regressor.joblib",
) -> dict[str, Any]:
    regressor = _load_artifact(regressor_path)
    required_columns = regressor["feature_columns"]
    missing = [column for column in required_columns if column not in sample]
    if missing:
        raise ValueError(f"Inference input is missing required features: {missing}")

    X = pd.DataFrame([{column: sample.get(column) for column in required_columns}])
    recommended_water = float(regressor["pipeline"].predict(X)[0])
    return {
        "predicted_irrigation_requirement": recommended_water,
        "water_unit": regressor.get("water_unit"),
        "regression_model": regressor.get("model_name"),
    }


make_irrigation_advisory = predict_irrigation_requirement
