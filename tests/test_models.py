from __future__ import annotations

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.models.common import build_preprocessor
from src.models.predict import predict_irrigation_requirement


def test_regression_training_pipeline_predicts_shapes():
    X = pd.DataFrame(
        {
            "soil_moisture": [18, 25, 31, 40],
            "rainfall_3d": [0, 1, 3, 8],
            "crop_stage": ["early", "early", "growth", "maturity"],
        }
    )
    y = pd.Series([20.0, 15.0, 9.0, 3.0])
    model = Pipeline(
        [
            ("preprocess", build_preprocessor(X)),
            ("model", RandomForestRegressor(n_estimators=5, random_state=42)),
        ]
    ).fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (4,)


def test_prediction_interface_types(tmp_path):
    X = pd.DataFrame({"soil_moisture": [18, 30, 42], "rainfall_3d": [0, 2, 8]})
    reg = Pipeline(
        [
            ("preprocess", build_preprocessor(X)),
            ("model", RandomForestRegressor(n_estimators=5, random_state=42)),
        ]
    ).fit(X, pd.Series([20.0, 12.0, 2.0]))
    reg_path = tmp_path / "reg.joblib"
    joblib.dump(
        {
            "pipeline": reg,
            "feature_columns": list(X.columns),
            "model_name": "random_forest",
            "water_unit": "mm/day",
        },
        reg_path,
    )
    result = predict_irrigation_requirement(
        {"soil_moisture": 22, "rainfall_3d": 1},
        regressor_path=reg_path,
    )
    assert isinstance(result["predicted_irrigation_requirement"], float)
    assert result["water_unit"] == "mm/day"


def test_prediction_interface_rejects_missing_input(tmp_path):
    X = pd.DataFrame({"soil_moisture": [18, 30, 42]})
    reg = Pipeline([("preprocess", build_preprocessor(X)), ("model", RandomForestRegressor(n_estimators=5))]).fit(
        X, [20.0, 12.0, 2.0]
    )
    reg_path = tmp_path / "reg.joblib"
    joblib.dump({"pipeline": reg, "feature_columns": ["soil_moisture"]}, reg_path)
    with pytest.raises(ValueError):
        predict_irrigation_requirement({}, regressor_path=reg_path)
