from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_text

from src.models.common import Splits, build_preprocessor, combine_train_validation
from src.models.evaluate import evaluate_regressor


def _gradient_boosting_regressor(random_seed: int) -> tuple[Any, dict[str, list[Any]], str]:
    try:
        from xgboost import XGBRegressor

        return (
            XGBRegressor(objective="reg:squarederror", random_state=random_seed, n_jobs=-1),
            {
                "model__n_estimators": [80, 150, 250],
                "model__learning_rate": [0.03, 0.08, 0.15],
                "model__max_depth": [2, 3, 5],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__reg_lambda": [1.0, 5.0],
            },
            "xgboost",
        )
    except Exception:
        return (
            HistGradientBoostingRegressor(random_state=random_seed),
            {
                "model__max_iter": [80, 150, 250],
                "model__learning_rate": [0.03, 0.08, 0.15],
                "model__max_leaf_nodes": [15, 31],
                "model__l2_regularization": [0.0, 0.1, 1.0],
            },
            "sklearn_hist_gradient_boosting",
        )


def train_regressors(
    X: pd.DataFrame,
    y: pd.Series,
    splits: Splits,
    random_seed: int,
    cv_splits: int,
    n_iter: int,
    output_dir: str | Path,
    feature_columns: list[str],
    water_unit: str | None,
    temporal_cv: bool = False,
) -> tuple[dict[str, dict[str, object]], str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gb_model, gb_params, gb_impl = _gradient_boosting_regressor(random_seed)
    candidates: dict[str, tuple[Any, dict[str, list[Any]], bool]] = {
        "linear_regression": (Ridge(), {"model__alpha": [0.0, 0.1, 1.0, 10.0]}, True),
        "decision_tree": (
            DecisionTreeRegressor(random_state=random_seed),
            {
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 5],
            },
            False,
        ),
        "random_forest": (
            RandomForestRegressor(random_state=random_seed, n_jobs=-1),
            {
                "model__n_estimators": [100, 200, 350],
                "model__max_depth": [4, 8, None],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
            False,
        ),
        "gradient_boosting": (gb_model, gb_params, False),
    }

    x_tune, y_tune = combine_train_validation(splits)
    cv = TimeSeriesSplit(n_splits=max(2, cv_splits)) if temporal_cv else KFold(
        n_splits=max(2, cv_splits), shuffle=True, random_state=random_seed
    )
    results: dict[str, dict[str, object]] = {}
    best_name = ""
    best_rmse = float("inf")
    for name, (estimator, params, scale) in candidates.items():
        if name == "linear_regression" and params["model__alpha"][0] == 0.0:
            estimator = LinearRegression()
            params = {}
        pipe = Pipeline([("preprocess", build_preprocessor(X, scale_numeric=scale)), ("model", estimator)])
        if params:
            search = RandomizedSearchCV(
                pipe,
                params,
                n_iter=min(n_iter, max(1, sum(len(v) for v in params.values()))),
                scoring="neg_root_mean_squared_error",
                cv=cv,
                random_state=random_seed,
                n_jobs=1,
                refit=True,
            )
            search.fit(x_tune, y_tune)
            model = search.best_estimator_
            best_params = search.best_params_
            cv_score = float(search.best_score_)
        else:
            model = pipe.fit(x_tune, y_tune)
            best_params = {}
            cv_score = float("nan")
        metrics = evaluate_regressor(model, splits.X_test, splits.y_test)
        artifact = {
            "pipeline": model,
            "feature_columns": feature_columns,
            "model_name": name,
            "water_unit": water_unit,
            "gradient_boosting_implementation": gb_impl if name == "gradient_boosting" else None,
        }
        joblib.dump(artifact, output / f"{name}.joblib")
        results[name] = {
            "best_params": best_params,
            "cv_score": cv_score,
            "test_metrics": metrics,
            "gradient_boosting_implementation": gb_impl if name == "gradient_boosting" else None,
        }
        if name == "decision_tree":
            tree_text = export_text(model.named_steps["model"], max_depth=4)
            (output / "decision_tree_regressor.txt").write_text(tree_text, encoding="utf-8")
        rmse = float(metrics["rmse"])
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name

    joblib.dump(joblib.load(output / f"{best_name}.joblib"), output / "best_regressor.joblib")
    return results, best_name
