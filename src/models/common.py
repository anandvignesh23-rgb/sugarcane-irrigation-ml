from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0,
        remainder="drop",
    )


def make_holdout_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    validation_size: float,
    random_seed: int,
    stratify: bool = False,
    temporal_order: pd.Series | None = None,
) -> Splits:
    if temporal_order is not None:
        ordered_index = temporal_order.sort_values().index
        X_ordered = X.loc[ordered_index]
        y_ordered = y.loc[ordered_index]
        n = len(X_ordered)
        test_n = max(1, int(round(n * test_size)))
        val_n = max(1, int(round(n * validation_size)))
        train_end = max(1, n - test_n - val_n)
        val_end = max(train_end + 1, n - test_n)
        return Splits(
            X_ordered.iloc[:train_end],
            X_ordered.iloc[train_end:val_end],
            X_ordered.iloc[val_end:],
            y_ordered.iloc[:train_end],
            y_ordered.iloc[train_end:val_end],
            y_ordered.iloc[val_end:],
        )

    stratify_y = y if stratify and y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_y,
    )
    relative_val = validation_size / max(1e-9, 1 - test_size)
    stratify_train = y_train_val if stratify and y_train_val.nunique() == 2 and y_train_val.value_counts().min() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val,
        random_state=random_seed,
        stratify=stratify_train,
    )
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)


def combine_train_validation(splits: Splits) -> tuple[pd.DataFrame, pd.Series]:
    return pd.concat([splits.X_train, splits.X_val]), pd.concat([splits.y_train, splits.y_val])
