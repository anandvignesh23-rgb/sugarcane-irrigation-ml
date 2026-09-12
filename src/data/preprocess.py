from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PreparedData:
    frame: pd.DataFrame
    regression_target: str
    report: dict[str, Any]


def clean_and_prepare_targets(
    df: pd.DataFrame,
    regression_target: str,
) -> PreparedData:
    if regression_target not in df.columns:
        raise ValueError(f"Regression target '{regression_target}' is not present in the dataset.")

    original_rows = len(df)
    missing_before = df.isna().sum().astype(int).to_dict()
    duplicate_count = int(df.duplicated().sum())
    clean = df.drop_duplicates().copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)

    clean[regression_target] = pd.to_numeric(clean[regression_target], errors="coerce")
    invalid_target_rows = int(clean[regression_target].isna().sum())
    clean = clean.dropna(subset=[regression_target])
    if clean.empty:
        raise ValueError("No rows remain after removing missing regression targets.")

    numeric_cols = clean.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_cols:
        if column == regression_target:
            continue
        lower = clean[column].quantile(0.01)
        upper = clean[column].quantile(0.99)
        clean[column] = clean[column].clip(lower=lower, upper=upper)

    report: dict[str, Any] = {
        "original_row_count": int(original_rows),
        "final_row_count": int(len(clean)),
        "missing_values_before": missing_before,
        "missing_values_after": clean.isna().sum().astype(int).to_dict(),
        "duplicate_count": duplicate_count,
        "invalid_regression_target_rows": invalid_target_rows,
        "columns_removed": [],
        "columns_transformed": [],
        "modeling_objective": "regression_only",
        "classification_status": (
            "Skipped because the SIDSS dataset provides a numerical irrigation target "
            "but no scientifically reliable binary irrigation/no-irrigation label."
        ),
    }
    return PreparedData(clean, regression_target, report)
