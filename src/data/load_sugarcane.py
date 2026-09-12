from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_sugarcane_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Sugarcane dataset not found at {dataset_path}. Place the real crop/soil "
            "dataset there or update data.sugarcane_path in config.yaml."
        )
    if dataset_path.suffix.lower() == ".csv":
        df = pd.read_csv(dataset_path)
    elif dataset_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(dataset_path)
    else:
        raise ValueError(f"Unsupported sugarcane dataset format: {dataset_path.suffix}")
    if df.empty:
        raise ValueError("Sugarcane dataset is empty.")
    return df


def inspect_schema(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": df.isna().sum().astype(int).to_dict(),
    }


def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {column.lower().strip(): column for column in df.columns}
    for candidate in candidates:
        if candidate and candidate.lower().strip() in normalized:
            return normalized[candidate.lower().strip()]
    return None
