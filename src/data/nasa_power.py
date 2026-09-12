from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

NASA_DAILY_POINT_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
SENTINELS = {-999, -999.0, -9999, -9999.0}
FIELD_NAMES = {
    "T2M": "temperature_mean",
    "T2M_MIN": "temperature_min",
    "T2M_MAX": "temperature_max",
    "RH2M": "relative_humidity",
    "PRECTOTCORR": "rainfall",
    "WS2M": "wind_speed",
    "ALLSKY_SFC_SW_DWN": "solar_radiation",
    "PS": "surface_pressure",
}


def validate_power_request(latitude: float, longitude: float, start_date: str, end_date: str) -> None:
    if not -90 <= latitude <= 90:
        raise ValueError("NASA POWER latitude must be between -90 and 90.")
    if not -180 <= longitude <= 180:
        raise ValueError("NASA POWER longitude must be between -180 and 180.")
    start = pd.to_datetime(start_date, errors="raise")
    end = pd.to_datetime(end_date, errors="raise")
    if start > end:
        raise ValueError("NASA POWER start_date must be on or before end_date.")


def fetch_nasa_power(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    parameters: list[str],
    cache_path: str | Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache = Path(cache_path)
    raw_json_path = cache.with_suffix(".raw.json")
    if cache.exists() and not force_refresh:
        return pd.read_csv(cache, parse_dates=["date"])

    validate_power_request(latitude, longitude, start_date, end_date)
    cache.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "community": "AG",
        "latitude": latitude,
        "longitude": longitude,
        "start": pd.to_datetime(start_date).strftime("%Y%m%d"),
        "end": pd.to_datetime(end_date).strftime("%Y%m%d"),
        "parameters": ",".join(parameters),
        "format": "JSON",
    }
    try:
        response = requests.get(NASA_DAILY_POINT_URL, params=params, timeout=45)
        response.raise_for_status()
    except requests.RequestException as exc:
        if cache.exists():
            return pd.read_csv(cache, parse_dates=["date"])
        raise RuntimeError(
            "NASA POWER data is mandatory, but the API request failed and no valid "
            f"cache exists at {cache}."
        ) from exc

    payload: dict[str, Any] = response.json()
    raw_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    parsed = parse_nasa_power_response(payload)
    parsed.to_csv(cache, index=False)
    return parsed


def parse_nasa_power_response(payload: dict[str, Any]) -> pd.DataFrame:
    parameter_block = payload.get("properties", {}).get("parameter")
    if not isinstance(parameter_block, dict) or not parameter_block:
        raise ValueError("NASA POWER response does not contain properties.parameter data.")

    rows: dict[str, dict[str, float | None]] = {}
    for raw_name, values in parameter_block.items():
        normalized = FIELD_NAMES.get(raw_name, raw_name.lower())
        for date_key, value in values.items():
            rows.setdefault(date_key, {})[normalized] = None if value in SENTINELS else value

    df = pd.DataFrame.from_dict(rows, orient="index").reset_index(names="date")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="raise")
    for column in df.columns:
        if column != "date":
            df[column] = pd.to_numeric(df[column], errors="coerce").replace(list(SENTINELS), np.nan)
    return df.sort_values("date").reset_index(drop=True)
