from __future__ import annotations

import pandas as pd
import pytest

from src.data.acquire_sugarcane import DatasetAcquisitionError, describe_primary_dataset, ensure_sidss_sugarcane_dataset
from src.data.load_sugarcane import inspect_schema, load_sugarcane_dataset
from src.data.merge_data import merge_sugarcane_with_weather
from src.data.nasa_power import parse_nasa_power_response, validate_power_request
from src.data.preprocess import clean_and_prepare_targets
from src.features.build_features import assert_no_target_leakage, build_features, select_feature_columns


def test_load_sugarcane_csv_and_schema(tmp_path):
    path = tmp_path / "sugarcane.csv"
    pd.DataFrame({"soil_moisture": [20], "irrigation_requirement": [12]}).to_csv(path, index=False)
    df = load_sugarcane_dataset(path)
    report = inspect_schema(df)
    assert report["row_count"] == 1
    assert "soil_moisture" in report["columns"]


def test_sidss_dataset_identity_is_described():
    description = describe_primary_dataset(
        {
            "name": "Sugarcane Irrigation Dataset for the SIDSS Framework",
            "institution_source_association": "NIT Patna",
            "role": "Primary sugarcane, soil, crop, and irrigation dataset",
        }
    )
    assert "SIDSS Framework" in description
    assert "NIT Patna" in description


def test_sidss_dataset_md5_mismatch_fails_clearly(tmp_path):
    path = tmp_path / "sugarcane.csv"
    path.write_text("not,the,dataset\n", encoding="utf-8")
    with pytest.raises(DatasetAcquisitionError, match="does not match the expected MD5"):
        ensure_sidss_sugarcane_dataset(
            path,
            {
                "name": "Sugarcane Irrigation Dataset for the SIDSS Framework",
                "expected_md5": "3efe98841dd7d873006fd64645584012",
            },
        )


def test_parse_nasa_power_response_normalizes_fields_and_sentinels():
    payload = {
        "properties": {
            "parameter": {
                "T2M": {"20260101": 30.5},
                "PRECTOTCORR": {"20260101": -999},
            }
        }
    }
    df = parse_nasa_power_response(payload)
    assert df.loc[0, "temperature_mean"] == 30.5
    assert pd.isna(df.loc[0, "rainfall"])


def test_validate_power_request_rejects_bad_latitude():
    with pytest.raises(ValueError):
        validate_power_request(100.0, 77.0, "2026-01-01", "2026-01-02")


def test_date_alignment_reports_join_success_rate():
    crop = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "irrigation_requirement": [10, 20]})
    weather = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "rainfall": [2.0]})
    merged, report = merge_sugarcane_with_weather(crop, weather, "date")
    assert merged["rainfall"].notna().sum() == 1
    assert report["join_success_rate"] == 0.5


def test_missing_values_are_reported_for_regression_pipeline():
    raw = pd.DataFrame(
        {
            "soil_moisture": [20, None, 35],
            "irrigation_requirement": [0.0, 12.0, 4.0],
        }
    )
    prepared = clean_and_prepare_targets(
        raw,
        regression_target="irrigation_requirement",
    )
    assert prepared.regression_target == "irrigation_requirement"
    assert prepared.report["modeling_objective"] == "regression_only"


def test_feature_generation_and_no_target_leakage():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "soil_moisture": [20, 21, 22],
            "temperature_mean": [30, 31, 32],
            "temperature_min": [25, 26, 27],
            "temperature_max": [35, 36, 37],
            "rainfall": [0, 1, 2],
            "irrigation_requirement": [10, 9, 8],
            "predicted_water_liters_per_day": [11, 10, 9],
        }
    )
    featured = build_features(df, "date")
    assert "temperature_range" in featured.columns
    assert featured.loc[1, "rainfall_3d"] == 0
    columns = select_feature_columns(featured, "irrigation_requirement", "date")
    assert_no_target_leakage(columns, "irrigation_requirement")
    assert "irrigation_requirement" not in columns
    assert "predicted_water_liters_per_day" not in columns
