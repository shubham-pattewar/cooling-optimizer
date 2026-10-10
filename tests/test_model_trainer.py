"""
Unit tests for model_trainer.py
===============================
Validates model training, baseline comparisons, schema consistency,
and absence of target leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_trainer import (
    FEATURE_COLUMNS,
    TARGET_POWER,
    TARGET_TEMP,
    load_trained_models,
    train_models,
)


def test_feature_schema_consistency():
    """Verify strictly defined feature schema has no target leakage."""
    assert len(FEATURE_COLUMNS) == 5
    assert "ambient_temp_c" in FEATURE_COLUMNS
    assert "humidity_pct" in FEATURE_COLUMNS
    assert "it_load_kw" in FEATURE_COLUMNS
    assert "fan_speed_pct" in FEATURE_COLUMNS
    assert "pump_speed_pct" in FEATURE_COLUMNS

    # Critical: No target leakage
    forbidden_features = [
        "inlet_temp_c",
        "outlet_temp_c",
        "cooling_power_kw",
        "total_facility_power_kw",
        "pue",
        "timestamp",
    ]
    for feat in forbidden_features:
        assert (
            feat not in FEATURE_COLUMNS
        ), f"Target leakage detected: {feat} found in feature schema!"


def test_train_models_pipeline(tmp_path):
    """Verify full training pipeline on synthetic data in a temporary directory."""
    from data_generator import generate_cooling_data, save_dataset

    data_file = tmp_path / "temp_cooling.csv"
    df = generate_cooling_data(n_samples=800, random_seed=42)
    save_dataset(df, data_file)

    models_dir = tmp_path / "models"
    meta = train_models(data_path=data_file, models_dir=models_dir, test_size=0.2, random_state=42)

    # Check saved files
    assert (models_dir / "cooling_power_model.joblib").exists()
    assert (models_dir / "inlet_temp_model.joblib").exists()
    assert (models_dir / "cooling_power_baseline.joblib").exists()
    assert (models_dir / "inlet_temp_baseline.joblib").exists()
    assert (models_dir / "model_metadata.json").exists()

    # Check metrics existence
    power_rf = meta["cooling_power_metrics"]["random_forest"]
    power_dummy = meta["cooling_power_metrics"]["dummy_baseline"]
    temp_rf = meta["inlet_temp_metrics"]["random_forest"]
    temp_dummy = meta["inlet_temp_metrics"]["dummy_baseline"]

    assert "mae" in power_rf and "rmse" in power_rf and "r2" in power_rf
    assert "mae" in temp_rf and "rmse" in temp_rf and "r2" in temp_rf

    # Random forest should outperform dummy baseline on test set
    assert power_rf["r2"] > power_dummy["r2"]
    assert power_rf["mae"] < power_dummy["mae"]
    assert temp_rf["r2"] > temp_dummy["r2"]
    assert temp_rf["mae"] < temp_dummy["mae"]

    # Verify model loading
    p_mod, t_mod, meta_loaded = load_trained_models(models_dir=models_dir)
    assert hasattr(p_mod, "predict")
    assert hasattr(t_mod, "predict")

    # Verify inference on a single sample
    sample_input = pd.DataFrame(
        [
            {
                "ambient_temp_c": 25.0,
                "humidity_pct": 55.0,
                "it_load_kw": 800.0,
                "fan_speed_pct": 60.0,
                "pump_speed_pct": 60.0,
            }
        ]
    )[FEATURE_COLUMNS]

    pred_p = p_mod.predict(sample_input)
    pred_t = t_mod.predict(sample_input)

    assert len(pred_p) == 1 and np.isfinite(pred_p[0]) and pred_p[0] > 0
    assert len(pred_t) == 1 and np.isfinite(pred_t[0]) and pred_t[0] > 10
