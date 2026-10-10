"""
Unit tests for data_generator.py
================================
Validates synthetic data generation, physical realism, schema consistency,
and absence of missing/infinite values.
"""

import numpy as np
import pandas as pd
import pytest

from data_generator import (
    DATASET_COLUMNS,
    generate_cooling_data,
    load_or_generate_dataset,
)


def test_dataset_generation_shape():
    """Verify generated dataset contains expected number of rows and columns."""
    n_samples = 2500
    df = generate_cooling_data(n_samples=n_samples, random_seed=42)
    assert len(df) == n_samples
    assert len(df.columns) == len(DATASET_COLUMNS)
    assert list(df.columns) == DATASET_COLUMNS


def test_no_missing_or_infinite_values():
    """Verify dataset contains zero nulls, NaNs, or infinite values."""
    df = generate_cooling_data(n_samples=1000, random_seed=123)
    assert df.isnull().sum().sum() == 0
    assert df.isna().sum().sum() == 0

    numeric_cols = [c for c in df.columns if c != "timestamp"]
    for col in numeric_cols:
        assert np.all(np.isfinite(df[col].values)), f"Non-finite values found in {col}"


def test_physical_ranges():
    """Verify that simulated variables conform to realistic physical laws and boundaries."""
    df = generate_cooling_data(n_samples=2000, random_seed=999)

    # IT and Cooling Power must be strictly positive
    assert (df["it_load_kw"] > 0).all(), "IT load must be positive"
    assert (df["cooling_power_kw"] > 0).all(), "Cooling power must be positive"

    # Total facility power must be strictly greater than IT load
    assert (df["total_facility_power_kw"] > df["it_load_kw"]).all()

    # Actuator speeds bounded between 0% and 100%
    assert (df["fan_speed_pct"] >= 0.0).all() and (df["fan_speed_pct"] <= 100.0).all()
    assert (df["pump_speed_pct"] >= 0.0).all() and (df["pump_speed_pct"] <= 100.0).all()

    # Temperatures
    assert (df["ambient_temp_c"] >= 0.0).all() and (df["ambient_temp_c"] <= 55.0).all()
    assert (df["inlet_temp_c"] >= 15.0).all() and (df["inlet_temp_c"] <= 35.0).all()

    # Hot aisle outlet temp must be warmer than cold aisle inlet temp
    assert (df["outlet_temp_c"] > df["inlet_temp_c"]).all()

    # PUE definition: Total Facility Power / IT Load
    computed_pue = np.round(df["total_facility_power_kw"] / df["it_load_kw"], 4)
    # Check agreement within rounding margin
    np.testing.assert_allclose(df["pue"], computed_pue, atol=1e-3)
    assert (df["pue"] >= 1.0).all(), "PUE cannot be less than 1.0"


def test_dataset_reproducibility():
    """Verify identical random seeds produce identical datasets."""
    df1 = generate_cooling_data(n_samples=500, random_seed=777)
    df2 = generate_cooling_data(n_samples=500, random_seed=777)
    pd.testing.assert_frame_equal(df1, df2)


def test_load_or_generate_dataset(tmp_path):
    """Verify dataset loading and forced regeneration behaviors."""
    target_csv = tmp_path / "test_cooling.csv"
    assert not target_csv.exists()

    # Generate initial dataset
    df1 = load_or_generate_dataset(file_path=target_csv, n_samples=600, random_seed=42)
    assert target_csv.exists()
    assert len(df1) == 600

    # Load existing without regenerating
    df2 = load_or_generate_dataset(file_path=target_csv, force_regenerate=False)
    assert len(df2) == 600
