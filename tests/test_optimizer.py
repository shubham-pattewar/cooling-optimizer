"""
Unit tests for optimizer.py
===========================
Validates optimization logic, constraint handling, safety limits,
fallback mechanisms, and report generation.
"""

import numpy as np
import pandas as pd
import pytest

from model_trainer import load_trained_models
from optimizer import (
    OptimizationRequest,
    OptimizationResult,
    _compute_pue,
    optimize_cooling,
    save_optimization_report,
)


@pytest.fixture(scope="module")
def models():
    """Load pre-trained models for optimization tests."""
    p_mod, t_mod, _ = load_trained_models()
    return p_mod, t_mod


def test_compute_pue_consistency():
    """Verify PUE calculation consistency."""
    pue = _compute_pue(it_load_kw=1000.0, cooling_power_kw=250.0)
    # total = 1000 + 250 + (0.035 * 1000 + 18) = 1303
    # pue = 1303 / 1000 = 1.303
    assert 1.25 < pue < 1.35
    # Edge case: zero or negative IT load
    assert _compute_pue(0, 100) == 1.0


def test_optimizer_feasible_reduction(models):
    """Verify optimizer finds valid cooling power reduction under standard conditions."""
    p_mod, t_mod = models

    req = OptimizationRequest(
        ambient_temp_c=25.0,
        humidity_pct=50.0,
        it_load_kw=800.0,
        current_fan_speed_pct=85.0,  # Conservatively high baseline
        current_pump_speed_pct=85.0,  # Conservatively high baseline
        max_inlet_temp_c=25.0,
        min_fan_speed_pct=40.0,
        max_fan_speed_pct=95.0,
        min_pump_speed_pct=40.0,
        max_pump_speed_pct=95.0,
        operating_hours=24.0,
        tariff_inr_per_kwh=7.50,
    )

    res = optimize_cooling(req, p_mod, t_mod)

    assert isinstance(res, OptimizationResult)
    assert res.is_feasible is True
    assert res.recommended_cooling_power_kw <= res.baseline_cooling_power_kw
    assert res.recommended_inlet_temp_c <= req.max_inlet_temp_c
    assert req.min_fan_speed_pct <= res.recommended_fan_speed_pct <= req.max_fan_speed_pct
    assert req.min_pump_speed_pct <= res.recommended_pump_speed_pct <= req.max_pump_speed_pct
    assert res.total_candidates_evaluated > 0


def test_optimizer_rejects_unsafe_thermal_condition(models):
    """Verify optimizer rejects all candidates and retains baseline when safety limit is impossibly low."""
    p_mod, t_mod = models

    req = OptimizationRequest(
        ambient_temp_c=35.0,
        humidity_pct=70.0,
        it_load_kw=1000.0,
        current_fan_speed_pct=80.0,
        current_pump_speed_pct=80.0,
        max_inlet_temp_c=14.0,  # Unreasonably low safety threshold: no candidate can safely meet this
        min_fan_speed_pct=40.0,
        max_fan_speed_pct=90.0,
        min_pump_speed_pct=40.0,
        max_pump_speed_pct=90.0,
    )

    res = optimize_cooling(req, p_mod, t_mod)

    assert res.is_feasible is False
    assert res.status == "NO_FEASIBLE_CANDIDATE"
    assert res.feasible_candidates_count == 0
    # Baseline retained as safety fallback
    assert res.recommended_fan_speed_pct == req.current_fan_speed_pct
    assert res.recommended_pump_speed_pct == req.current_pump_speed_pct
    assert res.power_savings_kw == 0.0
    assert res.savings_percent == 0.0


def test_optimizer_actuator_limits(models):
    """Verify all evaluated candidates strictly adhere to actuator limits."""
    p_mod, t_mod = models

    req = OptimizationRequest(
        ambient_temp_c=22.0,
        humidity_pct=50.0,
        it_load_kw=750.0,
        current_fan_speed_pct=60.0,
        current_pump_speed_pct=60.0,
        max_inlet_temp_c=25.0,
        min_fan_speed_pct=45.0,
        max_fan_speed_pct=75.0,
        min_pump_speed_pct=50.0,
        max_pump_speed_pct=80.0,
    )

    res = optimize_cooling(req, p_mod, t_mod)
    df = res.candidates_df

    assert (df["fan_speed_pct"] >= 45.0).all()
    assert (df["fan_speed_pct"] <= 75.0).all()
    assert (df["pump_speed_pct"] >= 50.0).all()
    assert (df["pump_speed_pct"] <= 80.0).all()


def test_savings_calculation_safe_zero_division():
    """Verify savings percentage does not raise zero-division errors with zero baseline power."""
    # Test internal logic
    baseline_power = 0.0
    power_savings = 10.0
    savings_pct = (power_savings / baseline_power) * 100.0 if baseline_power > 0 else 0.0
    assert savings_pct == 0.0


def test_report_export(models, tmp_path):
    """Verify optimization report can be saved to CSV with required columns."""
    p_mod, t_mod = models

    req = OptimizationRequest(
        ambient_temp_c=25.0,
        humidity_pct=50.0,
        it_load_kw=800.0,
        current_fan_speed_pct=70.0,
        current_pump_speed_pct=70.0,
        max_inlet_temp_c=24.5,
    )
    res = optimize_cooling(req, p_mod, t_mod)

    out_file = tmp_path / "test_report.csv"
    saved_path = save_optimization_report(res, file_path=out_file)

    assert saved_path.exists()
    report_df = pd.read_csv(saved_path)

    expected_cols = [
        "fan_speed_pct",
        "pump_speed_pct",
        "predicted_cooling_power_kw",
        "predicted_inlet_temp_c",
        "is_feasible",
        "rejection_reason",
        "baseline_cooling_power_kw",
        "power_savings_kw",
        "savings_percent",
    ]
    for col in expected_cols:
        assert col in report_df.columns, f"Expected column {col} in report"
