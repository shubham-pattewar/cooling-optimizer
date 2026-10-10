"""
AI Data Center Cooling Optimizer - Optimization Engine
======================================================
Inspired by Google DeepMind's data center cooling efficiency research.

This module implements a constraint-aware candidate search algorithm that
evaluates potential cooling actuator configurations (fan speed and pump speed)
to minimize total cooling power while enforcing thermal safety constraints
(server inlet air temperature <= maximum allowable threshold).

Safety & Modeling Integrity Principles:
---------------------------------------
1. No Fake Heuristics:
   Optimization is strictly driven by evaluating surrogate ML model predictions
   across a candidate control grid under identical ambient and workload conditions.
2. Safety Constraint Filtering:
   Every candidate with a predicted server inlet temperature exceeding the safety
   threshold is rejected.
3. Transparent Fallback:
   If no candidate satisfies thermal constraints or offers energy improvements,
   the baseline configuration is preserved and flagged with full explanations.
4. Physical vs Learned Disclaimer:
   Learned models approximate physical plant dynamics; model-predicted safety does
   not constitute physical certification for production data centers.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from model_trainer import FEATURE_COLUMNS, load_trained_models


@dataclass
class OptimizationRequest:
    """Input parameters for data center cooling optimization."""

    ambient_temp_c: float
    humidity_pct: float
    it_load_kw: float
    current_fan_speed_pct: float
    current_pump_speed_pct: float
    max_inlet_temp_c: float = 24.5
    min_fan_speed_pct: float = 35.0
    max_fan_speed_pct: float = 95.0
    min_pump_speed_pct: float = 35.0
    max_pump_speed_pct: float = 95.0
    fan_step_pct: float = 2.5
    pump_step_pct: float = 2.5
    operating_hours: float = 24.0
    tariff_inr_per_kwh: float = 7.50


@dataclass
class OptimizationResult:
    """Structured output of the optimization process."""

    status: str
    is_feasible: bool
    baseline_fan_speed_pct: float
    baseline_pump_speed_pct: float
    baseline_cooling_power_kw: float
    baseline_inlet_temp_c: float
    baseline_pue: float
    recommended_fan_speed_pct: float
    recommended_pump_speed_pct: float
    recommended_cooling_power_kw: float
    recommended_inlet_temp_c: float
    recommended_pue: float
    power_savings_kw: float
    savings_percent: float
    energy_savings_kwh: float
    cost_savings_inr: float
    operating_hours: float
    tariff_inr_per_kwh: float
    total_candidates_evaluated: int
    feasible_candidates_count: int
    rejected_candidates_count: int
    execution_time_ms: float
    message: str
    candidates_df: pd.DataFrame


def _compute_pue(it_load_kw: float, cooling_power_kw: float) -> float:
    """Calculate Power Usage Effectiveness assuming standard ancillary facility overhead."""
    if it_load_kw <= 0:
        return 1.0
    ancillary_power = 0.035 * it_load_kw + 18.0
    total_facility = it_load_kw + cooling_power_kw + ancillary_power
    return round(float(total_facility / it_load_kw), 4)


def optimize_cooling(
    request: OptimizationRequest,
    power_model: Any,
    temp_model: Any,
) -> OptimizationResult:
    """Execute constraint-aware candidate search to optimize cooling settings.

    Args:
        request: OptimizationRequest containing environmental conditions,
                 current settings, and constraints.
        power_model: Trained model predicting cooling_power_kw.
        temp_model: Trained model predicting inlet_temp_c.

    Returns:
        OptimizationResult with baseline vs optimized telemetry and candidate breakdown.
    """
    start_time = time.perf_counter()

    # 1. Establish Baseline Predictions using identical surrogate models
    baseline_features = pd.DataFrame(
        [
            {
                "ambient_temp_c": request.ambient_temp_c,
                "humidity_pct": request.humidity_pct,
                "it_load_kw": request.it_load_kw,
                "fan_speed_pct": request.current_fan_speed_pct,
                "pump_speed_pct": request.current_pump_speed_pct,
            }
        ]
    )[FEATURE_COLUMNS]

    baseline_cooling_power = float(power_model.predict(baseline_features)[0])
    baseline_inlet_temp = float(temp_model.predict(baseline_features)[0])
    baseline_pue = _compute_pue(request.it_load_kw, baseline_cooling_power)

    # 2. Generate Candidate Grid within actuator limits
    fan_min = max(20.0, float(request.min_fan_speed_pct))
    fan_max = min(100.0, float(request.max_fan_speed_pct))
    pump_min = max(20.0, float(request.min_pump_speed_pct))
    pump_max = min(100.0, float(request.max_pump_speed_pct))

    fan_step = max(0.5, float(request.fan_step_pct))
    pump_step = max(0.5, float(request.pump_step_pct))

    fan_grid = np.arange(fan_min, fan_max + 1e-4, fan_step)
    pump_grid = np.arange(pump_min, pump_max + 1e-4, pump_step)

    mesh_fan, mesh_pump = np.meshgrid(fan_grid, pump_grid)
    fan_candidates = mesh_fan.ravel()
    pump_candidates = mesh_pump.ravel()
    n_candidates = len(fan_candidates)

    candidate_df = pd.DataFrame(
        {
            "ambient_temp_c": np.full(n_candidates, request.ambient_temp_c),
            "humidity_pct": np.full(n_candidates, request.humidity_pct),
            "it_load_kw": np.full(n_candidates, request.it_load_kw),
            "fan_speed_pct": fan_candidates,
            "pump_speed_pct": pump_candidates,
        }
    )[FEATURE_COLUMNS]

    # 3. Batch Predict Cooling Power and Inlet Temperature for all candidates
    pred_powers = power_model.predict(candidate_df)
    pred_temps = temp_model.predict(candidate_df)

    candidate_df["predicted_cooling_power_kw"] = np.round(pred_powers, 2)
    candidate_df["predicted_inlet_temp_c"] = np.round(pred_temps, 2)

    # 4. Apply Thermal Safety & Actuator Constraints
    is_temp_safe = candidate_df["predicted_inlet_temp_c"] <= request.max_inlet_temp_c
    is_fan_safe = (candidate_df["fan_speed_pct"] >= fan_min) & (candidate_df["fan_speed_pct"] <= fan_max)
    is_pump_safe = (candidate_df["pump_speed_pct"] >= pump_min) & (candidate_df["pump_speed_pct"] <= pump_max)

    candidate_df["is_feasible"] = is_temp_safe & is_fan_safe & is_pump_safe

    # Document rejection reason
    rejection_reasons: List[str] = []
    for safe_t, safe_f, safe_p in zip(is_temp_safe, is_fan_safe, is_pump_safe):
        if safe_t and safe_f and safe_p:
            rejection_reasons.append("None (Feasible)")
        else:
            reasons = []
            if not safe_t:
                reasons.append(f"Inlet Temp > Max ({request.max_inlet_temp_c:.1f}°C)")
            if not safe_f or not safe_p:
                reasons.append("Outside Actuator Bounds")
            rejection_reasons.append("; ".join(reasons))

    candidate_df["rejection_reason"] = rejection_reasons

    # 5. Evaluate Feasible Set & Find Minimum Energy Candidate
    feasible_df = candidate_df[candidate_df["is_feasible"]]
    feasible_count = len(feasible_df)
    rejected_count = n_candidates - feasible_count

    execution_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    if feasible_count == 0:
        # No candidate meets thermal constraints
        return OptimizationResult(
            status="NO_FEASIBLE_CANDIDATE",
            is_feasible=False,
            baseline_fan_speed_pct=round(request.current_fan_speed_pct, 2),
            baseline_pump_speed_pct=round(request.current_pump_speed_pct, 2),
            baseline_cooling_power_kw=round(baseline_cooling_power, 2),
            baseline_inlet_temp_c=round(baseline_inlet_temp, 2),
            baseline_pue=baseline_pue,
            recommended_fan_speed_pct=round(request.current_fan_speed_pct, 2),
            recommended_pump_speed_pct=round(request.current_pump_speed_pct, 2),
            recommended_cooling_power_kw=round(baseline_cooling_power, 2),
            recommended_inlet_temp_c=round(baseline_inlet_temp, 2),
            recommended_pue=baseline_pue,
            power_savings_kw=0.0,
            savings_percent=0.0,
            energy_savings_kwh=0.0,
            cost_savings_inr=0.0,
            operating_hours=request.operating_hours,
            tariff_inr_per_kwh=request.tariff_inr_per_kwh,
            total_candidates_evaluated=n_candidates,
            feasible_candidates_count=0,
            rejected_candidates_count=rejected_count,
            execution_time_ms=execution_time_ms,
            message=(
                f"No candidate configuration satisfied the safety limit of {request.max_inlet_temp_c:.1f}°C. "
                "Baseline configuration retained to prevent thermal risk."
            ),
            candidates_df=candidate_df,
        )

    # Sort feasible candidates by predicted cooling power (ascending)
    best_candidate = feasible_df.sort_values("predicted_cooling_power_kw", ascending=True).iloc[0]
    rec_power = float(best_candidate["predicted_cooling_power_kw"])
    rec_temp = float(best_candidate["predicted_inlet_temp_c"])
    rec_fan = float(best_candidate["fan_speed_pct"])
    rec_pump = float(best_candidate["pump_speed_pct"])
    rec_pue = _compute_pue(request.it_load_kw, rec_power)

    # 6. Check if best feasible candidate improves upon baseline
    power_savings = baseline_cooling_power - rec_power

    if power_savings <= 0.05:
        # Baseline is already at or near optimal within feasible set
        return OptimizationResult(
            status="BASELINE_ALREADY_OPTIMAL",
            is_feasible=True,
            baseline_fan_speed_pct=round(request.current_fan_speed_pct, 2),
            baseline_pump_speed_pct=round(request.current_pump_speed_pct, 2),
            baseline_cooling_power_kw=round(baseline_cooling_power, 2),
            baseline_inlet_temp_c=round(baseline_inlet_temp, 2),
            baseline_pue=baseline_pue,
            recommended_fan_speed_pct=round(request.current_fan_speed_pct, 2),
            recommended_pump_speed_pct=round(request.current_pump_speed_pct, 2),
            recommended_cooling_power_kw=round(baseline_cooling_power, 2),
            recommended_inlet_temp_c=round(baseline_inlet_temp, 2),
            recommended_pue=baseline_pue,
            power_savings_kw=0.0,
            savings_percent=0.0,
            energy_savings_kwh=0.0,
            cost_savings_inr=0.0,
            operating_hours=request.operating_hours,
            tariff_inr_per_kwh=request.tariff_inr_per_kwh,
            total_candidates_evaluated=n_candidates,
            feasible_candidates_count=feasible_count,
            rejected_candidates_count=rejected_count,
            execution_time_ms=execution_time_ms,
            message=(
                "Current baseline is already operating at or near minimum feasible cooling power. "
                "No safe configuration yielded lower power consumption."
            ),
            candidates_df=candidate_df,
        )

    # Feasible recommendation with validated power savings
    savings_pct = (
        (power_savings / baseline_cooling_power) * 100.0 if baseline_cooling_power > 0 else 0.0
    )
    energy_saved_kwh = power_savings * request.operating_hours
    cost_saved_inr = energy_saved_kwh * request.tariff_inr_per_kwh

    # Mark recommended in candidates_df
    candidate_df["is_recommended"] = (candidate_df["fan_speed_pct"] == rec_fan) & (
        candidate_df["pump_speed_pct"] == rec_pump
    )

    return OptimizationResult(
        status="FEASIBLE_RECOMMENDATION",
        is_feasible=True,
        baseline_fan_speed_pct=round(request.current_fan_speed_pct, 2),
        baseline_pump_speed_pct=round(request.current_pump_speed_pct, 2),
        baseline_cooling_power_kw=round(baseline_cooling_power, 2),
        baseline_inlet_temp_c=round(baseline_inlet_temp, 2),
        baseline_pue=baseline_pue,
        recommended_fan_speed_pct=round(rec_fan, 2),
        recommended_pump_speed_pct=round(rec_pump, 2),
        recommended_cooling_power_kw=round(rec_power, 2),
        recommended_inlet_temp_c=round(rec_temp, 2),
        recommended_pue=rec_pue,
        power_savings_kw=round(power_savings, 2),
        savings_percent=round(savings_pct, 2),
        energy_savings_kwh=round(energy_saved_kwh, 2),
        cost_savings_inr=round(cost_saved_inr, 2),
        operating_hours=request.operating_hours,
        tariff_inr_per_kwh=request.tariff_inr_per_kwh,
        total_candidates_evaluated=n_candidates,
        feasible_candidates_count=feasible_count,
        rejected_candidates_count=rejected_count,
        execution_time_ms=execution_time_ms,
        message=(
            f"Feasible optimization identified: {savings_pct:.1f}% cooling power reduction "
            f"({power_savings:.1f} kW) while maintaining predicted inlet temperature at {rec_temp:.1f}°C."
        ),
        candidates_df=candidate_df,
    )


def save_optimization_report(
    result: OptimizationResult,
    file_path: Optional[Path | str] = None,
) -> Path:
    """Save optimization summary and candidate telemetry to a CSV file.

    Args:
        result: OptimizationResult object.
        file_path: Target path for the report CSV.

    Returns:
        Path to the saved report file.
    """
    default_dir = Path(__file__).resolve().parent / "reports"
    default_dir.mkdir(parents=True, exist_ok=True)
    destination = (
        Path(file_path)
        if file_path
        else default_dir
        / f"optimization_report_{int(time.time())}.csv"
    )

    # Export candidate grid with summary metadata
    df = result.candidates_df.copy()
    df["baseline_cooling_power_kw"] = result.baseline_cooling_power_kw
    df["baseline_inlet_temp_c"] = result.baseline_inlet_temp_c
    df["power_savings_kw"] = result.power_savings_kw
    df["savings_percent"] = result.savings_percent
    df["status"] = result.status

    df.to_csv(destination, index=False)
    return destination


if __name__ == "__main__":
    print("Testing Cooling Optimization Engine...")
    p_model, t_model, _ = load_trained_models()

    req = OptimizationRequest(
        ambient_temp_c=28.0,
        humidity_pct=60.0,
        it_load_kw=900.0,
        current_fan_speed_pct=85.0,
        current_pump_speed_pct=80.0,
        max_inlet_temp_c=24.0,
        min_fan_speed_pct=40.0,
        max_fan_speed_pct=95.0,
        min_pump_speed_pct=40.0,
        max_pump_speed_pct=95.0,
        operating_hours=24.0,
        tariff_inr_per_kwh=7.50,
    )

    res = optimize_cooling(req, p_model, t_model)

    print(f"\nOptimization Status: {res.status}")
    print(f"Message: {res.message}")
    print(
        f"Baseline: Fan {res.baseline_fan_speed_pct}%, Pump {res.baseline_pump_speed_pct}% -> "
        f"Power: {res.baseline_cooling_power_kw} kW, Inlet: {res.baseline_inlet_temp_c}°C, PUE: {res.baseline_pue}"
    )
    print(
        f"Optimized: Fan {res.recommended_fan_speed_pct}%, Pump {res.recommended_pump_speed_pct}% -> "
        f"Power: {res.recommended_cooling_power_kw} kW, Inlet: {res.recommended_inlet_temp_c}°C, PUE: {res.recommended_pue}"
    )
    print(f"Power Savings: {res.power_savings_kw} kW ({res.savings_percent}%)")
    print(f"24h Energy Savings: {res.energy_savings_kwh} kWh | Cost Savings: INR {res.cost_savings_inr}")
    print(
        f"Candidates Evaluated: {res.total_candidates_evaluated} (Feasible: {res.feasible_candidates_count}, "
        f"Rejected: {res.rejected_candidates_count}) in {res.execution_time_ms} ms"
    )

    report_path = save_optimization_report(res)
    print(f"Sample report exported to: {report_path.resolve()}")
