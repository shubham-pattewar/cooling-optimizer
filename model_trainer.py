"""
AI Data Center Cooling Optimizer - Machine Learning Model Trainer
================================================================
Trains constraint-aware surrogate regression models inspired by DeepMind's
data center cooling research:

1. Cooling Power Predictor:
   Surrogate model for predicting cooling power consumption (cooling_power_kw)
   given ambient weather conditions, IT workload, and cooling actuator controls.
   Primary: RandomForestRegressor.
   Baseline: DummyRegressor (predicts mean of training set).

2. Server Inlet Temperature Predictor:
   Surrogate model for predicting server inlet air temperature (inlet_temp_c)
   to ensure cold aisle conditions remain within safe thermal thresholds.
   Uses the exact same input feature schema without target leakage.
   Primary: RandomForestRegressor.
   Baseline: DummyRegressor (predicts mean of training set).
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "data_center_cooling.csv"
MODELS_DIR = BASE_DIR / "models"
POWER_MODEL_PATH = MODELS_DIR / "cooling_power_model.joblib"
TEMP_MODEL_PATH = MODELS_DIR / "inlet_temp_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Strict Feature Schema used for both training and inference
FEATURE_COLUMNS: List[str] = [
    "ambient_temp_c",
    "humidity_pct",
    "it_load_kw",
    "fan_speed_pct",
    "pump_speed_pct",
]

TARGET_POWER: str = "cooling_power_kw"
TARGET_TEMP: str = "inlet_temp_c"


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R2 regression performance metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def train_models(
    data_path: Optional[Path | str] = None,
    models_dir: Optional[Path | str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train Random Forest and Baseline models for cooling power and inlet temperature.

    Args:
        data_path: Path to input CSV dataset.
        models_dir: Directory where trained models and metadata are saved.
        test_size: Fraction of data for held-out testing.
        random_state: Random state for reproducibility.

    Returns:
        Dictionary containing metadata, performance metrics, and evaluation samples.
    """
    csv_path = Path(data_path) if data_path else DATA_PATH
    out_dir = Path(models_dir) if models_dir else MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Please generate it first using data_generator.py."
        )

    df = pd.read_csv(csv_path)

    # Validate feature and target columns exist
    required_cols = set(FEATURE_COLUMNS + [TARGET_POWER, TARGET_TEMP])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Prepare features and targets
    X = df[FEATURE_COLUMNS].copy()
    y_power = df[TARGET_POWER].values
    y_temp = df[TARGET_TEMP].values

    # Train-test split
    (
        X_train,
        X_test,
        y_power_train,
        y_power_test,
        y_temp_train,
        y_temp_test,
    ) = train_test_split(
        X,
        y_power,
        y_temp,
        test_size=test_size,
        random_state=random_state,
    )

    # 1. Model 1: Cooling Power Predictor
    rf_power = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_split=4,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_power.fit(X_train, y_power_train)

    dummy_power = DummyRegressor(strategy="mean")
    dummy_power.fit(X_train, y_power_train)

    y_power_pred_rf = rf_power.predict(X_test)
    y_power_pred_dummy = dummy_power.predict(X_test)

    power_metrics_rf = compute_regression_metrics(y_power_test, y_power_pred_rf)
    power_metrics_dummy = compute_regression_metrics(y_power_test, y_power_pred_dummy)

    # Feature Importance for Power
    power_importance = dict(
        zip(FEATURE_COLUMNS, [round(float(v), 4) for v in rf_power.feature_importances_])
    )

    # 2. Model 2: Server Inlet Temperature Predictor
    rf_temp = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_split=4,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_temp.fit(X_train, y_temp_train)

    dummy_temp = DummyRegressor(strategy="mean")
    dummy_temp.fit(X_train, y_temp_train)

    y_temp_pred_rf = rf_temp.predict(X_test)
    y_temp_pred_dummy = dummy_temp.predict(X_test)

    temp_metrics_rf = compute_regression_metrics(y_temp_test, y_temp_pred_rf)
    temp_metrics_dummy = compute_regression_metrics(y_temp_test, y_temp_pred_dummy)

    # Feature Importance for Temperature
    temp_importance = dict(
        zip(FEATURE_COLUMNS, [round(float(v), 4) for v in rf_temp.feature_importances_])
    )

    # Persist Models
    joblib.dump(rf_power, out_dir / "cooling_power_model.joblib")
    joblib.dump(rf_temp, out_dir / "inlet_temp_model.joblib")
    joblib.dump(dummy_power, out_dir / "cooling_power_baseline.joblib")
    joblib.dump(dummy_temp, out_dir / "inlet_temp_baseline.joblib")

    # Save a subset of test evaluation points for UI visualization
    sample_indices = np.linspace(0, len(X_test) - 1, min(300, len(X_test)), dtype=int)
    eval_samples = {
        "actual_power": [round(float(y_power_test[i]), 2) for i in sample_indices],
        "pred_power_rf": [round(float(y_power_pred_rf[i]), 2) for i in sample_indices],
        "pred_power_dummy": [round(float(y_power_pred_dummy[i]), 2) for i in sample_indices],
        "actual_temp": [round(float(y_temp_test[i]), 2) for i in sample_indices],
        "pred_temp_rf": [round(float(y_temp_pred_rf[i]), 2) for i in sample_indices],
        "pred_temp_dummy": [round(float(y_temp_pred_dummy[i]), 2) for i in sample_indices],
    }

    metadata: Dict[str, Any] = {
        "feature_columns": FEATURE_COLUMNS,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_size": test_size,
        "random_state": random_state,
        "training_timestamp": datetime.datetime.now().isoformat(),
        "cooling_power_metrics": {
            "random_forest": power_metrics_rf,
            "dummy_baseline": power_metrics_dummy,
        },
        "cooling_power_importance": power_importance,
        "inlet_temp_metrics": {
            "random_forest": temp_metrics_rf,
            "dummy_baseline": temp_metrics_dummy,
        },
        "inlet_temp_importance": temp_importance,
        "eval_samples": eval_samples,
    }

    with open(out_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_trained_models(
    models_dir: Optional[Path | str] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load persisted models and metadata from disk.

    Returns:
        (cooling_power_model, inlet_temp_model, metadata_dict)
    """
    directory = Path(models_dir) if models_dir else MODELS_DIR
    power_path = directory / "cooling_power_model.joblib"
    temp_path = directory / "inlet_temp_model.joblib"
    meta_path = directory / "model_metadata.json"

    if not power_path.exists() or not temp_path.exists():
        raise FileNotFoundError(
            f"Trained models not found in {directory}. Please run model_trainer.py first."
        )

    power_model = joblib.load(power_path)
    temp_model = joblib.load(temp_path)

    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return power_model, temp_model, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cooling optimizer ML models.")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to training CSV")
    parser.add_argument("--out", type=str, default=str(MODELS_DIR), help="Output models directory")
    args = parser.parse_args()

    print("Initiating ML training pipeline for AI Data Center Cooling...")
    meta = train_models(data_path=args.data, models_dir=args.out)

    print("\n--- Model 1: Cooling Power Predictor (kW) ---")
    rf_pm = meta["cooling_power_metrics"]["random_forest"]
    bl_pm = meta["cooling_power_metrics"]["dummy_baseline"]
    print(f"Random Forest  -> MAE: {rf_pm['mae']} kW | RMSE: {rf_pm['rmse']} kW | R²: {rf_pm['r2']}")
    print(f"Dummy Baseline -> MAE: {bl_pm['mae']} kW | RMSE: {bl_pm['rmse']} kW | R²: {bl_pm['r2']}")

    print("\n--- Model 2: Server Inlet Temperature Predictor (°C) ---")
    rf_tm = meta["inlet_temp_metrics"]["random_forest"]
    bl_tm = meta["inlet_temp_metrics"]["dummy_baseline"]
    print(f"Random Forest  -> MAE: {rf_tm['mae']} °C | RMSE: {rf_tm['rmse']} °C | R²: {rf_tm['r2']}")
    print(f"Dummy Baseline -> MAE: {bl_tm['mae']} °C | RMSE: {bl_tm['rmse']} °C | R²: {bl_tm['r2']}")

    print(f"\nTrained models and metadata saved successfully to: {MODELS_DIR.resolve()}")
