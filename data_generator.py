"""
AI Data Center Cooling Optimizer - Synthetic Dataset Generator
==============================================================
Inspired by Google DeepMind's data center cooling efficiency research.

This module generates a reproducible, physically plausible synthetic dataset
simulating data center environmental telemetry, IT workloads, cooling actuator
settings (fan and pump speeds), cooling power consumption, and Power Usage
Effectiveness (PUE).

Physical Modeling Assumptions:
------------------------------
1. Conservation of Energy:
   All electrical power consumed by IT equipment (servers, storage, network) is
   converted into thermal heat that must be continuously extracted:
   Q_thermal ≈ P_IT.

2. Actuator Affinity Laws:
   - Fan power scales non-linearly with fan speed according to turbomachinery
     affinity laws (P_fan ∝ Speed^2.7, accounting for VFD and motor efficiencies).
   - Chilled water pump power similarly follows pump affinity laws (P_pump ∝ Speed^2.5).

3. Chiller Thermodynamics & COP:
   Chiller electrical power is given by P_chiller = Q_thermal / COP.
   Coefficient of Performance (COP) degrades with higher ambient temperatures
   (higher condensing temperature) and improves with adequate chilled water flow.

4. Server Thermal Envelope (ASHRAE TC 9.9):
   Server inlet air temperature (cold aisle) is driven by ambient weather,
   IT workload intensity, and cooling delivery (airflow from CRAH fans and chilled
   water circulation). Higher fan and pump speeds suppress inlet temperatures.
   Server outlet temperature reflects convective heat uptake:
   T_outlet = T_inlet + ΔT_servers.

5. Power Usage Effectiveness (PUE):
   PUE = Total Facility Power / IT Equipment Power
       = (P_IT + P_cooling + P_ancillary) / P_IT.
   PUE is strictly positive and mathematically consistent across every record.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Default paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_DATASET_PATH = DEFAULT_DATA_DIR / "data_center_cooling.csv"

# Dataset feature column specification
DATASET_COLUMNS = [
    "timestamp",
    "ambient_temp_c",
    "humidity_pct",
    "it_load_kw",
    "inlet_temp_c",
    "outlet_temp_c",
    "fan_speed_pct",
    "pump_speed_pct",
    "cooling_power_kw",
    "total_facility_power_kw",
    "pue",
]


def generate_cooling_data(
    n_samples: int = 3500,
    random_seed: int = 42,
    start_date: str = "2025-01-01 00:00:00",
) -> pd.DataFrame:
    """Generate a reproducible, physically plausible synthetic data center cooling dataset.

    Args:
        n_samples: Number of hourly timestamps to simulate (2,000 to 5,000).
        random_seed: Random seed for reproducibility.
        start_date: Starting ISO timestamp for the time series.

    Returns:
        pd.DataFrame containing environmental and cooling metrics.
    """
    if n_samples < 500:
        raise ValueError("n_samples must be at least 500 for meaningful time-series simulation.")

    rng = np.random.default_rng(random_seed)

    # 1. Timestamps (hourly intervals)
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq="h")
    hour_of_day = timestamps.hour.values
    day_of_year = timestamps.dayofyear.values

    # 2. Ambient Temperature (°C)
    # Diurnal cycle peaks at ~14:00; seasonal cycle peaks in summer (around day 180-200)
    diurnal_temp = 6.5 * np.sin(2 * np.pi * (hour_of_day - 8) / 24.0)
    seasonal_temp = 8.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    temp_noise = rng.normal(0.0, 1.2, size=n_samples)
    ambient_temp = 22.0 + diurnal_temp + seasonal_temp + temp_noise
    ambient_temp = np.clip(ambient_temp, 10.0, 42.0)

    # 3. Relative Humidity (%)
    # Inversely correlated with ambient temperature (psychrometric diurnal cycle)
    diurnal_humidity = -15.0 * np.sin(2 * np.pi * (hour_of_day - 8) / 24.0)
    humidity_noise = rng.normal(0.0, 3.5, size=n_samples)
    humidity = 58.0 + diurnal_humidity + humidity_noise
    humidity = np.clip(humidity, 25.0, 92.0)

    # 4. IT Load (kW)
    # Enterprise workload pattern: daytime peak, nighttime valley, slight weekend dip
    is_weekend = (timestamps.dayofweek.values >= 5)
    diurnal_workload = 180.0 * np.sin(2 * np.pi * (hour_of_day - 6) / 24.0)
    weekend_reduction = np.where(is_weekend, -60.0, 0.0)
    workload_noise = rng.normal(0.0, 25.0, size=n_samples)
    it_load = 820.0 + diurnal_workload + weekend_reduction + workload_noise
    it_load = np.clip(it_load, 450.0, 1250.0)

    # 5. Baseline Actuator Settings (Fan & Pump Speeds in %)
    # Simulates conventional heuristic/reactive control:
    # Operators increase fan and pump speeds when ambient temperature and IT load rise.
    # Legacy operations run conservative overcooling margins.
    fan_speed_base = (
        46.0
        + 0.95 * (ambient_temp - 22.0)
        + 0.032 * (it_load - 800.0)
        + rng.normal(0.0, 2.5, size=n_samples)
    )
    fan_speed = np.clip(fan_speed_base, 35.0, 95.0)

    pump_speed_base = (
        48.0
        + 0.85 * (ambient_temp - 22.0)
        + 0.028 * (it_load - 800.0)
        + rng.normal(0.0, 2.5, size=n_samples)
    )
    pump_speed = np.clip(pump_speed_base, 35.0, 95.0)

    # 6. Server Inlet Temperature (°C) - Cold Aisle Air Delivered
    # Governed by heat balance: ambient temp and IT load push temperature up;
    # increased fan airflow and chilled water flow suppress inlet temperature.
    inlet_temp_base = (
        18.5
        + 0.14 * (ambient_temp - 22.0)
        + 0.0065 * (it_load - 800.0)
        - 0.10 * (fan_speed - 50.0)
        - 0.06 * (pump_speed - 50.0)
        + rng.normal(0.0, 0.20, size=n_samples)
    )
    # ASHRAE Recommended Class A1 is 18°C - 27°C
    inlet_temp = np.clip(inlet_temp_base, 17.5, 28.5)

    # 7. Server Outlet Temperature (°C) - Hot Aisle
    # Servers warm air passing across CPUs: ΔT_servers = P_IT / (m_dot * Cp)
    delta_t_servers = 10.5 + 0.0075 * (it_load - 800.0) - 0.035 * (fan_speed - 50.0)
    delta_t_servers = np.clip(delta_t_servers, 8.0, 16.0)
    outlet_temp = inlet_temp + delta_t_servers + rng.normal(0.0, 0.25, size=n_samples)

    # 8. Cooling Power Consumption (kW)
    # Physics breakdown:
    # - CRAH Fans: Affinity law power ~ (Speed / 100)^2.7 * Rated_Fan_Power (Rated = 85 kW)
    fan_power = 85.0 * np.power(fan_speed / 100.0, 2.7)

    # - Chilled Water Pumps: Affinity law power ~ (Speed / 100)^2.5 * Rated_Pump_Power (Rated = 50 kW)
    pump_power = 50.0 * np.power(pump_speed / 100.0, 2.5)

    # - Chiller Plant & Heat Rejection:
    #   P_chiller = Thermal_Load / COP
    #   COP decreases when ambient temp is high (higher condenser head) and humidity is high (wet-bulb lift)
    #   Adequate chilled water pumping helps maintain evaporator heat transfer
    cop = (
        5.1
        - 0.055 * (ambient_temp - 22.0)
        - 0.012 * (humidity - 50.0)
        + 0.015 * (pump_speed - 50.0)
        + rng.normal(0.0, 0.1, size=n_samples)
    )
    cop = np.clip(cop, 2.6, 6.2)
    chiller_power = it_load / cop

    cooling_power_base = fan_power + pump_power + chiller_power + rng.normal(0.0, 4.0, size=n_samples)
    cooling_power = np.clip(cooling_power_base, 80.0, 650.0)

    # 9. Total Facility Power (kW) and Power Usage Effectiveness (PUE)
    # Ancillary electrical overheads: lighting, UPS conversion losses, power distribution units
    ancillary_power = 0.035 * it_load + 18.0 + rng.normal(0.0, 1.0, size=n_samples)
    ancillary_power = np.clip(ancillary_power, 15.0, 80.0)

    total_facility_power = it_load + cooling_power + ancillary_power
    pue = total_facility_power / it_load

    # Construct clean, rounded DataFrame
    df = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "ambient_temp_c": np.round(ambient_temp, 2),
            "humidity_pct": np.round(humidity, 2),
            "it_load_kw": np.round(it_load, 2),
            "inlet_temp_c": np.round(inlet_temp, 2),
            "outlet_temp_c": np.round(outlet_temp, 2),
            "fan_speed_pct": np.round(fan_speed, 2),
            "pump_speed_pct": np.round(pump_speed, 2),
            "cooling_power_kw": np.round(cooling_power, 2),
            "total_facility_power_kw": np.round(total_facility_power, 2),
            "pue": np.round(pue, 4),
        }
    )

    return df


def save_dataset(df: pd.DataFrame, file_path: Optional[Path | str] = None) -> Path:
    """Save the generated DataFrame to a CSV file.

    Args:
        df: DataFrame to save.
        file_path: Destination path. Defaults to data/data_center_cooling.csv.

    Returns:
        Path to the saved CSV file.
    """
    destination = Path(file_path) if file_path else DEFAULT_DATASET_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return destination


def load_or_generate_dataset(
    file_path: Optional[Path | str] = None,
    force_regenerate: bool = False,
    n_samples: int = 3500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load the dataset if it exists, or generate and save it if absent or forced.

    Args:
        file_path: Path to check or create.
        force_regenerate: If True, overwrite existing dataset.
        n_samples: Number of rows if generating.
        random_seed: Seed for generation.

    Returns:
        Loaded or generated pd.DataFrame.
    """
    path = Path(file_path) if file_path else DEFAULT_DATASET_PATH
    if path.exists() and not force_regenerate:
        df = pd.read_csv(path)
        # Verify schema
        missing_cols = set(DATASET_COLUMNS) - set(df.columns)
        if not missing_cols:
            return df

    df = generate_cooling_data(n_samples=n_samples, random_seed=random_seed)
    save_dataset(df, path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data center cooling dataset.")
    parser.add_argument("--samples", type=int, default=3500, help="Number of rows to generate (default: 3500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Target CSV output path",
    )
    args = parser.parse_args()

    print(f"Generating synthetic cooling telemetry ({args.samples} samples, seed={args.seed})...")
    data = generate_cooling_data(n_samples=args.samples, random_seed=args.seed)
    out_path = save_dataset(data, args.output)
    print(f"Successfully generated and saved {len(data)} rows to {out_path.resolve()}")
    print("\nDataset Summary Preview:")
    print(data.describe().round(2))
