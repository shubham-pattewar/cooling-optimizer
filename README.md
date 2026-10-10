# AI Data Center Cooling Optimizer — A DeepMind-Inspired Case Study

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit Dashboard](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Research & Simulation Disclaimer:**  
> This project is an educational, reproducible simulation inspired by published research from **Google DeepMind** (2016 & 2018). Telemetry is generated via physics-based thermodynamic equations and evaluated using machine learning surrogate models. It does not connect to physical cooling equipment and does not reproduce Google's proprietary production control infrastructure. Model-predicted safety reflects statistical surrogate constraints, not certified physical safety.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Key Features](#key-features)
4. [System Architecture](#system-architecture)
5. [Technology Stack](#technology-stack)
6. [Dataset Schema & Physical Modeling Assumptions](#dataset-schema--physical-modeling-assumptions)
7. [Machine Learning Methodology](#machine-learning-methodology)
8. [Constraint-Aware Optimization Engine](#constraint-aware-optimization-engine)
9. [Installation & Setup (Windows PowerShell)](#installation--setup-windows-powershell)
10. [Quickstart Commands](#quickstart-commands)
11. [Running the Application](#running-the-application)
12. [Automated Testing](#automated-testing)
13. [Example Simulation Results](#example-simulation-results)
14. [Limitations & Future Roadmap](#limitations--future-roadmap)
15. [References & Citations](#references--citations)

---

## 1. Project Overview

Cloud computing hyperscalers, artificial intelligence clusters, and modern enterprise data centers consume immense quantities of electrical power. Thousands of servers operating continuously transform electrical energy into thermal heat that must be dissipated. Historically, cooling infrastructure—including Computer Room Air Handlers (CRAH), chillers, cooling towers, and circulation pumps—accounts for **30% to 40%** of total data center power consumption.

Inspired by Google DeepMind's seminal breakthroughs in data center cooling efficiency, this project implements an end-to-end machine learning system that:
- Models complex, non-linear thermodynamic interactions between weather, computational workload, and cooling actuators.
- Employs surrogate machine learning regressors to forecast cooling power demand and server cold-aisle inlet temperatures without target leakage.
- Implements a constraint-aware candidate search engine that discovers energy-efficient actuator setpoints while strictly respecting server thermal envelopes (ASHRAE TC 9.9 guidelines).
- Delivers an interactive, professional web application built in Streamlit featuring executive KPI telemetry, multi-dimensional sensor exploration, diagnostic model evaluation, and financial impact analyses.

---

## 2. Problem Statement

Data center cooling systems are notoriously difficult to optimize with conventional rule-based or static heuristics because:
1. **Multi-Variable Non-Linearity**: Chiller Coefficient of Performance (COP), turbomachinery fan affinity laws, and psychrometric wet-bulb temperatures interact in non-linear ways.
2. **Lagging Dynamics**: Adjusting fan or pump speed does not instantaneously change server temperatures throughout large computer rooms.
3. **Conservative Overcooling**: Plant operators routinely run fans and pumps at excessively high speeds to prevent thermal hotspots, wasting megawatts of electricity.
4. **Safety-Critical Constraints**: Lowering cooling power must never push cold aisle server inlet air beyond safe operating boundaries ($T_{inlet} \le 24.5^\circ\text{C}$), which could trigger server CPU thermal throttling or hardware failure.

**Objective**: Find actuator setpoints $(Fan^*, Pump^*)$ that minimize total cooling power $P_{cooling}$ subject to:
$$\text{Minimize } P_{cooling}(T_{amb}, \text{Hum}, P_{IT}, Fan, Pump)$$
$$\text{Subject to: } T_{inlet}(T_{amb}, \text{Hum}, P_{IT}, Fan, Pump) \le T_{max}$$
$$Fan_{min} \le Fan \le Fan_{max}, \quad Pump_{min} \le Pump \le Pump_{max}$$

---

## 3. Key Features

- **Executive Overview Dashboard**: Real-time KPI metric cards (IT Load, Cooling Power, PUE, Recommended Actuators, Energy Savings) and interactive Plotly visual analytics.
- **Cooling Optimizer Engine**: Interactive control panel with sliders for ambient temperature, humidity, IT load, current settings, and safety bounds. Evaluates a full grid of candidates and displays feasible vs. rejected configurations.
- **Sensor Data Explorer**: Inspect, search, and filter 3,500+ hourly telemetry records. View summary statistics, a correlation heatmap, and customize multi-variable plots.
- **ML Model Diagnostics**: Side-by-side evaluation of Random Forest Regressors against baseline dummy models (MAE, RMSE, $R^2$), predicted vs. actual scatter plots, residual distributions, and feature importance rankings.
- **Energy Savings & Financial Impact**: Compute kWh saved, commercial electricity bill reductions (INR ₹), and facility power stack breakdowns over configurable operating horizons (24h, 1 week, 1 month, 1 year).
- **DeepMind Case Study**: Educational narrative detailing Google DeepMind's 2016 recommendation deployment, 2018 autonomous closed-loop control system, safety-first architecture, and historical timeline.
- **Full Reproducibility**: 100% self-contained synthetic dataset generator, persisted Scikit-learn models, and comprehensive Pytest test suite.

---

## 4. System Architecture

```
ai-data-center-cooling-optimizer/
│
├── app.py                      # Main Streamlit web application & 7 multi-page routers
├── data_generator.py           # Physics-based synthetic telemetry generator
├── model_trainer.py            # Random Forest surrogate training & evaluation pipeline
├── optimizer.py                # Constraint-aware candidate search engine
├── utils.py                    # Design system, CSS tokens, and Plotly theme utilities
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation, setup guide, and case study notes
├── .gitignore                  # Git exclusions for models, datasets, and cache
│
├── data/                       # Telemetry storage
│   ├── .gitkeep
│   └── data_center_cooling.csv # Generated dataset (3,500 rows)
│
├── models/                     # Persisted models & metadata
│   ├── .gitkeep
│   ├── cooling_power_model.joblib
│   ├── inlet_temp_model.joblib
│   ├── cooling_power_baseline.joblib
│   ├── inlet_temp_baseline.joblib
│   └── model_metadata.json
│
├── reports/                    # Optimization exports
│   └── .gitkeep
│
└── tests/                      # Automated test suite
    ├── test_data_generator.py  # Tests for physical ranges, schema, and PUE
    ├── test_model_trainer.py   # Tests for leak-free features and model performance
    ├── test_optimizer.py       # Tests for actuator limits, thermal rejection, and reports
    └── test_app.py             # Tests for Streamlit page renderers and model loaders
```

---

## 5. Technology Stack

| Component | Library / Framework | Purpose |
|---|---|---|
| **Language** | Python 3.11+ / 3.13 | Core application runtime |
| **User Interface** | Streamlit | Responsive, professional multi-page web application |
| **Machine Learning** | Scikit-learn | Random Forest surrogate regression and baseline comparison |
| **Numerical Processing** | Pandas & NumPy | Data manipulation, time-series generation, and telemetry filtering |
| **Interactive Visuals** | Plotly (Graph Objects & Express) | Clean, responsive charting with DeepMind/Google Cloud aesthetics |
| **Model Persistence** | Joblib | Serialization and loading of trained estimators |
| **Testing** | Pytest | Automated verification of physical boundaries and logic |

---

## 6. Dataset Schema & Physical Modeling Assumptions

### Feature Column Specification

| Column Name | Units | Range | Physical Meaning |
|---|---|---|---|
| `timestamp` | ISO String | Hourly | Chronological simulation timestamp |
| `ambient_temp_c` | °C | 10.0 – 42.0 | Outdoor dry-bulb air temperature |
| `humidity_pct` | % | 25.0 – 92.0 | Outdoor relative humidity |
| `it_load_kw` | kW | 450.0 – 1250.0 | Total electrical compute power consumed by servers |
| `inlet_temp_c` | °C | 17.5 – 28.5 | Cold aisle air temperature entering server chassis |
| `outlet_temp_c` | °C | 25.0 – 42.0 | Hot aisle air temperature leaving server chassis |
| `fan_speed_pct` | % | 35.0 – 95.0 | CRAH variable frequency drive fan speed |
| `pump_speed_pct` | % | 35.0 – 95.0 | Chilled water loop circulation pump speed |
| `cooling_power_kw` | kW | 80.0 – 650.0 | Combined power consumed by fans, pumps, and chillers |
| `total_facility_power_kw` | kW | 550.0 – 1950.0 | Sum of IT load, cooling power, and ancillary overheads |
| `pue` | Ratio | 1.15 – 1.65 | Power Usage Effectiveness ($\text{Total Power} / \text{IT Load}$) |

### Core Physical Modeling Principles:
1. **Conservation of Thermal Energy**: Server compute power is converted into thermal heat that must be extracted: $Q_{thermal} \approx P_{IT}$.
2. **Turbomachinery Affinity Laws**:
   - CRAH Fan Power: $P_{fan} = 85.0 \times \left(\frac{\text{Speed}}{100}\right)^{2.7} \text{ kW}$
   - Pump Power: $P_{pump} = 50.0 \times \left(\frac{\text{Speed}}{100}\right)^{2.5} \text{ kW}$
3. **Chiller Plant Thermodynamics & COP**:
   $$P_{chiller} = \frac{P_{IT}}{\text{COP}}$$
   where $\text{COP} = 5.1 - 0.055(T_{amb} - 22) - 0.012(\text{Hum} - 50) + 0.015(\text{Pump} - 50) + \epsilon$.
4. **Server Cold Aisle Heat Balance (ASHRAE TC 9.9 Class A1)**:
   $$T_{inlet} = 18.5 + 0.14(T_{amb} - 22) + 0.0065(P_{IT} - 800) - 0.10(\text{Fan} - 50) - 0.06(\text{Pump} - 50) + \epsilon$$
5. **Power Usage Effectiveness (PUE)**:
   $$\text{PUE} = \frac{P_{IT} + P_{cooling} + P_{ancillary}}{P_{IT}}$$

---

## 7. Machine Learning Methodology

The project trains two separate surrogate regression models:

### 1. Model 1: Cooling Power Predictor (`cooling_power_kw`)
- **Inputs**: `[ambient_temp_c, humidity_pct, it_load_kw, fan_speed_pct, pump_speed_pct]`
- **Primary Algorithm**: `RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_split=4, random_state=42)`
- **Evaluation Baseline**: `DummyRegressor(strategy='mean')`
- **Performance**:
  - Random Forest: MAE = 4.45 kW, RMSE = 5.66 kW, $R^2 = 0.976$
  - Dummy Baseline: MAE = 31.37 kW, RMSE = 36.39 kW, $R^2 = -0.003$

### 2. Model 2: Server Inlet Temperature Predictor (`inlet_temp_c`)
- **Inputs**: Identical 5-feature schema (strict absence of target leakage).
- **Primary Algorithm**: `RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_split=4, random_state=42)`
- **Evaluation Baseline**: `DummyRegressor(strategy='mean')`
- **Performance**:
  - Random Forest: MAE = 0.20 °C, RMSE = 0.25 °C, $R^2 = 0.723$
  - Dummy Baseline: MAE = 0.38 °C, RMSE = 0.48 °C, $R^2 = -0.001$

---

## 8. Constraint-Aware Optimization Engine

The optimizer in `optimizer.py` executes candidate search:
1. **Baseline Evaluation**: Predicts cooling power and inlet temperature for the operator's current fan and pump setpoints.
2. **Candidate Generation**: Builds a 2D actuator search grid spanning $[Fan_{min}, Fan_{max}] \times [Pump_{min}, Pump_{max}]$ with configurable step size (default: 2.5%).
3. **Surrogate Batch Inference**: Evaluates all candidate configurations under the exact same ambient conditions and IT workload.
4. **Safety Constraint Filtering**:
   - Rejects any candidate whose predicted inlet temperature exceeds $T_{max}$ (e.g., $24.5^\circ\text{C}$).
   - Documents specific rejection reasons.
5. **Minimum Energy Selection**: Selects the feasible candidate with the lowest predicted cooling power.
6. **Fallback Mechanism**: If no candidate satisfies the thermal constraints or improves upon baseline, the system retains the baseline settings and flags an alert.
7. **Savings Calculation**:
   $$\text{Savings \%} = \frac{P_{baseline} - P_{optimized}}{P_{baseline}} \times 100$$
   $$\text{Financial Savings (INR)} = \Delta P_{cooling} \times \text{Hours} \times \text{Tariff Rate}$$

---

## 9. Installation & Setup (Windows PowerShell)

Open Windows PowerShell in the project directory:

```powershell
# 1. Navigate to the project root
cd c:\Users\patte\Downloads\DATACENTER

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 4. Install project dependencies
pip install -r requirements.txt
```

---

## 10. Quickstart Commands

```powershell
# Step 1: Generate synthetic telemetry dataset (3,500 hourly samples)
python data_generator.py

# Step 2: Train Random Forest surrogate models and save artifacts
python model_trainer.py

# Step 3: Run the constraint-aware optimizer test
python optimizer.py

# Step 4: Execute automated Pytest test suite
python -m pytest -v

# Step 5: Launch the interactive Streamlit dashboard
streamlit run app.py
```

---

## 11. Running the Application

Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```
Streamlit will automatically open your default browser to `http://localhost:8501`.

### Navigating the Multi-Page Application:
- **Overview Dashboard**: High-level KPIs, 7-day time series, ambient vs. power scatter plots, and baseline vs. AI comparisons.
- **Cooling Optimizer**: Interactive slider panel to test different weather conditions, workloads, and actuator bounds. Inspect feasible vs. rejected candidate tables and export reports.
- **Sensor Data Explorer**: Filter records by temperature, IT load, or power. Inspect statistical summaries and correlation heatmaps. Regenerate datasets on-demand.
- **ML Model Evaluation**: Review held-out test set metrics, actual vs. predicted diagnostics, residual plots, and feature importances.
- **Energy Savings Analysis**: Calculate kWh and INR cost savings over 24h, 1 week, 1 month, or 1 year with custom electricity tariffs.
- **Case Study: Google DeepMind**: Read an educational history of DeepMind's 2016 and 2018 data center cooling deployments.
- **About the Project**: System architecture, physics formulas, and engineering details.

---

## 12. Automated Testing

The automated test suite in `tests/` verifies data integrity, model behavior, safety boundaries, and app stability:

```powershell
python -m pytest -v
```

### Verified Test Cases:
- `test_dataset_generation_shape`: Validates row and column counts.
- `test_no_missing_or_infinite_values`: Confirms zero nulls, NaNs, or infinite entries.
- `test_physical_ranges`: Ensures positive IT power, positive cooling power, bounded speeds, and consistent PUE definitions ($PUE \ge 1.0$).
- `test_dataset_reproducibility`: Confirms identical random seeds produce identical datasets.
- `test_load_or_generate_dataset`: Validates dataset caching and forced regeneration.
- `test_feature_schema_consistency`: Verifies strict absence of target leakage.
- `test_train_models_pipeline`: Validates end-to-end training and confirms Random Forest outperforms naive baselines.
- `test_compute_pue_consistency`: Checks PUE formula correctness and zero-division handling.
- `test_optimizer_feasible_reduction`: Ensures valid cooling energy reductions under standard conditions.
- `test_optimizer_rejects_unsafe_thermal_condition`: Confirms unsafe candidates are rejected and baseline is retained.
- `test_optimizer_actuator_limits`: Verifies candidate speeds strictly adhere to bounds.
- `test_savings_calculation_safe_zero_division`: Verifies robust zero-division safeguards.
- `test_report_export`: Verifies optimization report CSV export and column schema.
- `test_app_pages_exist`: Verifies all 7 page renderers exist and can be called.
- `test_app_dataset_and_models_cache`: Verifies data and model caching functions.

---

## 13. Example Simulation Results

Under typical operating conditions:
- **Ambient Weather**: $28.0^\circ\text{C}$, $55\%$ Relative Humidity
- **IT Workload**: $850.0\text{ kW}$
- **Baseline Actuators**: Fan Speed = $80\%$, Pump Speed = $75\%$
- **Baseline Cooling Power**: $218.4\text{ kW}$ (Inlet Temp: $20.8^\circ\text{C}$, PUE: $1.286$)
- **AI Recommended Actuators**: Fan Speed = $62.5\%$, Pump Speed = $57.5\%$
- **Optimized Cooling Power**: $178.6\text{ kW}$ (Inlet Temp: $22.7^\circ\text{C}$, PUE: $1.239$)
- **Power Reduction**: **$39.8\text{ kW}$ ($18.2\%$ cooling energy savings)**
- **Cold Aisle Status**: $22.7^\circ\text{C} \le 24.5^\circ\text{C}$ (Safe, well within ASHRAE Recommended Envelope)
- **Estimated 24h Energy Saved**: $955.2\text{ kWh}$
- **Estimated 24h Financial Savings**: **₹7,164 INR** (@ ₹7.50/kWh commercial tariff)

---

## 14. Limitations & Future Roadmap

1. **Simulation vs. Physical Actuation**: This project uses mathematical approximations. Real physical plants exhibit airflow recirculation, duct friction, pump cavitation, and sensor calibration drift.
2. **Dynamic Workload Transitions**: The candidate search optimizes static snapshots. Future versions can incorporate Model Predictive Control (MPC) or Reinforcement Learning (RL) to optimize across dynamic multi-hour lookaheads.
3. **Multi-Chiller Staging**: The current model represents a consolidated cooling plant. Expanding to multi-chiller lead/lag sequencing and chilled water delta-T management would add further fidelity.

---

## 15. References & Citations

1. **Google DeepMind (2016)**:  
   *DeepMind AI Reduces Google Data Centre Cooling Bill by 40%*  
   [https://deepmind.google/discover/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-by-40/](https://deepmind.google/discover/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-by-40/)
2. **Google DeepMind (2018)**:  
   *Safety-first AI for autonomous data centre cooling and industrial control*  
   [https://deepmind.google/blog/safety-first-ai-for-autonomous-data-centre-cooling-and-industrial-control/](https://deepmind.google/blog/safety-first-ai-for-autonomous-data-centre-cooling-and-industrial-control/)
3. **ASHRAE Technical Committee 9.9**:  
   *Thermal Guidelines for Data Processing Environments (5th Edition)*, American Society of Heating, Refrigerating and Air-Conditioning Engineers.
4. **The Green Grid**:  
   *Power Usage Effectiveness (PUE) Metric Specification and Usage Guidelines*.

---

*AI Data Center Cooling Optimizer — Educational Case Study for Engineering, Machine Learning, and Sustainability.*
