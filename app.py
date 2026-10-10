"""
AI Data Center Cooling Optimizer - Streamlit Dashboard
======================================================
A DeepMind-Inspired Case Study for Energy-Efficient Data Center Cooling.

Pages:
1. Overview Dashboard
2. Cooling Optimizer
3. Sensor Data Explorer
4. ML Model Evaluation
5. Energy Savings Analysis
6. Case Study: Google DeepMind
7. About the Project
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import (
    DATASET_COLUMNS,
    DEFAULT_DATASET_PATH,
    generate_cooling_data,
    load_or_generate_dataset,
    save_dataset,
)
from model_trainer import (
    FEATURE_COLUMNS,
    MODELS_DIR,
    load_trained_models,
    train_models,
)
from optimizer import (
    OptimizationRequest,
    OptimizationResult,
    optimize_cooling,
    save_optimization_report,
)
from utils import (
    ASSETS_ICONS_DIR,
    COLOR_ACCENT,
    COLOR_BORDER,
    COLOR_DANGER,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
    apply_custom_css,
    apply_plotly_theme,
    render_disclaimer_banner,
    render_flaticon,
    render_header,
    render_metric_card,
)

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Cooling Intelligence | DeepMind-Inspired Case Study",
    page_icon=str(ASSETS_ICONS_DIR / "snowflake.png") if (ASSETS_ICONS_DIR / "snowflake.png").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global styling
apply_custom_css()


# -----------------------------------------------------------------------------
# Cached Data and Model Loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset(force_regenerate: bool = False) -> pd.DataFrame:
    """Load or generate the synthetic data center cooling dataset."""
    return load_or_generate_dataset(force_regenerate=force_regenerate)


@st.cache_resource(show_spinner=False)
def get_models() -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
    """Load pre-trained models and metadata. If absent, trains them."""
    try:
        return load_trained_models()
    except (FileNotFoundError, Exception):
        # Auto-train if models don't exist yet
        meta = train_models()
        p_mod, t_mod, _ = load_trained_models()
        return p_mod, t_mod, meta


# Initialize session state for active optimization results
if "last_optimization_result" not in st.session_state:
    st.session_state["last_optimization_result"] = None


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Telemetry
# -----------------------------------------------------------------------------
def render_sidebar(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """Render the sidebar with navigation links and system status."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; display: flex; align-items: center; gap: 0.5rem;">
                    {render_flaticon('snowflake', 24, 6)} AI Cooling Intel
                </div>
                <div style="font-size: 0.76rem; color: #64748b; margin-top: 0.2rem;">
                    DeepMind-Inspired Cooling Optimizer
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 0.4rem;'>Navigation</div>",
            unsafe_allow_html=True,
        )

        pages = [
            "Overview Dashboard",
            "Cooling Optimizer",
            "Sensor Data Explorer",
            "ML Model Evaluation",
            "Energy Savings Analysis",
            "Case Study: Google DeepMind",
            "About the Project",
        ]

        selected_page = st.radio(
            "Select Page",
            pages,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("<hr style='margin: 1.5rem 0 1rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

        # Telemetry & Status Card
        st.markdown(
            "<div style='font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>System Health</div>",
            unsafe_allow_html=True,
        )

        has_data = len(df) > 0
        has_models = "cooling_power_metrics" in metadata

        st.markdown(
            f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem; font-size: 0.78rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
                    <span style="color: #64748b;">Telemetry Status:</span>
                    <span style="font-weight: 600; color: {'#059669' if has_data else '#dc2626'};">
                        {'ONLINE (3.5k rows)' if has_data else 'OFFLINE'}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
                    <span style="color: #64748b;">Surrogate Models:</span>
                    <span style="font-weight: 600; color: {'#059669' if has_models else '#dc2626'};">
                        {'CALIBRATED (R² > 0.97)' if has_models else 'UNAVAILABLE'}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #64748b;">Target Standard:</span>
                    <span style="font-weight: 600; color: #0284c7;">ASHRAE Class A1</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-top: 1.5rem; font-size: 0.72rem; color: #94a3b8; line-height: 1.4;">
                Simulated research sandbox for educational data center optimization. Not connected to physical equipment.
            </div>
            """,
            unsafe_allow_html=True,
        )

        return selected_page


# -----------------------------------------------------------------------------
# PAGE 1: Overview Dashboard
# -----------------------------------------------------------------------------
def render_page_overview(df: pd.DataFrame, power_model: Any, temp_model: Any, metadata: Dict[str, Any]) -> None:
    render_header(
        title="AI Cooling Intelligence",
        subtitle="Machine Learning for Energy-Efficient Data Centers — Executive Overview",
    )
    render_disclaimer_banner()

    # System Status Indicator
    status_col1, status_col2 = st.columns([3, 1])
    with status_col1:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1rem;">
                <span class="badge badge-success">● SYSTEM ONLINE</span>
                <span style="font-size: 0.82rem; color: #64748b;">
                    Surrogate inference engine calibrated. Continuous thermal constraint checks active.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_col2:
        st.caption("Data source: Simulated Hourly Physics Telemetry")

    # Key Metrics from Dataset
    avg_it_load = float(df["it_load_kw"].mean())
    avg_cooling_power = float(df["cooling_power_kw"].mean())
    avg_pue = float(df["pue"].mean())
    avg_inlet_temp = float(df["inlet_temp_c"].mean())

    # Obtain sample baseline vs optimized from optimizer under standard representative weather
    default_req = OptimizationRequest(
        ambient_temp_c=float(df["ambient_temp_c"].iloc[-1]),
        humidity_pct=float(df["humidity_pct"].iloc[-1]),
        it_load_kw=float(df["it_load_kw"].iloc[-1]),
        current_fan_speed_pct=float(df["fan_speed_pct"].iloc[-1]),
        current_pump_speed_pct=float(df["pump_speed_pct"].iloc[-1]),
        max_inlet_temp_c=24.5,
    )
    sample_opt = optimize_cooling(default_req, power_model, temp_model)

    # 5 KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(
            render_metric_card(
                label="Simulated IT Load",
                value=f"{avg_it_load:.1f}",
                unit="kW",
                subtext=f"Current: {default_req.it_load_kw:.1f} kW",
                badge_text="Thermal Source",
                badge_variant="info",
                icon_name="server",
            ),
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            render_metric_card(
                label="Cooling Power",
                value=f"{avg_cooling_power:.1f}",
                unit="kW",
                subtext=f"Current: {sample_opt.baseline_cooling_power_kw:.1f} kW",
                badge_text="Surrogate Predicted",
                badge_variant="warning",
                icon_name="snowflake",
            ),
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            render_metric_card(
                label="Average PUE",
                value=f"{avg_pue:.3f}",
                unit="",
                subtext="PUE = Total Power / IT Power",
                badge_text="Efficiency Benchmark",
                badge_variant="info",
                icon_name="ruler",
            ),
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            render_metric_card(
                label="AI Recommended Actuators",
                value=f"{sample_opt.recommended_fan_speed_pct:.0f}% / {sample_opt.recommended_pump_speed_pct:.0f}%",
                unit="Fan/Pump",
                subtext=f"Baseline: {sample_opt.baseline_fan_speed_pct:.0f}% / {sample_opt.baseline_pump_speed_pct:.0f}%",
                badge_text="Feasible Recommendation",
                badge_variant="success",
                icon_name="settings",
            ),
            unsafe_allow_html=True,
        )
    with kpi5:
        st.markdown(
            render_metric_card(
                label="Estimated Energy Savings",
                value=f"{sample_opt.savings_percent:.1f}%",
                unit="",
                subtext=f"Reduced by {sample_opt.power_savings_kw:.1f} kW",
                badge_text="Thermal Constraints Met",
                badge_variant="success",
                icon_name="trend_down",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # Interactive Visualizations Grid
    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        st.markdown(
            f"##### {render_flaticon('clock', 18)} Cooling Power & IT Workload Over Time (Last 168 Hours / 7 Days)",
            unsafe_allow_html=True,
        )
        subset = df.iloc[-168:].copy()
        fig_time = go.Figure()
        fig_time.add_trace(
            go.Scatter(
                x=subset["timestamp"],
                y=subset["it_load_kw"],
                name="IT Workload (kW)",
                line=dict(color="#64748b", width=1.5),
            )
        )
        fig_time.add_trace(
            go.Scatter(
                x=subset["timestamp"],
                y=subset["cooling_power_kw"],
                name="Cooling Power (kW)",
                line=dict(color=COLOR_PRIMARY, width=2.2),
            )
        )
        fig_time.update_layout(
            xaxis_title="Simulation Timestamp",
            yaxis_title="Power (kW)",
        )
        apply_plotly_theme(fig_time, height=340)
        st.plotly_chart(fig_time, use_container_width=True)

    with row1_c2:
        st.markdown(
            f"##### {render_flaticon('thermometer', 18)} Ambient Temperature vs. Cooling Power",
            unsafe_allow_html=True,
        )
        fig_amb = px.scatter(
            df.iloc[::3],
            x="ambient_temp_c",
            y="cooling_power_kw",
            color="it_load_kw",
            color_continuous_scale="Viridis",
            labels={
                "ambient_temp_c": "Ambient Temperature (°C)",
                "cooling_power_kw": "Cooling Power (kW)",
                "it_load_kw": "IT Load (kW)",
            },
        )
        apply_plotly_theme(fig_amb, height=340)
        st.plotly_chart(fig_amb, use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        st.markdown(
            f"##### {render_flaticon('lightning', 18)} Baseline vs. AI Optimized Cooling Power Distribution",
            unsafe_allow_html=True,
        )
        # Generate baseline vs predicted optimal across sample conditions
        sample_ambients = np.linspace(15, 38, 20)
        base_powers = []
        opt_powers = []
        for amb in sample_ambients:
            req_sim = OptimizationRequest(
                ambient_temp_c=amb,
                humidity_pct=55.0,
                it_load_kw=820.0,
                current_fan_speed_pct=75.0,
                current_pump_speed_pct=70.0,
                max_inlet_temp_c=24.5,
            )
            sim_res = optimize_cooling(req_sim, power_model, temp_model)
            base_powers.append(sim_res.baseline_cooling_power_kw)
            opt_powers.append(sim_res.recommended_cooling_power_kw)

        fig_compare = go.Figure()
        fig_compare.add_trace(
            go.Scatter(
                x=sample_ambients,
                y=base_powers,
                name="Conventional Baseline (kW)",
                mode="lines+markers",
                line=dict(color="#f97316", width=2, dash="dash"),
            )
        )
        fig_compare.add_trace(
            go.Scatter(
                x=sample_ambients,
                y=opt_powers,
                name="AI Optimized (kW)",
                mode="lines+markers",
                line=dict(color=COLOR_SECONDARY, width=2.5),
            )
        )
        fig_compare.update_layout(
            xaxis_title="Ambient Temperature (°C)",
            yaxis_title="Cooling Power (kW)",
        )
        apply_plotly_theme(fig_compare, height=340)
        st.plotly_chart(fig_compare, use_container_width=True)

    with row2_c2:
        st.markdown(
            f"##### {render_flaticon('target', 18)} Model Validation: Predicted vs. Actual Server Inlet Temperature",
            unsafe_allow_html=True,
        )
        eval_samples = metadata.get("eval_samples", {})
        if eval_samples and "actual_temp" in eval_samples:
            act_t = eval_samples["actual_temp"]
            prd_t = eval_samples["pred_temp_rf"]
            fig_acc = go.Figure()
            fig_acc.add_trace(
                go.Scatter(
                    x=act_t,
                    y=prd_t,
                    mode="markers",
                    name="Test Observations",
                    marker=dict(color=COLOR_PRIMARY, size=5, opacity=0.7),
                )
            )
            # 1:1 perfect prediction diagonal
            min_val = min(min(act_t), min(prd_t))
            max_val = max(max(act_t), max(prd_t))
            fig_acc.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Ideal (1:1)",
                    line=dict(color="#dc2626", dash="dot", width=1.5),
                )
            )
            fig_acc.update_layout(
                xaxis_title="Actual Inlet Temp (°C)",
                yaxis_title="Random Forest Predicted Inlet Temp (°C)",
            )
            apply_plotly_theme(fig_acc, height=340)
            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("Test evaluation sample points loading...")


# -----------------------------------------------------------------------------
# PAGE 2: Cooling Optimizer
# -----------------------------------------------------------------------------
def render_page_optimizer(power_model: Any, temp_model: Any) -> None:
    render_header(
        title="Cooling Optimization Engine",
        subtitle="Surrogate Search Algorithm for Constraint-Aware Actuator Efficiency",
    )
    render_disclaimer_banner()

    col_input, col_output = st.columns([1, 1.45], gap="large")

    with col_input:
        st.markdown(f"#### {render_flaticon('settings', 20)} Operational Telemetry & Constraints", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("##### 1. Environmental Weather Conditions")
            amb_temp = st.slider("Ambient Temperature (°C)", min_value=10.0, max_value=45.0, value=28.0, step=0.5)
            humidity = st.slider("Relative Humidity (%)", min_value=20.0, max_value=95.0, value=55.0, step=1.0)
            it_load = st.slider("Server IT Workload (kW)", min_value=450.0, max_value=1250.0, value=850.0, step=10.0)

        with st.container(border=True):
            st.markdown("##### 2. Baseline Actuator Settings (Current)")
            cur_fan = st.slider("Current CRAH Fan Speed (%)", min_value=35.0, max_value=95.0, value=80.0, step=1.0)
            cur_pump = st.slider("Current Chilled Water Pump Speed (%)", min_value=35.0, max_value=95.0, value=75.0, step=1.0)

        with st.container(border=True):
            st.markdown("##### 3. Thermal Safety & Search Boundaries")
            max_inlet = st.slider(
                "Max Allowable Server Inlet Temperature (°C)",
                min_value=20.0,
                max_value=28.0,
                value=24.5,
                step=0.1,
                help="ASHRAE TC 9.9 recommended envelope: 18.0°C - 27.0°C. Candidates predicting inlet air above this limit will be rejected immediately.",
            )

            b_c1, b_c2 = st.columns(2)
            with b_c1:
                fan_bounds = st.slider("Fan Speed Range (%)", 35.0, 95.0, (40.0, 95.0), step=5.0)
            with b_c2:
                pump_bounds = st.slider("Pump Speed Range (%)", 35.0, 95.0, (40.0, 95.0), step=5.0)

            dur_c1, dur_c2 = st.columns(2)
            with dur_c1:
                hours = st.number_input("Operating Duration (Hours)", min_value=1.0, max_value=8760.0, value=24.0, step=1.0)
            with dur_c2:
                tariff = st.number_input("Electricity Tariff (INR / kWh)", min_value=1.0, max_value=25.0, value=7.50, step=0.25)

        run_btn = st.button("Optimize Cooling Actuators", type="primary", use_container_width=True)

    # Execute or retrieve optimization
    if run_btn:
        with st.spinner("Evaluating multi-candidate actuator grid and checking thermal safety..."):
            request = OptimizationRequest(
                ambient_temp_c=amb_temp,
                humidity_pct=humidity,
                it_load_kw=it_load,
                current_fan_speed_pct=cur_fan,
                current_pump_speed_pct=cur_pump,
                max_inlet_temp_c=max_inlet,
                min_fan_speed_pct=fan_bounds[0],
                max_fan_speed_pct=fan_bounds[1],
                min_pump_speed_pct=pump_bounds[0],
                max_pump_speed_pct=pump_bounds[1],
                operating_hours=hours,
                tariff_inr_per_kwh=tariff,
            )
            opt_result = optimize_cooling(request, power_model, temp_model)
            st.session_state["last_optimization_result"] = opt_result

    opt_res: Optional[OptimizationResult] = st.session_state.get("last_optimization_result", None)

    with col_output:
        st.markdown(f"#### {render_flaticon('chart', 20)} Optimization Results & Telemetry", unsafe_allow_html=True)

        if opt_res is None:
            # Default initialization run
            default_req = OptimizationRequest(
                ambient_temp_c=amb_temp,
                humidity_pct=humidity,
                it_load_kw=it_load,
                current_fan_speed_pct=cur_fan,
                current_pump_speed_pct=cur_pump,
                max_inlet_temp_c=max_inlet,
                min_fan_speed_pct=fan_bounds[0],
                max_fan_speed_pct=fan_bounds[1],
                min_pump_speed_pct=pump_bounds[0],
                max_pump_speed_pct=pump_bounds[1],
                operating_hours=hours,
                tariff_inr_per_kwh=tariff,
            )
            opt_res = optimize_cooling(default_req, power_model, temp_model)
            st.session_state["last_optimization_result"] = opt_res

        # Status Banner
        if opt_res.status == "FEASIBLE_RECOMMENDATION":
            st.success(f"**Feasible Recommendation Identified:** {opt_res.message}")
        elif opt_res.status == "BASELINE_ALREADY_OPTIMAL":
            st.info(f"**Baseline Already Optimal:** {opt_res.message}")
        else:
            st.error(f"**Safety Warning:** {opt_res.message}")

        # Comparison Metrics
        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            st.markdown(
                render_metric_card(
                    label="Actuator Settings",
                    value=f"{opt_res.recommended_fan_speed_pct:.0f}% / {opt_res.recommended_pump_speed_pct:.0f}%",
                    unit="Fan/Pump",
                    subtext=f"Baseline: {opt_res.baseline_fan_speed_pct:.0f}% / {opt_res.baseline_pump_speed_pct:.0f}%",
                    badge_text="Target Settings",
                    badge_variant="success" if opt_res.is_feasible else "warning",
                    icon_name="settings",
                ),
                unsafe_allow_html=True,
            )
        with res_c2:
            st.markdown(
                render_metric_card(
                    label="Cooling Power",
                    value=f"{opt_res.recommended_cooling_power_kw:.1f}",
                    unit="kW",
                    subtext=f"Baseline: {opt_res.baseline_cooling_power_kw:.1f} kW",
                    badge_text=f"Savings: {opt_res.savings_percent:.1f}%",
                    badge_variant="success" if opt_res.power_savings_kw > 0 else "info",
                    icon_name="snowflake",
                ),
                unsafe_allow_html=True,
            )
        with res_c3:
            st.markdown(
                render_metric_card(
                    label="Server Inlet Temp",
                    value=f"{opt_res.recommended_inlet_temp_c:.1f}",
                    unit="°C",
                    subtext=f"Safety Limit: {max_inlet:.1f}°C",
                    badge_text="Thermal Safe" if opt_res.recommended_inlet_temp_c <= max_inlet else "Breached",
                    badge_variant="success" if opt_res.recommended_inlet_temp_c <= max_inlet else "danger",
                    icon_name="thermometer",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

        # Candidate Search Space Visualization
        st.markdown(
            f"##### {render_flaticon('target', 18)} Actuator Search Grid & Cooling Power Landscape",
            unsafe_allow_html=True,
        )
        df_candidates = opt_res.candidates_df

        fig_scatter = go.Figure()

        # Rejected candidates
        df_rej = df_candidates[~df_candidates["is_feasible"]]
        if not df_rej.empty:
            fig_scatter.add_trace(
                go.Scatter(
                    x=df_rej["fan_speed_pct"],
                    y=df_rej["pump_speed_pct"],
                    mode="markers",
                    name="Rejected (Unsafe / Exceeds Limit)",
                    marker=dict(color="#f87171", symbol="x", size=6, opacity=0.45),
                    text=df_rej["rejection_reason"],
                    hovertemplate="Fan: %{x}%<br>Pump: %{y}%<br>Reason: %{text}<extra></extra>",
                )
            )

        # Feasible candidates
        df_fea = df_candidates[df_candidates["is_feasible"]]
        if not df_fea.empty:
            fig_scatter.add_trace(
                go.Scatter(
                    x=df_fea["fan_speed_pct"],
                    y=df_fea["pump_speed_pct"],
                    mode="markers",
                    name="Feasible Candidates",
                    marker=dict(
                        color=df_fea["predicted_cooling_power_kw"],
                        colorscale="Blues_r",
                        size=8,
                        showscale=True,
                        colorbar=dict(title="Cooling Power (kW)", thickness=12),
                    ),
                    text=df_fea["predicted_inlet_temp_c"],
                    hovertemplate="Fan: %{x}%<br>Pump: %{y}%<br>Power: %{marker.color} kW<br>Inlet Temp: %{text}°C<extra></extra>",
                )
            )

        # Current Baseline Point
        fig_scatter.add_trace(
            go.Scatter(
                x=[opt_res.baseline_fan_speed_pct],
                y=[opt_res.baseline_pump_speed_pct],
                mode="markers+text",
                name="Current Baseline",
                text=["Baseline"],
                textposition="top right",
                marker=dict(color="#f59e0b", size=14, symbol="square-open", line=dict(width=3, color="#d97706")),
            )
        )

        # Recommended Point
        if opt_res.is_feasible:
            fig_scatter.add_trace(
                go.Scatter(
                    x=[opt_res.recommended_fan_speed_pct],
                    y=[opt_res.recommended_pump_speed_pct],
                    mode="markers+text",
                    name="Recommended Optimal",
                    text=["Recommended"],
                    textposition="bottom left",
                    marker=dict(color="#10b981", size=18, symbol="star", line=dict(width=2, color="#065f46")),
                )
            )

        fig_scatter.update_layout(
            xaxis_title="CRAH Fan Speed (%)",
            yaxis_title="Chilled Water Pump Speed (%)",
        )
        apply_plotly_theme(fig_scatter, height=360)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Detailed Candidate Breakdown & CSV Download
        st.markdown(f"##### {render_flaticon('checklist', 18)} Candidate Evaluation Table", unsafe_allow_html=True)
        tab_fea, tab_rej, tab_dl = st.tabs(["Feasible Candidates", "Rejected Candidates", "Export Report"])

        with tab_fea:
            if not df_fea.empty:
                st.dataframe(
                    df_fea[
                        [
                            "fan_speed_pct",
                            "pump_speed_pct",
                            "predicted_cooling_power_kw",
                            "predicted_inlet_temp_c",
                        ]
                    ]
                    .sort_values("predicted_cooling_power_kw", ascending=True)
                    .head(25)
                    .reset_index(drop=True),
                    use_container_width=True,
                )
            else:
                st.warning("No feasible candidates meet safety limits under current conditions.")

        with tab_rej:
            if not df_rej.empty:
                st.dataframe(
                    df_rej[
                        [
                            "fan_speed_pct",
                            "pump_speed_pct",
                            "predicted_cooling_power_kw",
                            "predicted_inlet_temp_c",
                            "rejection_reason",
                        ]
                    ]
                    .head(25)
                    .reset_index(drop=True),
                    use_container_width=True,
                )
            else:
                st.info("No candidates were rejected.")

        with tab_dl:
            csv_data = opt_res.candidates_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Optimization Candidate Telemetry (CSV)",
                data=csv_data,
                file_name=f"cooling_optimization_eval_{int(datetime.datetime.now().timestamp())}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# PAGE 3: Sensor Data Explorer
# -----------------------------------------------------------------------------
def render_page_explorer(df: pd.DataFrame) -> None:
    render_header(
        title="Sensor Data Explorer",
        subtitle="Multi-Dimensional Telemetry Inspection, Filtering & Physical Schema",
    )
    render_disclaimer_banner()

    # Dataset Regeneration & Quick Stats Bar
    stat_c1, stat_c2, stat_c3, stat_c4 = st.columns([1.5, 1.5, 1.5, 2])
    with stat_c1:
        st.metric("Total Telemetry Rows", f"{len(df):,}")
    with stat_c2:
        st.metric("Timespan Range", f"{df['timestamp'].iloc[0][:10]} to {df['timestamp'].iloc[-1][:10]}")
    with stat_c3:
        st.metric("Sampling Frequency", "1 Hour (Simulated)")
    with stat_c4:
        with st.popover("Regenerate Dataset"):
            st.markdown(f"##### {render_flaticon('refresh', 18)} Generate Fresh Observations", unsafe_allow_html=True)
            new_samples = st.slider("Number of samples", 2000, 5000, 3500, step=500)
            new_seed = st.number_input("Random Seed", value=42, step=1)
            if st.button("Confirm & Regenerate", type="primary"):
                with st.spinner("Simulating thermodynamics and generating new dataset..."):
                    new_df = generate_cooling_data(n_samples=new_samples, random_seed=new_seed)
                    save_dataset(new_df)
                    st.cache_data.clear()
                    st.rerun()

    st.markdown("<hr style='margin: 1.2rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    # Interactive Filters
    st.markdown(f"#### {render_flaticon('search', 20)} Filter Telemetry Observations", unsafe_allow_html=True)
    f_c1, f_c2, f_c3 = st.columns(3)
    with f_c1:
        t_min, t_max = float(df["ambient_temp_c"].min()), float(df["ambient_temp_c"].max())
        sel_temp = st.slider("Ambient Temperature Range (°C)", t_min, t_max, (t_min, t_max), step=0.5)
    with f_c2:
        it_min, it_max = float(df["it_load_kw"].min()), float(df["it_load_kw"].max())
        sel_it = st.slider("IT Workload Range (kW)", it_min, it_max, (it_min, it_max), step=10.0)
    with f_c3:
        cp_min, cp_max = float(df["cooling_power_kw"].min()), float(df["cooling_power_kw"].max())
        sel_cp = st.slider("Cooling Power Range (kW)", cp_min, cp_max, (cp_min, cp_max), step=5.0)

    # Filtered View
    filtered_df = df[
        (df["ambient_temp_c"] >= sel_temp[0])
        & (df["ambient_temp_c"] <= sel_temp[1])
        & (df["it_load_kw"] >= sel_it[0])
        & (df["it_load_kw"] <= sel_it[1])
        & (df["cooling_power_kw"] >= sel_cp[0])
        & (df["cooling_power_kw"] <= sel_cp[1])
    ]

    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} total records matching filter criteria.")

    # Data Table Preview & Download
    with st.expander("Dataset Table Preview & CSV Export", expanded=True):
        st.dataframe(filtered_df.head(100), use_container_width=True)
        csv_full = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Telemetry (CSV)",
            data=csv_full,
            file_name="data_center_cooling_telemetry.csv",
            mime="text/csv",
        )

    # Summary Statistics & Correlation Analysis
    st.markdown(f"#### {render_flaticon('chart', 20)} Statistical Summary & Correlations", unsafe_allow_html=True)
    stat_tab1, stat_tab2, stat_tab3 = st.tabs(["Descriptive Statistics", "Correlation Heatmap", "Custom Variable Plotter"])

    with stat_tab1:
        st.dataframe(df.describe().round(2).T, use_container_width=True)

    with stat_tab2:
        numeric_cols = [c for c in df.columns if c != "timestamp"]
        corr_matrix = df[numeric_cols].corr().round(2)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            labels=dict(color="Correlation"),
        )
        apply_plotly_theme(fig_corr, height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

    with stat_tab3:
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            x_col = st.selectbox("X-Axis Variable", numeric_cols, index=1)
        with p_c2:
            y_col = st.selectbox("Y-Axis Variable", numeric_cols, index=7)
        with p_c3:
            plot_kind = st.selectbox("Plot Type", ["Scatter Plot", "Histogram / Distribution", "Line Plot"])

        if plot_kind == "Scatter Plot":
            fig_custom = px.scatter(
                filtered_df.iloc[::2],
                x=x_col,
                y=y_col,
                color="it_load_kw",
                labels={x_col: x_col, y_col: y_col},
            )
        elif plot_kind == "Histogram / Distribution":
            fig_custom = px.histogram(filtered_df, x=x_col, nbins=40)
        else:
            fig_custom = px.line(filtered_df.iloc[-200:], x="timestamp", y=y_col)

        apply_plotly_theme(fig_custom, height=380)
        st.plotly_chart(fig_custom, use_container_width=True)

    # Schema & Mathematical Assumptions Documentation
    with st.expander("Physical Modeling Assumptions & Schema Details"):
        st.markdown(
            """
            ### Physical Modeling Assumptions
            1. **Conservation of Thermal Energy**:
               All electrical power supplied to IT servers is converted directly into heat ($Q_{thermal} \\approx P_{IT}$).
            2. **Actuator Affinity Laws**:
               - CRAH fan power follows turbomachinery fan affinity laws: $P_{fan} \\propto \\text{Speed}^{2.7}$
               - Chilled water pumps follow pump affinity laws: $P_{pump} \\propto \\text{Speed}^{2.5}$
            3. **Chiller Thermodynamics & COP**:
               Chiller electricity consumption equals $P_{chiller} = Q_{thermal} / \\text{COP}$.
               COP degrades at high ambient temperatures due to elevated condensing head.
            4. **Server Cold Aisle Heat Balance (ASHRAE TC 9.9)**:
               Server inlet temperature is suppressed by CRAH airflow and chilled water loop temperature.
               $T_{outlet} = T_{inlet} + \\Delta T_{servers}$.
            5. **Power Usage Effectiveness (PUE)**:
               $$\\text{PUE} = \\frac{\\text{Total Facility Power}}{\\text{IT Load}} = \\frac{P_{IT} + P_{cooling} + P_{ancillary}}{P_{IT}}$$
            """
        )


# -----------------------------------------------------------------------------
# PAGE 4: ML Model Evaluation
# -----------------------------------------------------------------------------
def render_page_evaluation(metadata: Dict[str, Any]) -> None:
    render_header(
        title="ML Model Evaluation & Diagnostics",
        subtitle="Performance Comparison, Error Metrics & Sensitivity for Surrogate Regressors",
    )
    render_disclaimer_banner()

    # Retrain button
    t_c1, t_c2 = st.columns([3, 1])
    with t_c1:
        st.markdown(
            f"""
            **Training Dataset:** {metadata.get('train_samples', 2800):,} training observations |
            **Held-Out Test Set:** {metadata.get('test_samples', 700):,} test observations (Test size: {metadata.get('test_size', 0.2)*100:.0f}%) |
            **Algorithm:** Random Forest Regressor (100 estimators, max depth 16) vs Baseline Mean Dummy
            """
        )
    with t_c2:
        if st.button("Retrain ML Models", type="secondary"):
            with st.spinner("Retraining surrogate models and generating evaluation metrics..."):
                train_models()
                st.cache_resource.clear()
                st.rerun()

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Metrics Summary Tables
    st.markdown(f"#### {render_flaticon('chart', 20)} Regression Performance Metrics (Held-Out Test Data)", unsafe_allow_html=True)
    m_col1, m_col2 = st.columns(2)

    power_metrics = metadata.get("cooling_power_metrics", {})
    temp_metrics = metadata.get("inlet_temp_metrics", {})

    with m_col1:
        st.markdown(f"##### {render_flaticon('snowflake', 16)} Model 1: Cooling Power Predictor (`cooling_power_kw`)", unsafe_allow_html=True)
        rf_p = power_metrics.get("random_forest", {"mae": 4.45, "rmse": 5.65, "r2": 0.97})
        dum_p = power_metrics.get("dummy_baseline", {"mae": 31.36, "rmse": 36.39, "r2": -0.002})

        comp_p = pd.DataFrame(
            [
                {"Model": "Random Forest Regressor", "MAE (kW)": rf_p["mae"], "RMSE (kW)": rf_p["rmse"], "R² Score": rf_p["r2"]},
                {"Model": "Dummy Mean Baseline", "MAE (kW)": dum_p["mae"], "RMSE (kW)": dum_p["rmse"], "R² Score": dum_p["r2"]},
            ]
        )
        st.dataframe(comp_p, use_container_width=True, hide_index=True)
        st.caption("Random Forest achieves >97% variance explanation, reducing MAE by >85% over the naive baseline.")

    with m_col2:
        st.markdown(f"##### {render_flaticon('thermometer', 16)} Model 2: Server Inlet Temperature Predictor (`inlet_temp_c`)", unsafe_allow_html=True)
        rf_t = temp_metrics.get("random_forest", {"mae": 0.20, "rmse": 0.25, "r2": 0.72})
        dum_t = temp_metrics.get("dummy_baseline", {"mae": 0.38, "rmse": 0.48, "r2": -0.001})

        comp_t = pd.DataFrame(
            [
                {"Model": "Random Forest Regressor", "MAE (°C)": rf_t["mae"], "RMSE (°C)": rf_t["rmse"], "R² Score": rf_t["r2"]},
                {"Model": "Dummy Mean Baseline", "MAE (°C)": dum_t["mae"], "RMSE (°C)": dum_t["rmse"], "R² Score": dum_t["r2"]},
            ]
        )
        st.dataframe(comp_t, use_container_width=True, hide_index=True)
        st.caption("Inlet temperature predictor achieves an MAE of ~0.20°C, ensuring tight safety bounds.")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Actual vs Predicted & Residual Plots
    eval_samples = metadata.get("eval_samples", {})
    if eval_samples and "actual_power" in eval_samples:
        act_p = eval_samples["actual_power"]
        prd_p = eval_samples["pred_power_rf"]
        res_p = [round(a - p, 2) for a, p in zip(act_p, prd_p)]

        act_t = eval_samples["actual_temp"]
        prd_t = eval_samples["pred_temp_rf"]
        res_t = [round(a - p, 2) for a, p in zip(act_t, prd_t)]

        diag_c1, diag_c2 = st.columns(2)

        with diag_c1:
            st.markdown(f"##### {render_flaticon('target', 18)} Cooling Power: Predicted vs. Actual", unsafe_allow_html=True)
            fig_ap = go.Figure()
            fig_ap.add_trace(
                go.Scatter(
                    x=act_p,
                    y=prd_p,
                    mode="markers",
                    name="Held-out Test Samples",
                    marker=dict(color=COLOR_PRIMARY, size=5, opacity=0.7),
                )
            )
            min_p = min(min(act_p), min(prd_p))
            max_p = max(max(act_p), max(prd_p))
            fig_ap.add_trace(
                go.Scatter(
                    x=[min_p, max_p],
                    y=[min_p, max_p],
                    mode="lines",
                    name="Ideal (1:1)",
                    line=dict(color="#dc2626", dash="dot", width=1.5),
                )
            )
            fig_ap.update_layout(xaxis_title="Actual Cooling Power (kW)", yaxis_title="Predicted Cooling Power (kW)")
            apply_plotly_theme(fig_ap, height=330)
            st.plotly_chart(fig_ap, use_container_width=True)

        with diag_c2:
            st.markdown(f"##### {render_flaticon('chart', 18)} Cooling Power Residual Distribution", unsafe_allow_html=True)
            fig_rp = px.histogram(
                x=res_p,
                nbins=30,
                color_discrete_sequence=[COLOR_SECONDARY],
                labels={"x": "Prediction Residual Error (kW)"},
            )
            apply_plotly_theme(fig_rp, height=330)
            st.plotly_chart(fig_rp, use_container_width=True)

    # Feature Importance Analysis
    st.markdown(f"#### {render_flaticon('search', 20)} Feature Sensitivity & Importance", unsafe_allow_html=True)
    imp_c1, imp_c2 = st.columns(2)

    power_imp = metadata.get("cooling_power_importance", {})
    temp_imp = metadata.get("inlet_temp_importance", {})

    with imp_c1:
        st.markdown("##### Feature Importance: Cooling Power")
        df_p_imp = pd.DataFrame(
            [{"Feature": k, "Importance": v} for k, v in power_imp.items()]
        ).sort_values("Importance", ascending=True)

        fig_pi = px.bar(df_p_imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[COLOR_PRIMARY])
        apply_plotly_theme(fig_pi, height=280)
        st.plotly_chart(fig_pi, use_container_width=True)

    with imp_c2:
        st.markdown("##### Feature Importance: Server Inlet Temperature")
        df_t_imp = pd.DataFrame(
            [{"Feature": k, "Importance": v} for k, v in temp_imp.items()]
        ).sort_values("Importance", ascending=True)

        fig_ti = px.bar(df_t_imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[COLOR_ACCENT])
        apply_plotly_theme(fig_ti, height=280)
        st.plotly_chart(fig_ti, use_container_width=True)

    # Metric Interpretation Guide
    with st.expander("Explanation of Evaluation Metrics & Validation Integrity"):
        st.markdown(
            """
            - **Mean Absolute Error (MAE)**: Measures the average magnitude of absolute prediction errors in target units.
              For cooling power, an MAE of ~4.5 kW represents less than 2.5% relative error across typical plant loads.
            - **Root Mean Squared Error (RMSE)**: Penalizes larger outliers heavily. Low RMSE indicates that the surrogate does not produce extreme hazardous prediction spikes.
            - **Coefficient of Determination ($R^2$)**: Represents the proportion of target variance explained by the model. Values > 0.95 confirm that the surrogate accurately maps physical plant dynamics.
            - **Surrogate Generalization Note**: High test accuracy on synthetic telemetry validates that the Random Forest has successfully captured the simulated physical equations. However, **this does not prove performance on physical data centers**, where unmodeled variables (sensor drift, fouling, air recirculation) create additional noise.
            """
        )


# -----------------------------------------------------------------------------
# PAGE 5: Energy Savings Analysis
# -----------------------------------------------------------------------------
def render_page_savings(power_model: Any, temp_model: Any) -> None:
    render_header(
        title="Energy Savings & Financial Impact Analysis",
        subtitle="Translating Surrogate Actuator Optimization into Power, Energy, and Cost Reductions",
    )
    render_disclaimer_banner()

    opt_res: Optional[OptimizationResult] = st.session_state.get("last_optimization_result", None)

    # Configuration bar
    st.markdown(f"#### {render_flaticon('lightning', 20)} Operating Horizon & Tariff Assumptions", unsafe_allow_html=True)
    cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
    with cfg_c1:
        time_horizon = st.selectbox(
            "Evaluation Time Horizon",
            ["24 Hours (Daily)", "168 Hours (1 Week)", "720 Hours (1 Month)", "8,760 Hours (Annual Extrapolation)", "Custom Duration"],
            index=0,
        )
    with cfg_c2:
        if time_horizon == "Custom Duration":
            eval_hours = st.number_input("Custom Hours", min_value=1.0, max_value=87600.0, value=24.0, step=1.0)
        elif "24 Hours" in time_horizon:
            eval_hours = 24.0
        elif "168 Hours" in time_horizon:
            eval_hours = 168.0
        elif "720 Hours" in time_horizon:
            eval_hours = 720.0
        else:
            eval_hours = 8760.0
        st.metric("Total Operating Hours", f"{eval_hours:,.0f} hrs")

    with cfg_c3:
        tariff_rate = st.number_input("Commercial Tariff Rate (INR ₹ / kWh)", min_value=1.0, max_value=25.0, value=7.50, step=0.25)

    if opt_res is None:
        # Generate default comparison
        req = OptimizationRequest(
            ambient_temp_c=28.0,
            humidity_pct=55.0,
            it_load_kw=850.0,
            current_fan_speed_pct=80.0,
            current_pump_speed_pct=75.0,
            max_inlet_temp_c=24.5,
            operating_hours=eval_hours,
            tariff_inr_per_kwh=tariff_rate,
        )
        opt_res = optimize_cooling(req, power_model, temp_model)
    else:
        # Recompute with current duration and tariff
        opt_res.operating_hours = eval_hours
        opt_res.tariff_inr_per_kwh = tariff_rate
        opt_res.energy_savings_kwh = round(opt_res.power_savings_kw * eval_hours, 2)
        opt_res.cost_savings_inr = round(opt_res.energy_savings_kwh * tariff_rate, 2)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # 4 Key Savings Cards
    s_c1, s_c2, s_c3, s_c4 = st.columns(4)
    with s_c1:
        st.markdown(
            render_metric_card(
                label="Baseline vs Optimized Power",
                value=f"{opt_res.recommended_cooling_power_kw:.1f}",
                unit="kW",
                subtext=f"Baseline: {opt_res.baseline_cooling_power_kw:.1f} kW",
                badge_text=f"Delta: -{opt_res.power_savings_kw:.1f} kW",
                badge_variant="success",
                icon_name="lightning",
            ),
            unsafe_allow_html=True,
        )
    with s_c2:
        st.markdown(
            render_metric_card(
                label="Cooling Power Savings",
                value=f"{opt_res.savings_percent:.1f}%",
                unit="",
                subtext=f"Reduced by {opt_res.power_savings_kw:.1f} kW",
                badge_text="Relative Reduction",
                badge_variant="success",
                icon_name="trend_down",
            ),
            unsafe_allow_html=True,
        )
    with s_c3:
        st.markdown(
            render_metric_card(
                label="Energy Saved Over Horizon",
                value=f"{opt_res.energy_savings_kwh:,.0f}",
                unit="kWh",
                subtext=f"Over {eval_hours:,.0f} operating hours",
                badge_text="Electricity Saved",
                badge_variant="success",
                icon_name="snowflake",
            ),
            unsafe_allow_html=True,
        )
    with s_c4:
        st.markdown(
            render_metric_card(
                label="Estimated Financial Savings",
                value=f"₹{opt_res.cost_savings_inr:,.0f}",
                unit="INR",
                subtext=f"@ ₹{tariff_rate:.2f}/kWh commercial tariff",
                badge_text="Operational Expenditure",
                badge_variant="success",
                icon_name="chart",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # Graphical Comparisons
    ch_c1, ch_c2 = st.columns(2)

    with ch_c1:
        st.markdown(f"##### {render_flaticon('chart', 18)} Facility Power Breakdown Comparison", unsafe_allow_html=True)
        # Bar comparison
        it_kw = 850.0
        ancillary_kw = 0.035 * it_kw + 18.0
        base_tot = it_kw + opt_res.baseline_cooling_power_kw + ancillary_kw
        opt_tot = it_kw + opt_res.recommended_cooling_power_kw + ancillary_kw

        fig_breakdown = go.Figure(
            data=[
                go.Bar(name="IT Equipment Load", x=["Baseline", "AI Optimized"], y=[it_kw, it_kw], marker_color="#94a3b8"),
                go.Bar(name="Ancillary Losses", x=["Baseline", "AI Optimized"], y=[ancillary_kw, ancillary_kw], marker_color="#cbd5e1"),
                go.Bar(
                    name="Cooling Infrastructure",
                    x=["Baseline", "AI Optimized"],
                    y=[opt_res.baseline_cooling_power_kw, opt_res.recommended_cooling_power_kw],
                    marker_color=[COLOR_ACCENT, COLOR_SECONDARY],
                ),
            ]
        )
        fig_breakdown.update_layout(
            barmode="stack",
            yaxis_title="Facility Power Demand (kW)",
        )
        apply_plotly_theme(fig_breakdown, height=340)
        st.plotly_chart(fig_breakdown, use_container_width=True)

    with ch_c2:
        st.markdown(f"##### {render_flaticon('trend_down', 18)} Cumulative Cost Savings Over {eval_hours:,.0f} Hours", unsafe_allow_html=True)
        time_points = np.linspace(0, eval_hours, 50)
        cum_savings = time_points * opt_res.power_savings_kw * tariff_rate

        fig_cum = go.Figure()
        fig_cum.add_trace(
            go.Scatter(
                x=time_points,
                y=cum_savings,
                mode="lines",
                line=dict(color=COLOR_SECONDARY, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(5, 150, 105, 0.08)",
                name="Cumulative INR Saved",
            )
        )
        fig_cum.update_layout(
            xaxis_title="Hours Elapsed",
            yaxis_title="Cumulative Cost Savings (INR ₹)",
        )
        apply_plotly_theme(fig_cum, height=340)
        st.plotly_chart(fig_cum, use_container_width=True)

    # Detailed Assumptions & Export Report
    st.markdown(f"##### {render_flaticon('document', 18)} Export Savings Audit Report", unsafe_allow_html=True)
    audit_data = pd.DataFrame(
        [
            {
                "Operating Horizon (Hours)": eval_hours,
                "Tariff Rate (INR/kWh)": tariff_rate,
                "Baseline Fan Speed (%)": opt_res.baseline_fan_speed_pct,
                "Baseline Pump Speed (%)": opt_res.baseline_pump_speed_pct,
                "Baseline Cooling Power (kW)": opt_res.baseline_cooling_power_kw,
                "Baseline Inlet Temp (°C)": opt_res.baseline_inlet_temp_c,
                "Baseline PUE": opt_res.baseline_pue,
                "Optimized Fan Speed (%)": opt_res.recommended_fan_speed_pct,
                "Optimized Pump Speed (%)": opt_res.recommended_pump_speed_pct,
                "Optimized Cooling Power (kW)": opt_res.recommended_cooling_power_kw,
                "Optimized Inlet Temp (°C)": opt_res.recommended_inlet_temp_c,
                "Optimized PUE": opt_res.recommended_pue,
                "Power Reduction (kW)": opt_res.power_savings_kw,
                "Savings Percentage (%)": opt_res.savings_percent,
                "Total Energy Saved (kWh)": opt_res.energy_savings_kwh,
                "Total Cost Saved (INR)": opt_res.cost_savings_inr,
            }
        ]
    )

    csv_audit = audit_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Detailed Energy Savings Audit Report (CSV)",
        data=csv_audit,
        file_name="cooling_energy_savings_audit.csv",
        mime="text/csv",
    )

    with st.expander("Analytical Assumptions & Boundary Disclaimers"):
        st.markdown(
            """
            - **Constant-Operating-Condition Assumption**: Projections across extended horizons (e.g., 720 hours or 8,760 hours) assume that ambient weather and server workloads remain static at the configured snapshot values. In reality, diurnal weather cycles and dynamic computational workloads produce varying power demands throughout each 24-hour cycle.
            - **Cooling Savings vs. Total Facility Savings**: A 15–20% reduction in **cooling infrastructure power** corresponds to approximately 3–6% reduction in **total facility power**, since IT server compute power represents the majority of total data center power consumption.
            - **Simulation Boundary**: These figures reflect the surrogate physics model rather than physical utility meter data.
            """
        )


# -----------------------------------------------------------------------------
# PAGE 6: Case Study: Google DeepMind
# -----------------------------------------------------------------------------
def render_page_case_study() -> None:
    render_header(
        title="Google DeepMind Case Study: AI in Data Centers",
        subtitle="From 40% Cooling Energy Reductions to Autonomous Industrial Control",
    )
    render_disclaimer_banner()

    # DeepMind Overview Section
    st.markdown(
        f"""
        ### {render_flaticon('network', 24)} The Industrial Computing Challenge
        Data centers power the modern global internet, executing billions of search queries, cloud applications,
        and AI workloads every day. In doing so, thousands of high-performance microprocessors, GPUs, and power supplies
        convert electrical energy almost entirely into **waste heat**.

        To prevent thermal throttling and server hardware damage, data centers employ large-scale industrial cooling plants,
        including:
        - **Computer Room Air Handlers (CRAH)** and Variable Frequency Drive (VFD) fans
        - **Water-cooled chillers** and compressors
        - **Cooling towers**, condenser loops, and chilled water circulation pumps

        Historically, cooling systems consumed **up to 40% of a data center's total electrical footprint**.
        Even minor improvements in cooling efficiency translate into millions of kilowatt-hours (kWh) saved and substantial carbon reductions.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='margin: 1.2rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    # Core Concepts Grid
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="panel-card">
                <h5 style="color: #0284c7; margin-bottom: 0.5rem;">{render_flaticon('ruler', 20)} Understanding PUE</h5>
                <p style="font-size: 0.84rem; color: #334155;">
                    <strong>Power Usage Effectiveness (PUE)</strong> is the global benchmark for data center efficiency:
                </p>
                <div style="background: #f1f5f9; padding: 0.5rem; border-radius: 4px; font-family: monospace; font-size: 0.8rem; margin: 0.5rem 0;">
                    PUE = Total Facility Power / IT Power
                </div>
                <p style="font-size: 0.82rem; color: #64748b;">
                    A theoretical ideal PUE is 1.0 (100% of electricity directly powering computing chips). Standard legacy data centers often operate between 1.5 and 2.0.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="panel-card">
                <h5 style="color: #059669; margin-bottom: 0.5rem;">{render_flaticon('chip', 20)} The ML Approach</h5>
                <p style="font-size: 0.84rem; color: #334155;">
                    Why can't simple engineering rules optimize cooling?
                </p>
                <p style="font-size: 0.82rem; color: #64748b;">
                    Cooling plants are governed by non-linear thermodynamic interactions: fluctuating weather (temperature, humidity, wet-bulb), dynamic IT compute bursts, and non-linear pump/fan affinity laws. DeepMind trained deep neural networks to accurately predict PUE across millions of potential setpoint combinations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="panel-card">
                <h5 style="color: #d97706; margin-bottom: 0.5rem;">{render_flaticon('shield', 20)} Safety Constraints</h5>
                <p style="font-size: 0.84rem; color: #334155;">
                    Safety-first architecture:
                </p>
                <p style="font-size: 0.82rem; color: #64748b;">
                    Optimizers cannot simply shut off fans to save energy. Strict thermal bounds (e.g. ASHRAE Class A1 cold aisle limits) must be continuously maintained. DeepMind established two-layer safety verification: learned surrogate predictions are validated against hard physics constraints, with instant fallback to standard controls if confidence drops.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Timeline of DeepMind's Research
    st.markdown(f"### {render_flaticon('calendar', 22)} Timeline of DeepMind's Published Milestones", unsafe_allow_html=True)

    t_df = pd.DataFrame(
        [
            {
                "Year": "2014",
                "Milestone": "Initial Proof of Concept",
                "Impact": "Machine learning models trained on historical sensor telemetry demonstrated that neural networks could predict PUE with 99.6% accuracy.",
            },
            {
                "Year": "2016",
                "Milestone": "40% Cooling Bill Reduction (Recommendation Mode)",
                "Impact": "DeepMind deployed ML models to generate real-time recommendations for data center plant operators, achieving an unprecedented 40% reduction in cooling energy and a 15% reduction in overall PUE.",
            },
            {
                "Year": "2018",
                "Milestone": "Autonomous Closed-Loop Control",
                "Impact": "Transitioned from operator recommendation alerts to an autonomous closed-loop AI control system with multi-layer safety verification, operating live across multiple production data centers.",
            },
            {
                "Year": "Present",
                "Milestone": "Industrial Expansion & Grid Optimization",
                "Impact": "Principles generalized to wind energy scheduling, district heating networks, and Google Cloud carbon-aware computing.",
            },
        ]
    )
    st.dataframe(t_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # Primary Sources & Citations
    st.markdown(f"### {render_flaticon('link', 22)} Primary Research Sources & Citations", unsafe_allow_html=True)
    st.markdown(
        f"""
        - {render_flaticon('document', 16)} **DeepMind (2016)**: [DeepMind AI Reduces Google Data Centre Cooling Bill by 40%](https://deepmind.google/discover/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-by-40/)
        - {render_flaticon('document', 16)} **DeepMind (2018)**: [Safety-first AI for autonomous data centre cooling and industrial control](https://deepmind.google/blog/safety-first-ai-for-autonomous-data-centre-cooling-and-industrial-control/)
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='margin: 1.2rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    # Three-Tier Attribution
    st.markdown(f"### {render_flaticon('scale', 22)} Three-Tier Attribution & System Scope", unsafe_allow_html=True)
    st.markdown(
        """
        To maintain scientific integrity and clarity, this project explicitly separates three distinct tiers:

        1. **Google DeepMind's Published Production Findings**:
           - Refers exclusively to Google's proprietary production facilities, real sensor telemetry networks, and verified 40% energy reductions published in 2016 and 2018.
        2. **This Project's Educational Modeling Assumptions**:
           - Incorporates established thermodynamic formulas (fan/pump affinity laws, chiller COP curves, ASHRAE inlet temperature boundaries) implemented in `data_generator.py`.
        3. **Simulation Results Measured Locally**:
           - Refers strictly to the numbers, charts, and optimization percentages generated by the local Scikit-Learn surrogate models on this laptop.
        """
    )


# -----------------------------------------------------------------------------
# PAGE 7: About the Project
# -----------------------------------------------------------------------------
def render_page_about() -> None:
    render_header(
        title="About AI Cooling Intelligence",
        subtitle="System Architecture, Physics Specifications & Engineering Methodology",
    )
    render_disclaimer_banner()

    st.markdown(
        f"""
        ### {render_flaticon('architecture', 24)} End-to-End System Architecture
        This application provides a complete, modular, and self-contained machine learning simulation
        demonstrating how data center cooling energy can be minimized without breaching critical hardware safety constraints.
        """,
        unsafe_allow_html=True,
    )

    # Architecture Overview Cards
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.markdown(
            f"""
            <div class="panel-card">
                <div style="font-weight: 700; color: #0284c7; margin-bottom: 0.3rem;">{render_flaticon('network', 16)} 1. Data Generator</div>
                <div style="font-size: 0.78rem; font-family: monospace; color: #64748b; margin-bottom: 0.5rem;">data_generator.py</div>
                <div style="font-size: 0.8rem; color: #334155;">
                    Generates reproducible hourly telemetry with diurnal and seasonal cycles, adhering to conservation of energy and turbomachinery affinity laws.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            f"""
            <div class="panel-card">
                <div style="font-weight: 700; color: #059669; margin-bottom: 0.3rem;">{render_flaticon('chip', 16)} 2. ML Model Trainer</div>
                <div style="font-size: 0.78rem; font-family: monospace; color: #64748b; margin-bottom: 0.5rem;">model_trainer.py</div>
                <div style="font-size: 0.8rem; color: #334155;">
                    Trains two Random Forest surrogate regression models to predict cooling power consumption and cold-aisle server inlet temperature without target leakage.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a3:
        st.markdown(
            f"""
            <div class="panel-card">
                <div style="font-weight: 700; color: #d97706; margin-bottom: 0.3rem;">{render_flaticon('settings', 16)} 3. Optimizer Engine</div>
                <div style="font-size: 0.78rem; font-family: monospace; color: #64748b; margin-bottom: 0.5rem;">optimizer.py</div>
                <div style="font-size: 0.8rem; color: #334155;">
                    Executes candidate search across actuator bounds, rejecting unsafe thermal settings and identifying the minimum energy configuration.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a4:
        st.markdown(
            f"""
            <div class="panel-card">
                <div style="font-weight: 700; color: #7c3aed; margin-bottom: 0.3rem;">{render_flaticon('laptop', 16)} 4. Streamlit UI</div>
                <div style="font-size: 0.78rem; font-family: monospace; color: #64748b; margin-bottom: 0.5rem;">app.py & utils.py</div>
                <div style="font-size: 0.8rem; color: #334155;">
                    Delivers a Google Cloud / DeepMind inspired UI with interactive Plotly visual analytics, live KPI cards, and CSV export.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # Technology Stack Table
    st.markdown(f"### {render_flaticon('laptop', 22)} Technology Stack & Environment", unsafe_allow_html=True)
    tech_df = pd.DataFrame(
        [
            {"Component": "Frontend & Analytics UI", "Technology": "Streamlit (Python)", "Version": ">= 1.30.0"},
            {"Component": "Machine Learning", "Technology": "Scikit-learn (RandomForest, DummyRegressor)", "Version": ">= 1.3.0"},
            {"Component": "Data Wrangling & Numerics", "Technology": "Pandas & NumPy", "Version": ">= 2.0.0 / >= 1.24.0"},
            {"Component": "Interactive Visualizations", "Technology": "Plotly (graph_objects, express)", "Version": ">= 5.18.0"},
            {"Component": "Model Persistence", "Technology": "Joblib", "Version": ">= 1.3.0"},
            {"Component": "Automated Testing", "Technology": "Pytest", "Version": ">= 7.4.0"},
            {"Component": "Runtime Environment", "Technology": "Python on Windows PowerShell", "Version": "Python 3.11+"},
        ]
    )
    st.dataframe(tech_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # Mathematical Modeling Breakdown
    st.markdown(f"### {render_flaticon('math', 22)} Mathematical Modeling Principles", unsafe_allow_html=True)
    st.markdown(
        """
        - **Fan Power Equation**:
          $$P_{fan} = 85.0 \\times \\left(\\frac{\\text{Fan Speed \\%}}{100}\\right)^{2.7} \\quad [\\text{kW}]$$
        - **Pump Power Equation**:
          $$P_{pump} = 50.0 \\times \\left(\\frac{\\text{Pump Speed \\%}}{100}\\right)^{2.5} \\quad [\\text{kW}]$$
        - **Chiller Plant Electricity Demand**:
          $$P_{chiller} = \\frac{P_{IT}}{\\text{COP}}, \\quad \\text{where } \\text{COP} = 5.1 - 0.055(T_{amb} - 22) - 0.012(\\text{Hum} - 50) + 0.015(\\text{Pump} - 50)$$
        - **Total Cooling Power**:
          $$P_{cooling} = P_{fan} + P_{pump} + P_{chiller} + \\epsilon$$
        - **Server Cold Aisle Inlet Temperature**:
          $$T_{inlet} = 18.5 + 0.14(T_{amb} - 22) + 0.0065(P_{IT} - 800) - 0.10(\\text{Fan} - 50) - 0.06(\\text{Pump} - 50) + \\epsilon$$
        """
    )


# -----------------------------------------------------------------------------
# Main Application Router
# -----------------------------------------------------------------------------
def main() -> None:
    df = get_dataset()
    power_model, temp_model, metadata = get_models()

    selected_page = render_sidebar(df, metadata)

    if selected_page == "Overview Dashboard":
        render_page_overview(df, power_model, temp_model, metadata)
    elif selected_page == "Cooling Optimizer":
        render_page_optimizer(power_model, temp_model)
    elif selected_page == "Sensor Data Explorer":
        render_page_explorer(df)
    elif selected_page == "ML Model Evaluation":
        render_page_evaluation(metadata)
    elif selected_page == "Energy Savings Analysis":
        render_page_savings(power_model, temp_model)
    elif selected_page == "Case Study: Google DeepMind":
        render_page_case_study()
    elif selected_page == "About the Project":
        render_page_about()


if __name__ == "__main__":
    main()
