"""
AI Data Center Cooling Optimizer - Utility Functions & UI Styling
=================================================================
Provides design tokens, Google Cloud/DeepMind inspired themes, Plotly templates,
and helper formatting utilities for the Streamlit dashboard.
"""

from __future__ import annotations

import base64
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_ICONS_DIR = BASE_DIR / "assets" / "icons"

# Color Palette Constants
COLOR_PRIMARY = "#0284c7"  # Sky blue
COLOR_SECONDARY = "#059669"  # Emerald green
COLOR_ACCENT = "#d97706"  # Amber
COLOR_DANGER = "#dc2626"  # Coral red
COLOR_TEXT_MAIN = "#0f172a"  # Slate dark
COLOR_TEXT_MUTED = "#64748b"  # Slate muted
COLOR_BG_CARD = "#ffffff"  # Card white
COLOR_BG_PAGE = "#f8fafc"  # Soft slate page background
COLOR_BORDER = "#e2e8f0"  # Subtle slate border


@st.cache_data
def get_icon_base64(name: str) -> str:
    """Read a local Flaticon image asset and encode as base64 string for offline HTML embedding."""
    icon_path = ASSETS_ICONS_DIR / f"{name}.png"
    if icon_path.exists():
        with open(icon_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    return ""


def render_flaticon(
    name: str,
    size: int = 18,
    margin_right: int = 6,
    vertical_align: str = "middle",
) -> str:
    """Return an HTML <img> tag rendering an authentic Flaticon vector icon.
    
    Args:
        name: Name of the icon file in assets/icons without .png extension.
        size: Width and height of the icon in pixels.
        margin_right: Right margin in pixels.
        vertical_align: CSS vertical alignment (default: 'middle').
    """
    uri = get_icon_base64(name)
    if not uri:
        return ""
    return (
        f'<img src="{uri}" width="{size}" height="{size}" '
        f'style="vertical-align: {vertical_align}; margin-right: {margin_right}px; display: inline-block;" '
        f'alt="{name} icon" />'
    )



def apply_custom_css() -> None:
    """Inject clean, modern CSS styling inspired by Google Cloud and DeepMind."""
    st.markdown(
        f"""
        <style>
            /* Main Application Styling */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

            .stApp {{
                background-color: {COLOR_BG_PAGE} !important;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                color: {COLOR_TEXT_MAIN};
            }}

            .stApp header {{
                background-color: transparent !important;
            }}

            /* Container top padding and layout */
            .block-container {{
                padding-top: 4.8rem !important;
                padding-bottom: 2.5rem;
                padding-left: 2.5rem;
                padding-right: 2.5rem;
                max-width: 1400px;
            }}

            /* Header and Subtitles */
            .main-header {{
                margin-top: 0.6rem;
                margin-bottom: 1.2rem;
                padding-bottom: 0.8rem;
                border-bottom: 1px solid {COLOR_BORDER};
            }}
            .main-title {{
                font-size: 1.85rem;
                font-weight: 700;
                color: {COLOR_TEXT_MAIN};
                letter-spacing: -0.02em;
                margin-bottom: 0.2rem;
            }}
            .main-subtitle {{
                font-size: 0.98rem;
                color: {COLOR_TEXT_MUTED};
                font-weight: 400;
            }}

            /* Disclaimer Banner */
            .disclaimer-banner {{
                background-color: #f1f5f9;
                border-left: 4px solid {COLOR_PRIMARY};
                border-radius: 6px;
                padding: 0.65rem 1rem;
                margin-bottom: 1.4rem;
                font-size: 0.82rem;
                color: #334155;
                line-height: 1.45;
            }}

            /* Metric Cards */
            .metric-card {{
                background: {COLOR_BG_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 10px;
                padding: 1rem 1.15rem;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
                transition: transform 0.15s ease, box-shadow 0.15s ease;
                margin-bottom: 0.8rem;
                height: 100%;
            }}
            .metric-card:hover {{
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
            }}
            .metric-label {{
                font-size: 0.78rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                color: {COLOR_TEXT_MUTED};
                margin-bottom: 0.25rem;
            }}
            .metric-value {{
                font-size: 1.65rem;
                font-weight: 700;
                color: {COLOR_TEXT_MAIN};
                line-height: 1.2;
            }}
            .metric-unit {{
                font-size: 0.88rem;
                font-weight: 500;
                color: {COLOR_TEXT_MUTED};
                margin-left: 0.2rem;
            }}
            .metric-subtext {{
                font-size: 0.78rem;
                margin-top: 0.35rem;
                color: {COLOR_TEXT_MUTED};
            }}

            /* Badges */
            .badge {{
                display: inline-block;
                padding: 0.22rem 0.6rem;
                font-size: 0.74rem;
                font-weight: 600;
                border-radius: 9999px;
                letter-spacing: 0.02em;
            }}
            .badge-success {{
                background-color: #ecfdf5;
                color: #065f46;
                border: 1px solid #a7f3d0;
            }}
            .badge-warning {{
                background-color: #fffbeb;
                color: #92400e;
                border: 1px solid #fde68a;
            }}
            .badge-danger {{
                background-color: #fef2f2;
                color: #991b1b;
                border: 1px solid #fecaca;
            }}
            .badge-info {{
                background-color: #f0f9ff;
                color: #0369a1;
                border: 1px solid #bae6fd;
            }}

            /* Section Cards & Panels */
            .panel-card {{
                background: {COLOR_BG_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 10px;
                padding: 1.25rem;
                margin-bottom: 1.2rem;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
            }}

            /* Sidebar Tweaks */
            [data-testid="stSidebar"] {{
                background-color: #ffffff !important;
                border-right: 1px solid {COLOR_BORDER};
            }}

            /* High-contrast sidebar navigation text */
            [data-testid="stSidebar"] .stRadio label p,
            [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label div p,
            [data-testid="stSidebar"] .stRadio label span {{
                color: #1e293b !important;
                font-weight: 500 !important;
                font-size: 0.88rem !important;
            }}
            [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover p {{
                color: {COLOR_PRIMARY} !important;
            }}

            /* Streamlit Button Tweaks */
            div.stButton > button:first-child {{
                border-radius: 6px;
                font-weight: 500;
                transition: all 0.15s ease;
            }}

            /* Clean Table Styling */
            .dataframe {{
                font-size: 0.84rem !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply a crisp, modern Google Cloud / DeepMind styling to a Plotly figure."""
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12, color=COLOR_TEXT_MAIN),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=height,
        margin=dict(l=45, r=25, t=35, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            linecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_TEXT_MUTED, size=11),
            title_font=dict(color=COLOR_TEXT_MAIN, size=12),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            linecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_TEXT_MUTED, size=11),
            title_font=dict(color=COLOR_TEXT_MAIN, size=12),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=COLOR_TEXT_MAIN),
            bgcolor="rgba(255,255,255,0.7)",
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#ffffff",
        ),
    )
    return fig


def render_metric_card(
    label: str,
    value: str | float | int,
    unit: str = "",
    subtext: str = "",
    badge_text: str = "",
    badge_variant: str = "success",
    icon_name: str = "",
) -> str:
    """Generate HTML string for a clean KPI metric card with optional Flaticon vector icon."""
    icon_html = render_flaticon(icon_name, size=16, margin_right=6) if icon_name else ""
    badge_html = ""
    if badge_text:
        badge_html = f'<div style="margin-top: 0.35rem;"><span class="badge badge-{badge_variant}">{badge_text}</span></div>'

    subtext_html = ""
    if subtext:
        subtext_html = f'<div class="metric-subtext">{subtext}</div>'

    return f"""
    <div class="metric-card">
        <div class="metric-label">{icon_html}{label}</div>
        <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
        {badge_html}
        {subtext_html}
    </div>
    """


def render_disclaimer_banner() -> None:
    """Render the standard DeepMind educational simulation disclaimer."""
    st.markdown(
        """
        <div class="disclaimer-banner">
            <strong>Simulation & Research Notice:</strong>
            This application is an educational simulation inspired by published research from
            Google DeepMind (2016 & 2018). Telemetry is generated via physics-based thermodynamic
            surrogate formulas and evaluated using machine learning algorithms.
            It does not connect to real physical equipment and does not reproduce Google's proprietary
            production control infrastructure. Predicted safety reflects statistical model constraints,
            not certified physical safety.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str) -> None:
    """Render the standard application header."""
    st.markdown(
        f"""
        <div class="main-header">
            <div class="main-title">{title}</div>
            <div class="main-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
