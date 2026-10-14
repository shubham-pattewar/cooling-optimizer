"""
Unit tests for app.py
=====================
Validates that all application pages and utility functions exist and can be loaded.
"""

import inspect
import pytest
import app


def test_app_pages_exist():
    """Verify that all 7 required dashboard pages are defined in app.py."""
    required_functions = [
        "render_page_overview",
        "render_page_optimizer",
        "render_page_explorer",
        "render_page_evaluation",
        "render_page_savings",
        "render_page_case_study",
        "render_page_about",
        "render_sidebar",
        "main",
    ]
    for func_name in required_functions:
        assert hasattr(app, func_name), f"Missing page renderer function: {func_name}"
        assert callable(getattr(app, func_name))


def test_app_dataset_and_models_cache():
    """Verify data and model loaders execute without error."""
    df = app.get_dataset()
    assert len(df) >= 2000
    assert "cooling_power_kw" in df.columns

    p_model, t_model, metadata = app.get_models()
    assert hasattr(p_model, "predict")
    assert hasattr(t_model, "predict")
    assert "cooling_power_metrics" in metadata


def test_utils_styling():
    """Verify utils design tokens and Plotly theme styling."""
    import plotly.graph_objects as go
    from utils import apply_plotly_theme, render_metric_card

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
    fig = apply_plotly_theme(fig, height=300)
    assert fig.layout.height == 300
    assert fig.layout.xaxis.title.font.size == 12

    card_html = render_metric_card(
        label="Test Metric",
        value=100.5,
        unit="kW",
        subtext="Sample",
        badge_text="Active",
        badge_variant="success",
        icon_name="snowflake",
    )
    assert "Test Metric" in card_html
    assert "100.5" in card_html
    assert "badge-success" in card_html
    assert "data:image/png;base64" in card_html


def test_flaticon_rendering():
    """Verify Flaticon vector icon loading and HTML tag generation."""
    from utils import get_icon_base64, render_flaticon

    # Existing icon
    img_tag = render_flaticon("snowflake", size=24)
    assert '<img src="data:image/png;base64,' in img_tag
    assert 'width="24"' in img_tag
    assert 'alt="snowflake icon"' in img_tag

    # Missing icon returns empty string safely
    missing_tag = render_flaticon("non_existent_icon_12345")
    assert missing_tag == ""


def test_all_pages_render_via_apptest():
    """Verify that all 7 pages in the app execute and render without unhandled exceptions."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Initial page load exceptions: {[e.value for e in at.exception]}"

    pages = [
        "Overview Dashboard",
        "Cooling Optimizer",
        "Sensor Data Explorer",
        "ML Model Evaluation",
        "Energy Savings Analysis",
        "Case Study: Google DeepMind",
        "About the Project",
    ]

    for page in pages:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"Exceptions on page '{page}': {[e.value for e in at.exception]}"


