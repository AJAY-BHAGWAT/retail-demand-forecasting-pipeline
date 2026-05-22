"""
tests/test_forecast.py
Unit tests for Prophet and ARIMA forecasting modules.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from forecast_prophet import forecast_store, _fallback_forecast
from forecast_arima import forecast_store_arima, _naive_forecast


@pytest.fixture
def weekly_df():
    """Synthetic weekly sales data for two stores."""
    np.random.seed(42)
    weeks = pd.date_range("2023-01-02", periods=52, freq="W-MON")
    rows = []
    for store in ["STORE_001", "STORE_002"]:
        base = 10000
        for i, w in enumerate(weeks):
            revenue = base * (1 + 0.3 * np.sin(2 * np.pi * i / 52)) + np.random.normal(0, 300)
            rows.append({
                "week_start": w,
                "store_id": store,
                "total_revenue": max(revenue, 0),
                "total_quantity": np.random.randint(200, 800),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Prophet / fallback tests
# ------------------------------------------------------------------

def test_fallback_forecast_returns_correct_shape():
    """Fallback must return exactly `periods` rows."""
    history = pd.DataFrame({
        "ds": pd.date_range("2023-01-01", periods=30, freq="W-MON"),
        "y": np.random.rand(30) * 1000,
    })
    result = _fallback_forecast(history, periods=12)
    assert len(result) == 12


def test_fallback_forecast_columns():
    """Fallback must return ds, yhat, yhat_lower, yhat_upper."""
    history = pd.DataFrame({
        "ds": pd.date_range("2023-01-01", periods=20, freq="W-MON"),
        "y": np.random.rand(20) * 1000,
    })
    result = _fallback_forecast(history, periods=8)
    for col in ["ds", "yhat", "yhat_lower", "yhat_upper"]:
        assert col in result.columns


def test_forecast_store_returns_dataframe(weekly_df):
    """forecast_store must return a non-empty DataFrame."""
    result = forecast_store(weekly_df, "STORE_001", forecast_weeks=12)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_forecast_store_has_store_id(weekly_df):
    """Forecast output must include store_id column."""
    result = forecast_store(weekly_df, "STORE_001", forecast_weeks=12)
    assert "store_id" in result.columns
    assert (result["store_id"] == "STORE_001").all()


def test_forecast_store_no_negative_yhat(weekly_df):
    """Forecast values must not be negative."""
    result = forecast_store(weekly_df, "STORE_001", forecast_weeks=12)
    assert (result["yhat"] >= 0).all()


def test_forecast_store_insufficient_data(weekly_df):
    """Stores with < 8 weeks of data must return empty DataFrame."""
    tiny = weekly_df[weekly_df["store_id"] == "STORE_001"].head(4).copy()
    result = forecast_store(tiny, "STORE_001", forecast_weeks=12)
    assert result.empty


# ------------------------------------------------------------------
# ARIMA / naive tests
# ------------------------------------------------------------------

def test_naive_forecast_length():
    """Naive forecast must return exactly `periods` values."""
    series = pd.Series(np.random.rand(30) * 1000)
    result = _naive_forecast(series, 12)
    assert len(result) == 12


def test_arima_forecast_returns_dataframe(weekly_df):
    """forecast_store_arima must return a non-empty DataFrame."""
    result = forecast_store_arima(weekly_df, "STORE_001", forecast_weeks=12)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_arima_forecast_no_negative(weekly_df):
    """ARIMA forecast yhat must be >= 0."""
    result = forecast_store_arima(weekly_df, "STORE_001", forecast_weeks=12)
    assert (result["yhat"] >= 0).all()


def test_arima_forecast_periods(weekly_df):
    """ARIMA must return exactly forecast_weeks rows."""
    result = forecast_store_arima(weekly_df, "STORE_001", forecast_weeks=8)
    assert len(result) == 8
