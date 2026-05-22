"""
tests/test_evaluate.py
Unit tests for evaluation metrics and model comparison.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluate import (
    mean_absolute_percentage_error,
    root_mean_squared_error,
    mean_absolute_error,
    compute_metrics,
    naive_seasonal_forecast,
    evaluate_models,
)


def test_mape_perfect_forecast():
    """MAPE must be 0 when predictions equal actuals."""
    y = np.array([100.0, 200.0, 300.0])
    assert mean_absolute_percentage_error(y, y) == pytest.approx(0.0)


def test_mape_known_value():
    """MAPE of [100, 200] vs [110, 220] must be exactly 10%."""
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 220.0])
    assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(10.0)


def test_mape_ignores_zeros():
    """MAPE must not divide by zero when actuals contain zeros."""
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([10.0, 110.0, 210.0])
    result = mean_absolute_percentage_error(y_true, y_pred)
    assert not np.isnan(result)


def test_rmse_perfect():
    """RMSE must be 0 for identical arrays."""
    y = np.array([1.0, 2.0, 3.0])
    assert root_mean_squared_error(y, y) == pytest.approx(0.0)


def test_rmse_known_value():
    """RMSE([0,0], [3,4]) must equal sqrt((9+16)/2) = sqrt(12.5)."""
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    expected = np.sqrt((9 + 16) / 2)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(expected)


def test_mae_perfect():
    """MAE must be 0 for identical arrays."""
    y = np.array([5.0, 10.0, 15.0])
    assert mean_absolute_error(y, y) == pytest.approx(0.0)


def test_compute_metrics_returns_all_keys():
    """compute_metrics must return MAPE, RMSE, MAE keys."""
    y = np.array([100.0, 200.0])
    metrics = compute_metrics(y, y)
    assert set(metrics.keys()) == {"MAPE", "RMSE", "MAE"}


def test_naive_seasonal_length():
    """Naive forecast must return exactly the requested number of periods."""
    train = np.arange(1, 53, dtype=float)
    result = naive_seasonal_forecast(train, periods=12)
    assert len(result) == 12


def test_evaluate_models_returns_dataframe():
    """evaluate_models must return a DataFrame with expected columns."""
    np.random.seed(0)
    weeks = pd.date_range("2023-01-02", periods=52, freq="W-MON")
    rows = []
    for store in ["STORE_001", "STORE_002"]:
        for w in weeks:
            rows.append({
                "week_start": w,
                "store_id": store,
                "total_revenue": np.random.uniform(5000, 15000),
            })
    weekly_df = pd.DataFrame(rows)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "report.csv")
        report = evaluate_models(weekly_df, stores=["STORE_001", "STORE_002"], output_path=out)

    assert isinstance(report, pd.DataFrame)
    assert "prophet_mape" in report.columns
    assert "arima_mape" in report.columns
    assert "naive_mape" in report.columns
