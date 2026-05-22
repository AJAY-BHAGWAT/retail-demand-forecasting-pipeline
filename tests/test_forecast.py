import pandas as pd
import pytest
import sys
sys.path.insert(0, "src")
from forecast_prophet import run_forecast

@pytest.fixture
def sample_df():
    return pd.read_csv("data/raw/synthetic_sales.csv")

def test_forecast_returns_dataframe(sample_df):
    forecast, mape = run_forecast(sample_df, "STORE_001")
    assert isinstance(forecast, pd.DataFrame)

def test_forecast_has_required_columns(sample_df):
    forecast, _ = run_forecast(sample_df, "STORE_001")
    assert "yhat" in forecast.columns
    assert "ds" in forecast.columns

def test_mape_is_reasonable(sample_df):
    _, mape = run_forecast(sample_df, "STORE_001")
    assert mape < 30, f"MAPE too high: {mape}%"

def test_forecast_length(sample_df):
    forecast, _ = run_forecast(sample_df, "STORE_001", periods=12)
    assert len(forecast) > 12

def test_no_negative_forecasts(sample_df):
    forecast, _ = run_forecast(sample_df, "STORE_001")
    assert (forecast["yhat"] >= 0).all()
