"""
forecast_prophet.py
Fits a Facebook Prophet model per store and generates 12-week forward forecasts.
Handles yearly + weekly seasonality. Outputs per-store forecast CSVs.
"""

import pandas as pd
import numpy as np
import os
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed — using fallback trend model")


def _fallback_forecast(history: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    """
    Simple linear trend fallback when Prophet is not available.
    Returns a DataFrame matching Prophet output schema (ds, yhat, yhat_lower, yhat_upper).
    """
    history = history.copy().sort_values("ds")
    x = np.arange(len(history))
    y = history["y"].values
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs

    last_date = history["ds"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=periods, freq="W-MON")

    future_x = np.arange(len(history), len(history) + periods)
    yhat = slope * future_x + intercept
    yhat = np.maximum(yhat, 0)

    forecast = pd.DataFrame({
        "ds": future_dates,
        "yhat": yhat,
        "yhat_lower": yhat * 0.85,
        "yhat_upper": yhat * 1.15,
    })
    return forecast


def forecast_store(
    weekly_df: pd.DataFrame,
    store_id: str,
    forecast_weeks: int = 12,
) -> pd.DataFrame:
    """
    Fit Prophet (or fallback) on one store's weekly revenue and return forecast DataFrame.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Output of preprocess.aggregate_weekly()
    store_id : str
        Store identifier e.g. 'STORE_001'
    forecast_weeks : int
        Number of weeks to forecast forward (default 12)

    Returns
    -------
    pd.DataFrame with columns: ds, yhat, yhat_lower, yhat_upper, store_id
    """
    store_data = weekly_df[weekly_df["store_id"] == store_id].copy()

    if len(store_data) < 8:
        logger.warning(f"{store_id}: insufficient data ({len(store_data)} weeks) — skipping")
        return pd.DataFrame()

    prophet_df = store_data[["week_start", "total_revenue"]].rename(
        columns={"week_start": "ds", "total_revenue": "y"}
    )
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    if PROPHET_AVAILABLE:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,  # Data is already weekly aggregated
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
            interval_width=0.80,
        )
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_weeks, freq="W-MON")
        forecast = model.predict(future)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_weeks).copy()
    else:
        result = _fallback_forecast(prophet_df, periods=forecast_weeks)

    result["store_id"] = store_id
    result["yhat"] = result["yhat"].clip(lower=0).round(2)
    result["yhat_lower"] = result["yhat_lower"].clip(lower=0).round(2)
    result["yhat_upper"] = result["yhat_upper"].clip(lower=0).round(2)
    result["model"] = "prophet" if PROPHET_AVAILABLE else "linear_trend"

    return result


def run_prophet_pipeline(
    input_path: str = "data/processed/weekly_sales.csv",
    output_dir: str = "data/processed",
    stores: list = None,
    forecast_weeks: int = 12,
) -> pd.DataFrame:
    """
    Run Prophet forecasting for all (or selected) stores.
    Saves per-store CSVs and a combined all_stores_forecast.csv.
    """
    weekly_df = pd.read_csv(input_path, parse_dates=["week_start"])

    if stores is None:
        stores = weekly_df["store_id"].unique().tolist()

    all_forecasts = []
    for store in stores:
        logger.info(f"Forecasting {store}...")
        forecast = forecast_store(weekly_df, store, forecast_weeks)
        if not forecast.empty:
            all_forecasts.append(forecast)
            # Save individual store forecast
            store_path = os.path.join(output_dir, f"forecast_{store.lower()}.csv")
            forecast.to_csv(store_path, index=False)

    combined = pd.concat(all_forecasts, ignore_index=True)
    combined_path = os.path.join(output_dir, "all_stores_forecast.csv")
    combined.to_csv(combined_path, index=False)

    logger.info(f"✅ Prophet forecasting complete")
    logger.info(f"   Stores forecast: {len(all_forecasts)}")
    logger.info(f"   Weeks per store: {forecast_weeks}")
    logger.info(f"   Combined output: {combined_path}")
    return combined


if __name__ == "__main__":
    # Forecast first 5 stores for demo
    df = run_prophet_pipeline(stores=[f"STORE_{i:03d}" for i in range(1, 6)])
    print(df.head(10).to_string(index=False))
