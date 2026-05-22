"""
forecast_arima.py
ARIMA statistical baseline model for retail demand forecasting.
Used as a benchmark against Prophet. Fits ARIMA(1,1,1) per store.
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
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not installed — using naive forecast fallback")


def _naive_forecast(series: pd.Series, periods: int) -> np.ndarray:
    """Naive seasonal forecast: repeat last 4 weeks cyclically."""
    last_values = series.values[-4:]
    reps = (periods // 4) + 1
    return np.tile(last_values, reps)[:periods]


def forecast_store_arima(
    weekly_df: pd.DataFrame,
    store_id: str,
    order: tuple = (1, 1, 1),
    forecast_weeks: int = 12,
) -> pd.DataFrame:
    """
    Fit ARIMA on one store's weekly revenue and return forecast DataFrame.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Output of preprocess.aggregate_weekly()
    store_id : str
        Store identifier e.g. 'STORE_001'
    order : tuple
        ARIMA(p,d,q) order — default (1,1,1)
    forecast_weeks : int
        Number of weeks to forecast forward

    Returns
    -------
    pd.DataFrame with columns: ds, yhat, yhat_lower, yhat_upper, store_id, model
    """
    store_data = weekly_df[weekly_df["store_id"] == store_id].copy()
    store_data = store_data.sort_values("week_start")

    if len(store_data) < 16:
        logger.warning(f"{store_id}: insufficient data ({len(store_data)} weeks) — skipping")
        return pd.DataFrame()

    series = store_data["total_revenue"].values
    last_date = pd.to_datetime(store_data["week_start"].iloc[-1])
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=forecast_weeks,
        freq="W-MON",
    )

    if STATSMODELS_AVAILABLE:
        try:
            model = ARIMA(series, order=order)
            fitted = model.fit()
            forecast_obj = fitted.get_forecast(steps=forecast_weeks)
            yhat = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int(alpha=0.20)  # 80% CI
            yhat_lower = conf_int[:, 0]
            yhat_upper = conf_int[:, 1]
            # Sanity check: if ARIMA produces implausible negatives, fall back
            if (np.maximum(yhat, 0) == 0).all():
                raise ValueError("ARIMA produced all-zero forecasts; using naive fallback")
        except Exception as e:
            logger.warning(f"{store_id}: ARIMA issue ({e}) — using naive fallback")
            yhat = _naive_forecast(pd.Series(series), forecast_weeks)
            yhat_lower = yhat * 0.85
            yhat_upper = yhat * 1.15
    else:
        yhat = _naive_forecast(pd.Series(series), forecast_weeks)
        yhat_lower = yhat * 0.85
        yhat_upper = yhat * 1.15

    result = pd.DataFrame({
        "ds": future_dates,
        "yhat": np.maximum(yhat, 0).round(2),
        "yhat_lower": np.maximum(yhat_lower, 0).round(2),
        "yhat_upper": np.maximum(yhat_upper, 0).round(2),
        "store_id": store_id,
        "model": "arima",
    })

    return result


def run_arima_pipeline(
    input_path: str = "data/processed/weekly_sales.csv",
    output_dir: str = "data/processed",
    stores: list = None,
    forecast_weeks: int = 12,
) -> pd.DataFrame:
    """
    Run ARIMA forecasting for all (or selected) stores.
    Saves combined arima_forecast.csv.
    """
    weekly_df = pd.read_csv(input_path, parse_dates=["week_start"])

    if stores is None:
        stores = weekly_df["store_id"].unique().tolist()

    all_forecasts = []
    for store in stores:
        logger.info(f"ARIMA — {store}...")
        forecast = forecast_store_arima(weekly_df, store, forecast_weeks=forecast_weeks)
        if not forecast.empty:
            all_forecasts.append(forecast)

    combined = pd.concat(all_forecasts, ignore_index=True)
    output_path = os.path.join(output_dir, "arima_forecast.csv")
    combined.to_csv(output_path, index=False)

    logger.info(f"✅ ARIMA forecasting complete → {output_path}")
    return combined


if __name__ == "__main__":
    df = run_arima_pipeline(stores=[f"STORE_{i:03d}" for i in range(1, 6)])
    print(df.head(10).to_string(index=False))
