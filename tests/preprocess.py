"""
preprocess.py
Cleans raw sales data, engineers features, and aggregates to weekly store-level granularity.
Output: data/processed/weekly_sales.csv — ready for forecasting models.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str = "data/raw/synthetic_sales.csv") -> pd.DataFrame:
    """Load raw sales CSV and enforce dtypes."""
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath, parse_dates=["date"])
    logger.info(f"Loaded {len(df):,} rows")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove nulls, negative quantities, and zero-revenue rows.
    Returns cleaned DataFrame.
    """
    original_len = len(df)

    # Drop nulls
    df = df.dropna(subset=["date", "store_id", "product_id", "quantity", "revenue"])

    # Remove negative or zero quantities
    df = df[df["quantity"] > 0]

    # Remove zero revenue
    df = df[df["revenue"] > 0]

    # Remove future dates (handle both datetime.date and Timestamp types)
    df = df[pd.to_datetime(df["date"]) <= pd.Timestamp.today()]

    removed = original_len - len(df)
    logger.info(f"Cleaned data: removed {removed:,} rows ({removed/original_len*100:.1f}%)")
    return df.copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features for downstream modelling."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["quarter"] = df["date"].dt.quarter

    # Week start date (Monday) for weekly aggregation join key
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.normalize()

    logger.info("Feature engineering complete")
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily transactions to weekly store-level metrics.
    Output columns: week_start, store_id, total_quantity, total_revenue,
                    avg_unit_price, transaction_count, unique_products
    """
    weekly = (
        df.groupby(["week_start", "store_id"])
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
            avg_unit_price=("unit_price", "mean"),
            transaction_count=("quantity", "count"),
            unique_products=("product_id", "nunique"),
        )
        .reset_index()
    )

    weekly["avg_unit_price"] = weekly["avg_unit_price"].round(2)
    weekly["total_revenue"] = weekly["total_revenue"].round(2)
    weekly = weekly.sort_values(["store_id", "week_start"]).reset_index(drop=True)

    logger.info(
        f"Weekly aggregation: {len(weekly):,} rows | "
        f"{weekly['store_id'].nunique()} stores | "
        f"{weekly['week_start'].nunique()} weeks"
    )
    return weekly


def run_preprocessing(
    input_path: str = "data/raw/synthetic_sales.csv",
    output_path: str = "data/processed/weekly_sales.csv",
) -> pd.DataFrame:
    """End-to-end preprocessing pipeline."""
    df_raw = load_raw_data(input_path)
    df_clean = clean_data(df_raw)
    df_features = engineer_features(df_clean)
    df_weekly = aggregate_weekly(df_features)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_weekly.to_csv(output_path, index=False)
    logger.info(f"✅ Preprocessed data saved → {output_path}")
    return df_weekly


if __name__ == "__main__":
    run_preprocessing()
