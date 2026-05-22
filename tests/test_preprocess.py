"""
tests/test_preprocess.py
Unit tests for data cleaning, feature engineering, and weekly aggregation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import clean_data, engineer_features, aggregate_weekly


@pytest.fixture
def raw_df():
    """Minimal synthetic raw sales DataFrame."""
    np.random.seed(0)
    dates = pd.date_range("2023-01-02", periods=30, freq="D")
    rows = []
    for d in dates:
        for store in ["STORE_001", "STORE_002"]:
            rows.append({
                "date": d,
                "store_id": store,
                "product_id": "PROD_0001",
                "quantity": np.random.randint(5, 50),
                "unit_price": 9.99,
                "revenue": np.random.randint(50, 500),
                "day_of_week": d.dayofweek,
                "month": d.month,
                "week_of_year": d.isocalendar()[1],
            })
    return pd.DataFrame(rows)


@pytest.fixture
def dirty_df(raw_df):
    """Add bad rows to test cleaning."""
    bad = pd.DataFrame([
        {"date": pd.Timestamp("2023-01-05"), "store_id": "STORE_001", "product_id": "PROD_0001",
         "quantity": -5, "unit_price": 9.99, "revenue": -50,
         "day_of_week": 3, "month": 1, "week_of_year": 1},
        {"date": None, "store_id": "STORE_001", "product_id": "PROD_0001",
         "quantity": 10, "unit_price": 9.99, "revenue": 100,
         "day_of_week": 3, "month": 1, "week_of_year": 1},
    ])
    return pd.concat([raw_df, bad], ignore_index=True)


def test_clean_removes_negatives(dirty_df):
    """Negative quantities must be removed."""
    cleaned = clean_data(dirty_df)
    assert (cleaned["quantity"] > 0).all()


def test_clean_removes_nulls(dirty_df):
    """Rows with null dates must be removed."""
    cleaned = clean_data(dirty_df)
    assert cleaned["date"].notna().all()


def test_clean_does_not_remove_valid_rows(raw_df):
    """All rows in a clean DataFrame must survive cleaning."""
    cleaned = clean_data(raw_df)
    assert len(cleaned) == len(raw_df)


def test_engineer_features_adds_columns(raw_df):
    """engineer_features must add year, month, week_of_year, is_weekend, week_start."""
    result = engineer_features(raw_df)
    for col in ["year", "month", "week_of_year", "is_weekend", "week_start"]:
        assert col in result.columns, f"Missing engineered column: {col}"


def test_is_weekend_correct(raw_df):
    """is_weekend must be 1 for Saturday/Sunday, 0 otherwise."""
    result = engineer_features(raw_df)
    weekends = result[result["day_of_week"] >= 5]["is_weekend"]
    weekdays = result[result["day_of_week"] < 5]["is_weekend"]
    assert (weekends == 1).all()
    assert (weekdays == 0).all()


def test_aggregate_weekly_granularity(raw_df):
    """Each week_start × store_id combination must appear exactly once."""
    featured = engineer_features(raw_df)
    weekly = aggregate_weekly(featured)
    assert weekly.duplicated(subset=["week_start", "store_id"]).sum() == 0


def test_aggregate_weekly_revenue_positive(raw_df):
    """Total revenue in weekly aggregate must be positive."""
    featured = engineer_features(raw_df)
    weekly = aggregate_weekly(featured)
    assert (weekly["total_revenue"] > 0).all()


def test_aggregate_weekly_columns(raw_df):
    """Weekly aggregate must contain all required columns."""
    featured = engineer_features(raw_df)
    weekly = aggregate_weekly(featured)
    required = ["week_start", "store_id", "total_quantity", "total_revenue",
                "avg_unit_price", "transaction_count", "unique_products"]
    for col in required:
        assert col in weekly.columns, f"Missing column: {col}"
