"""
tests/test_data_quality.py
Data quality checks — validates schema, ranges, and referential integrity
across the synthetic dataset and weekly aggregation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from generate_data import generate_sales_data
from preprocess import clean_data, engineer_features, aggregate_weekly


@pytest.fixture(scope="module")
def full_pipeline(tmp_path_factory):
    """Run the full data generation + preprocessing pipeline."""
    raw_path = str(tmp_path_factory.mktemp("data") / "sales.csv")
    raw_df = generate_sales_data(n_stores=5, n_products=5, days=90, output_path=raw_path)
    clean = clean_data(raw_df)
    featured = engineer_features(clean)
    weekly = aggregate_weekly(featured)
    return raw_df, clean, featured, weekly


def test_raw_no_null_required_columns(full_pipeline):
    """Required columns must have no nulls in raw data."""
    raw_df, *_ = full_pipeline
    for col in ["date", "store_id", "product_id", "quantity", "unit_price", "revenue"]:
        assert raw_df[col].notna().all(), f"Nulls found in {col}"


def test_clean_quantity_positive(full_pipeline):
    """After cleaning, all quantities must be positive."""
    _, clean, *_ = full_pipeline
    assert (clean["quantity"] > 0).all()


def test_clean_revenue_positive(full_pipeline):
    """After cleaning, all revenue values must be positive."""
    _, clean, *_ = full_pipeline
    assert (clean["revenue"] > 0).all()


def test_store_ids_format(full_pipeline):
    """Store IDs must match STORE_XXX format."""
    _, clean, *_ = full_pipeline
    pattern = r"^STORE_\d{3}$"
    assert clean["store_id"].str.match(pattern).all(), "Invalid store_id format found"


def test_product_ids_format(full_pipeline):
    """Product IDs must match PROD_XXXX format."""
    _, clean, *_ = full_pipeline
    pattern = r"^PROD_\d{4}$"
    assert clean["product_id"].str.match(pattern).all(), "Invalid product_id format found"


def test_weekly_no_negative_revenue(full_pipeline):
    """Weekly aggregated revenue must not be negative."""
    *_, weekly = full_pipeline
    assert (weekly["total_revenue"] >= 0).all()


def test_weekly_no_null_week_start(full_pipeline):
    """week_start must have no nulls in the weekly aggregate."""
    *_, weekly = full_pipeline
    assert weekly["week_start"].notna().all()


def test_weekly_store_coverage(full_pipeline):
    """All 5 stores must appear in the weekly aggregate."""
    raw_df, _, _, weekly = full_pipeline
    raw_stores = set(raw_df["store_id"].unique())
    weekly_stores = set(weekly["store_id"].unique())
    assert raw_stores == weekly_stores, f"Stores in weekly don't match raw: {raw_stores ^ weekly_stores}"


def test_weekly_transaction_count_positive(full_pipeline):
    """Transaction count per week must be at least 1."""
    *_, weekly = full_pipeline
    assert (weekly["transaction_count"] >= 1).all()


def test_weekly_unique_key(full_pipeline):
    """(week_start, store_id) must be a unique key in the weekly aggregate."""
    *_, weekly = full_pipeline
    dupes = weekly.duplicated(subset=["week_start", "store_id"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate week_start × store_id combinations"


def test_week_start_is_monday(full_pipeline):
    """week_start dates must all be Mondays (dayofweek == 0)."""
    *_, weekly = full_pipeline
    week_starts = pd.to_datetime(weekly["week_start"])
    non_mondays = (week_starts.dt.dayofweek != 0).sum()
    assert non_mondays == 0, f"{non_mondays} week_start values are not Mondays"
