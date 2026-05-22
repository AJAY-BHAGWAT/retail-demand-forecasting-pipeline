"""
tests/test_generate_data.py
Unit tests for synthetic data generation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from generate_data import generate_sales_data


@pytest.fixture(scope="module")
def sample_df(tmp_path_factory):
    """Generate a small dataset for testing."""
    out = tmp_path_factory.mktemp("data") / "test_sales.csv"
    df = generate_sales_data(
        n_stores=3,
        n_products=5,
        days=60,
        output_path=str(out),
    )
    return df


def test_output_is_dataframe(sample_df):
    """Result must be a pandas DataFrame."""
    assert isinstance(sample_df, pd.DataFrame)


def test_expected_columns(sample_df):
    """All required columns must be present."""
    required = ["date", "store_id", "product_id", "quantity", "unit_price", "revenue"]
    for col in required:
        assert col in sample_df.columns, f"Missing column: {col}"


def test_no_negative_quantities(sample_df):
    """Quantity must always be >= 0."""
    assert (sample_df["quantity"] >= 0).all(), "Found negative quantities"


def test_revenue_equals_qty_times_price(sample_df):
    """Revenue must equal quantity × unit_price (within floating-point tolerance)."""
    computed = (sample_df["quantity"] * sample_df["unit_price"]).round(2)
    assert (computed - sample_df["revenue"]).abs().max() < 0.02


def test_store_count(sample_df):
    """Should have exactly the number of stores requested."""
    assert sample_df["store_id"].nunique() == 3


def test_row_count_is_positive(sample_df):
    """Dataset must have at least one row."""
    assert len(sample_df) > 0


def test_date_range(sample_df):
    """Date range should span the requested number of days."""
    dates = pd.to_datetime(sample_df["date"])
    span = (dates.max() - dates.min()).days
    assert span >= 55, f"Expected ~60 day span, got {span}"


def test_csv_file_created(tmp_path):
    """generate_sales_data must write a CSV to the output path."""
    out = tmp_path / "out.csv"
    generate_sales_data(n_stores=2, n_products=2, days=10, output_path=str(out))
    assert out.exists()
    df = pd.read_csv(str(out))
    assert len(df) > 0
