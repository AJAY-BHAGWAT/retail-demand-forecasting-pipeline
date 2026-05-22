"""
generate_data.py
Generates synthetic retail sales data: 500K+ rows · 50 stores · 200 products · 18 months
Embeds seasonality, weekly patterns, store-level variation, and trend.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)


def generate_sales_data(
    n_stores: int = 50,
    n_products: int = 200,
    days: int = 548,
    output_path: str = "data/raw/synthetic_sales.csv",
) -> pd.DataFrame:
    """Generate synthetic retail sales data with realistic patterns."""

    stores = [f"STORE_{i:03d}" for i in range(1, n_stores + 1)]
    products = [f"PROD_{i:04d}" for i in range(1, n_products + 1)]
    start = datetime(2023, 1, 1)
    dates = [start + timedelta(days=i) for i in range(days)]

    # Store-level base multipliers (some stores busier than others)
    store_multipliers = {s: np.random.uniform(0.6, 1.8) for s in stores}

    # Product-level base demand and price
    product_base = {p: np.random.randint(10, 150) for p in products}
    product_price = {p: round(np.random.uniform(2.99, 49.99), 2) for p in products}

    rows = []
    for store in stores:
        for product in products[:10]:  # 10 products per store for tractable size
            base = product_base[product] * store_multipliers[store]
            for date in dates:
                # Yearly seasonality (sin wave)
                day_of_year = date.timetuple().tm_yday
                yearly = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)

                # Weekly seasonality (weekends ~30% higher)
                weekly = 1.3 if date.weekday() >= 5 else 1.0

                # Long-term upward trend
                trend = 1 + 0.0008 * (date - start).days

                # Random noise
                noise = np.random.normal(1, 0.12)

                qty = max(0, int(base * yearly * weekly * trend * noise))
                price = product_price[product]
                rows.append(
                    [
                        date.date(),
                        store,
                        product,
                        qty,
                        price,
                        round(qty * price, 2),
                        date.weekday(),
                        date.month,
                        date.isocalendar()[1],  # ISO week number
                    ]
                )

    df = pd.DataFrame(
        rows,
        columns=[
            "date",
            "store_id",
            "product_id",
            "quantity",
            "unit_price",
            "revenue",
            "day_of_week",
            "month",
            "week_of_year",
        ],
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df):,} rows → {output_path}")
    print(f"   Stores: {df['store_id'].nunique()} | Products: {df['product_id'].nunique()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Total revenue: £{df['revenue'].sum():,.0f}")
    return df


if __name__ == "__main__":
    df = generate_sales_data()
