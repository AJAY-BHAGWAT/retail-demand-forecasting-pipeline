import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
stores = [f"STORE_{i:03d}" for i in range(1, 51)]
products = [f"PROD_{i:04d}" for i in range(1, 201)]
start = datetime(2023, 1, 1)
dates = [start + timedelta(days=i) for i in range(548)]

rows = []
for store in stores:
    for product in products[:10]:
        base = np.random.randint(20, 200)
        for date in dates:
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            trend = 1 + 0.001 * (date - start).days
            noise = np.random.normal(1, 0.1)
            qty = max(0, int(base * seasonal * trend * noise))
            price = round(np.random.uniform(2.99, 49.99), 2)
            rows.append([date.date(), store, product, qty, price, qty * price])

df = pd.DataFrame(rows, columns=["date","store_id","product_id","quantity","unit_price","revenue"])
df.to_csv("data/raw/synthetic_sales.csv", index=False)
print(f"Generated {len(df):,} rows")
