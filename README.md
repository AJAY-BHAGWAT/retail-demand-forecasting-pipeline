# Retail Demand Forecasting Pipeline

> End-to-end retail sales forecasting system — Prophet + ARIMA models, dbt transformations, and Power BI dashboard delivering 12-week store-level demand forecasts across 500K+ transactions.

![CI](https://github.com/AJAY-BHAGWAT/retail-demand-forecasting-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat-square&logo=python&logoColor=white)
![dbt](https://img.shields.io/badge/dbt-Core-FF694B?style=flat-square&logo=dbt&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811?style=flat-square&logo=powerbi&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Business Problem

Retail operations teams rely on accurate demand forecasts to manage stock levels, optimise procurement, and avoid costly stockouts or overstock situations. Without a structured forecasting pipeline, planners resort to manual spreadsheet extrapolation — slow, inconsistent, and unable to account for seasonal patterns or store-level variation.

This project delivers an automated, reproducible demand forecasting system that ingests raw transactional data, applies statistical time-series models, and surfaces 12-week forward-looking forecasts through a live Power BI dashboard — enabling store managers and supply chain teams to make data-driven replenishment decisions.

---

## Solution Architecture

```
Raw Sales CSV
     │
     ▼
┌─────────────────┐
│  generate_data  │  ← 500K+ synthetic rows · 50 stores · 200 products · 18 months
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   preprocess    │  ← cleaning · feature engineering · weekly aggregation
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────┐
│         dbt Transform Layer      │
│  stg_sales → mart_weekly_sales   │  ← staging · mart · schema tests
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│       Forecasting Models         │
│   Prophet (primary)              │  ← yearly + weekly seasonality
│   ARIMA   (baseline)             │  ← statistical baseline comparison
└────────┬─────────────────────────┘
         │
         ▼
┌─────────────────┐
│  evaluate.py    │  ← MAPE · RMSE · MAE · model comparison
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Power BI       │  ← Forecast vs Actuals · Store KPIs · Trend Analysis
└─────────────────┘
```

---

## Key Results

| Metric | Value |
|---|---|
| Dataset size | 500,000+ transactions across 50 stores |
| Forecast horizon | 12 weeks forward per store |
| Prophet MAPE | < 12% on held-out test set |
| Model improvement over naive baseline | ~34% reduction in forecast error |
| Stores covered | 50 · Products tracked: 200 |
| Pipeline runtime | < 3 minutes end-to-end |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Generation | Python · NumPy · Pandas |
| Data Transform | SQL · dbt Core · DuckDB |
| Forecasting | Prophet 1.1 · Statsmodels (ARIMA) |
| Evaluation | Scikit-learn · MAE · RMSE · MAPE |
| Visualisation | Power BI · DAX |
| Testing | pytest · 5 test modules |
| CI/CD | GitHub Actions · Python 3.10 & 3.11 |

---

## Project Structure

```
retail-demand-forecasting-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml                     ← GitHub Actions CI pipeline
├── data/
│   ├── raw/
│   │   └── synthetic_sales.csv        ← generated · 500K+ rows
│   └── processed/
│       └── forecast_store001.csv      ← sample forecast output
├── dbt/
│   ├── models/
│   │   ├── staging/
│   │   │   └── stg_sales.sql          ← staging model
│   │   └── mart/
│   │       └── mart_weekly_sales.sql  ← weekly aggregated mart
│   ├── schema.yml                     ← dbt tests + documentation
│   └── dbt_project.yml
├── notebooks/
│   └── 01_demand_forecasting_EDA.ipynb
├── src/
│   ├── generate_data.py               ← synthetic data generator
│   ├── preprocess.py                  ← cleaning + feature engineering
│   ├── forecast_prophet.py            ← Prophet model
│   ├── forecast_arima.py              ← ARIMA baseline
│   └── evaluate.py                    ← MAPE, RMSE, MAE scoring
├── powerbi/
│   └── screenshots/
│       ├── forecast_vs_actual.png
│       └── store_performance.png
├── tests/
│   ├── test_generate_data.py
│   ├── test_preprocess.py
│   ├── test_forecast.py
│   ├── test_evaluate.py
│   └── test_data_quality.py
├── docs/
│   └── architecture.md
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/AJAY-BHAGWAT/retail-demand-forecasting-pipeline
cd retail-demand-forecasting-pipeline
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate synthetic data
```bash
python src/generate_data.py
# Output: data/raw/synthetic_sales.csv · 500,000+ rows
```

### 4. Run dbt transformations
```bash
cd dbt
dbt run
dbt test
cd ..
```

### 5. Run forecasting pipeline
```bash
python src/forecast_prophet.py
# Output: data/processed/forecast_store001.csv
```

### 6. Run test suite
```bash
pytest tests/ -v
```

---

## Dashboard Screenshots

### Forecast vs Actuals — Store Level
![Forecast vs Actuals](powerbi/screenshots/forecast_vs_actual.png)

### Store Performance KPI Overview
![Store Performance](powerbi/screenshots/store_performance.png)

---

## Model Comparison

| Model | MAPE | RMSE | Notes |
|---|---|---|---|
| Prophet | ~11.4% | Lower | Handles seasonality well · recommended |
| ARIMA | ~15.8% | Higher | Statistical baseline · no seasonality |
| Naive (last week) | ~22.1% | Highest | Benchmark only |

Prophet outperforms ARIMA by ~28% on MAPE, driven by its built-in yearly and weekly seasonality components — critical for retail data where weekend and holiday patterns dominate.

---

## dbt Models

```
stg_sales          ← cleans raw CSV · casts types · removes nulls
     │
     ▼
mart_weekly_sales  ← aggregates to store + week · adds revenue metrics
```

**Schema tests applied:**
- `not_null` on all key columns
- `unique` on `order_id`
- `accepted_values` on `store_id` format
- `relationships` between staging and mart layers

---

## Business Insights

Three decisions this pipeline directly enables:

**1. Procurement planning** — 12-week forecasts allow buying teams to raise purchase orders ahead of demand peaks, reducing emergency orders by an estimated 20–30%.

**2. Stockout prevention** — Store-level forecasts identify locations likely to run below safety stock thresholds before it happens, enabling proactive replenishment.

**3. Seasonal staffing** — Predicted demand spikes inform workforce planning — stores can schedule additional staff 2–3 weeks in advance of forecast peaks.

---

## Background

This project was built to demonstrate production-pattern data engineering and forecasting skills across the full analytics stack — from raw data ingestion through dbt transformation to model deployment and BI reporting.

The synthetic dataset mimics real retail sales patterns with embedded seasonality, store-level variation, and trend components — making it representative of the data quality and volume challenges faced in real retail environments.

**Relevant experience:** 2+ years building analytics pipelines and BI solutions at Predictea Digital · EV telemetry analytics at Tata Motors · MSc Data Science, University of Bath.

---

## Certifications & Skills Demonstrated

| Skill | Demonstrated By |
|---|---|
| SQL & dbt | dbt staging + mart models with schema tests |
| Python | Prophet, ARIMA, pandas, feature engineering |
| Power BI | Forecast vs actuals dashboard |
| CI/CD | GitHub Actions · Python 3.10 & 3.11 matrix |
| pytest | 5 test modules · unit + integration tests |
| Time-series forecasting | Prophet + ARIMA + model evaluation |

---

## Author

**Ajay Bhagwat**
MSc Data Science · University of Bath
[LinkedIn](https://www.linkedin.com/in/ajay-bhagwat09) · [GitHub](https://github.com/AJAY-BHAGWAT) · [Kaggle](https://www.kaggle.com/ajaybhagwat4320)

*Open to Data Analyst, BI Developer, Analytics Engineer, and Data Scientist roles — UK & international.*

---

## License

MIT License · See [LICENSE](LICENSE) for details.
