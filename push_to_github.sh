#!/usr/bin/env bash
# push_to_github.sh
# Run this from inside the retail-demand-forecasting-pipeline folder
# after copying all new files from Claude's delivery.

set -e

echo "=== Pushing retail-demand-forecasting-pipeline to GitHub ==="

# Make sure you're in the right directory
if [ ! -f "requirements.txt" ]; then
  echo "❌ Error: Run this from inside the repo root"
  exit 1
fi

git add .
git status

echo ""
echo "Committing all files..."
git commit -m "feat: complete pipeline — src/ modules, dbt models, pytest suite, CI/CD

- Add src/preprocess.py: cleaning, feature engineering, weekly aggregation
- Add src/forecast_prophet.py: Prophet + linear trend fallback
- Add src/forecast_arima.py: ARIMA(1,1,1) + naive fallback
- Add src/evaluate.py: MAPE, RMSE, MAE model comparison
- Add dbt/models/staging/stg_sales.sql
- Add dbt/models/mart/mart_weekly_sales.sql (with WoW growth, rolling avg)
- Add dbt/schema.yml: full column docs + not_null tests
- Add dbt/dbt_project.yml
- Add 46 pytest tests across 5 test modules (100% pass)
- Add .github/workflows/ci.yml: Python 3.10 + 3.11 matrix + flake8
- Add LICENSE (MIT) + CONTRIBUTING.md
- Update requirements.txt with pinned versions"

echo ""
echo "Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Done! Check:"
echo "   → https://github.com/AJAY-BHAGWAT/retail-demand-forecasting-pipeline"
echo "   → Actions tab for CI badge"
