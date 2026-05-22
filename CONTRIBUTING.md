# Contributing to retail-demand-forecasting-pipeline

Thank you for your interest in contributing! This project welcomes bug reports, feature suggestions, and pull requests.

---

## Getting Started

### 1. Fork and clone

```bash
git clone https://github.com/<your-username>/retail-demand-forecasting-pipeline
cd retail-demand-forecasting-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the test suite before making changes

```bash
pytest tests/ -v
```

All tests must pass before submitting a pull request.

---

## Contribution Guidelines

### Code style

- Follow PEP 8. Max line length: 100 characters.
- Use descriptive variable names.
- Add docstrings to all functions (Google style).
- Run `flake8 src/ tests/` before committing.

### Adding a new forecasting model

1. Create `src/forecast_<model_name>.py` following the pattern in `forecast_prophet.py`.
2. Expose a `forecast_store(weekly_df, store_id, forecast_weeks)` function returning a DataFrame with columns `[ds, yhat, yhat_lower, yhat_upper, store_id, model]`.
3. Add at least 5 unit tests in `tests/test_forecast_<model_name>.py`.
4. Update `src/evaluate.py` to include the new model in the comparison report.
5. Update README.md with the new model in the Model Comparison table.

### Adding a new dbt model

1. Add your SQL file to `dbt/models/staging/` or `dbt/models/mart/`.
2. Document all columns in `dbt/schema.yml`.
3. Add `not_null` tests for all key columns.

---

## Submitting a Pull Request

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes with clear commit messages.
3. Push and open a PR against `main`.
4. Describe what the PR does and why in the PR description.
5. Ensure CI passes (green badge).

---

## Reporting Bugs

Open a GitHub Issue with:
- A clear description of the bug
- Steps to reproduce
- Expected vs actual behaviour
- Python version and OS

---

## Questions

Reach out via [LinkedIn](https://www.linkedin.com/in/ajay-bhagwat09) or open a GitHub Discussion.
