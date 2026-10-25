# AI Workflow for Revenue Forecasting: Flask API, Docker, Monitoring and Unit Tests

Capstone project: a deployable AI workflow that forecasts daily revenue per country
(and for all countries combined) and exposes it through a REST API.

> Data note: `src/generate_data.py` creates a synthetic invoice dataset
> (`data/invoices.csv`) with trend, yearly seasonality and weekday effects. To use real data,
> replace that file with one that has the columns `date, country, invoice, price, times_viewed`.

## Project structure
```
src/ingest.py         data ingestion function ingest_data() + feature engineering
src/model.py          trains/compares models, saves the best, predict()
src/app.py            Flask API (/train, /predict, /monitor, /logs/<name>)
src/app_logging.py    training + prediction logs
src/monitor.py        performance monitoring (RMSE, latency, drift check)
src/eda.py            EDA visualizations -> reports/
tests/                unit tests for API, model, logging
run_tests.py          runs ALL unit tests with one command
Dockerfile            containerized API
```

## Quick start
```bash
pip install -r requirements.txt
python -m src.generate_data        # create data
python -m src.eda                  # EDA plots -> reports/
python run_tests.py                # run every unit test (API + model + logging)
python -m src.app                  # start API on :8080
```

## Docker
```bash
docker build -t revenue-forecasting .
docker run -p 8080:8080 revenue-forecasting
```
Models are trained during the image build, so predictions work immediately.

## API examples
```bash
# train one country, or all countries combined
curl -X POST "localhost:8080/train?country=germany"
curl -X POST "localhost:8080/train?country=all"

# predict for a specific country, and for all countries combined
curl "localhost:8080/predict?country=united_kingdom&date=2020-02-01"
curl "localhost:8080/predict?country=all&date=2020-02-01"

# monitoring + logs
curl localhost:8080/monitor
curl localhost:8080/logs/train
curl localhost:8080/logs/predict
```
Valid countries: `united_kingdom, germany, france, eire, spain, all`.
Predictions must be for a date after the last training date (2019-12-31).

## Modeling
Four approaches are compared per country on a time-ordered 80/20 split:
a **baseline** (same weekday last week), **Ridge**, **Random Forest** and **Gradient Boosting**,
using lag/rolling/calendar features. The lowest-RMSE model is refit on all data and saved.
Comparison charts: `reports/model_vs_baseline_<country>.png`. Every model beat the baseline
(e.g. all countries: RMSE 71.9 vs 86.6 baseline).

## Monitoring
`/monitor` reports training RMSE history and prediction latency (mean, p95) from the logs.
`monitor.drift_check()` flags when recent revenue deviates >25% from a reference window.

## Testing and isolation
Tests run against a temporary directory (via `REV_BASE_DIR`), so they never write to the
production `models/` or `logs/` folders.

## EDA
Plots in `reports/`: revenue over time, revenue by country, revenue distribution, weekday pattern.
