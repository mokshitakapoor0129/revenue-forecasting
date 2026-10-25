"""Train several models, compare against a baseline, save the best, predict."""
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.app_logging import log_predict, log_train
from src.config import models_dir, reports_dir
from src.ingest import FEATURES, engineer_features, ingest_data

VERSION = "1.0"


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def candidates():
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=150, random_state=0, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=0),
    }


def train_country(frame, country, plot=False):
    """Train all candidates for one country; return metrics and save best."""
    t0 = time.time()
    feats = engineer_features(frame)
    split = int(len(feats) * 0.8)
    tr, te = feats.iloc[:split], feats.iloc[split:]
    # Baseline: same weekday last week (lag_7)
    results = {"baseline_lag7": {"rmse": _rmse(te["revenue"], te["lag_7"]),
                                 "mae": float(mean_absolute_error(te["revenue"], te["lag_7"]))}}
    fitted = {}
    for name, mdl in candidates().items():
        mdl.fit(tr[FEATURES], tr["revenue"])
        p = mdl.predict(te[FEATURES])
        results[name] = {"rmse": _rmse(te["revenue"], p),
                         "mae": float(mean_absolute_error(te["revenue"], p))}
        fitted[name] = mdl
    best = min(fitted, key=lambda k: results[k]["rmse"])
    final = candidates()[best].fit(feats[FEATURES], feats["revenue"])
    history = frame.sort_values("date")[["date", "revenue"]].tail(60)
    joblib.dump({"model": final, "name": best, "history": history,
                 "version": VERSION, "metrics": results},
                os.path.join(models_dir(), f"{country}.joblib"))
    log_train(country, best, results[best]["rmse"], results[best]["mae"],
              time.time() - t0, VERSION)
    if plot:
        plot_comparison(results, country)
    return {"country": country, "best_model": best, "metrics": results}


def plot_comparison(results, country):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results)
    vals = [results[n]["rmse"] for n in names]
    colors = ["tab:gray" if n.startswith("baseline") else "tab:blue" for n in names]
    plt.figure(figsize=(7, 4))
    plt.bar(names, vals, color=colors)
    plt.ylabel("RMSE (lower is better)")
    plt.title(f"Model vs baseline: {country}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir(), f"model_vs_baseline_{country}.png"), dpi=120)
    plt.close()


def train_all(data=None, countries=None, plot=False):
    data = data if data is not None else ingest_data()
    out = []
    for c in (countries or sorted(data["country"].unique())):
        out.append(train_country(data[data["country"] == c], c, plot=plot))
    return out


def load(country):
    path = os.path.join(models_dir(), f"{country}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model for '{country}'. POST /train first.")
    return joblib.load(path)


def predict(country, date):
    """Predict revenue for `date` (YYYY-MM-DD) via recursive daily forecasting."""
    t0 = time.time()
    art = load(country)
    h = art["history"].sort_values("date")
    d = pd.to_datetime(date)
    rev = list(h["revenue"].values)
    last = h["date"].max()
    if d <= last:
        raise ValueError(f"date must be after last training date {last.date()}")
    cur, p = last, None
    while cur < d:
        cur += pd.Timedelta(days=1)
        row = pd.DataFrame([[rev[-1], rev[-7], np.mean(rev[-7:]), np.mean(rev[-30:]),
                             cur.dayofweek, cur.month, cur.dayofyear]], columns=FEATURES)
        p = float(art["model"].predict(row)[0])
        rev.append(p)
    log_predict(country, date, p, time.time() - t0, art["version"])
    return {"country": country, "date": date, "prediction": round(p, 2),
            "model": art["name"], "version": art["version"]}
