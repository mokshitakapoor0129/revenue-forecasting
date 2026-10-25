"""Data ingestion as a reusable function so it can be automated (cron, CI, API)."""
import os
import pandas as pd
from src.config import data_dir


def ingest_data(path=None):
    """Load invoices and return a daily revenue frame per country plus 'all'.
    Columns: date, country, revenue, invoices, views."""
    path = path or os.path.join(data_dir(), "invoices.csv")
    if not os.path.exists(path):
        from src.generate_data import generate
        generate(path=path)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["date", "country", "price"])
    df["country"] = df["country"].str.lower().str.replace(" ", "_")
    daily = (df.groupby(["country", "date"])
               .agg(revenue=("price", "sum"), invoices=("invoice", "nunique"),
                    views=("times_viewed", "sum")).reset_index())
    allc = (df.groupby("date")
              .agg(revenue=("price", "sum"), invoices=("invoice", "nunique"),
                   views=("times_viewed", "sum")).reset_index())
    allc["country"] = "all"
    return pd.concat([daily, allc], ignore_index=True).sort_values(["country", "date"])


def engineer_features(frame):
    """Lag/rolling features for one country's daily series."""
    f = frame.sort_values("date").copy()
    f["lag_1"] = f["revenue"].shift(1)
    f["lag_7"] = f["revenue"].shift(7)
    f["roll_7"] = f["revenue"].shift(1).rolling(7).mean()
    f["roll_30"] = f["revenue"].shift(1).rolling(30).mean()
    f["dow"] = f["date"].dt.dayofweek
    f["month"] = f["date"].dt.month
    f["doy"] = f["date"].dt.dayofyear
    return f.dropna().reset_index(drop=True)


FEATURES = ["lag_1", "lag_7", "roll_7", "roll_30", "dow", "month", "doy"]
