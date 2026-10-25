"""Creates synthetic invoice-level data.
Replace data/invoices.csv with real data if you have it; the pipeline only
needs the columns: date, country, invoice, price, times_viewed."""
import os
import numpy as np
import pandas as pd
from src.config import data_dir


def generate(seed=42, days=730, path=None):
    rng = np.random.default_rng(seed)
    countries = {"united_kingdom": 1.0, "germany": 0.35, "france": 0.30,
                 "eire": 0.25, "spain": 0.15}
    dates = pd.date_range("2018-01-01", periods=days, freq="D")
    rows = []
    for c, scale in countries.items():
        for i, d in enumerate(dates):
            season = 1 + 0.3 * np.sin(2 * np.pi * d.dayofyear / 365)
            weekday = 0.6 if d.dayofweek >= 5 else 1.0
            trend = 1 + i / days * 0.4
            n = max(1, int(rng.poisson(8 * scale * season * weekday * trend)))
            for k in range(n):
                rows.append((d.strftime("%Y-%m-%d"), c, f"{c[:2]}{i}-{k}",
                             round(float(rng.gamma(2.0, 6.0)), 2),
                             int(rng.integers(1, 40))))
    df = pd.DataFrame(rows, columns=["date", "country", "invoice", "price", "times_viewed"])
    path = path or os.path.join(data_dir(), "invoices.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    print("wrote", generate())
