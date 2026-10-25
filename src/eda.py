"""EDA with visualizations saved to reports/."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import reports_dir
from src.ingest import ingest_data


def run_eda():
    data = ingest_data()
    out = reports_dir()
    allc = data[data["country"] == "all"]
    plt.figure(figsize=(9, 4)); plt.plot(allc["date"], allc["revenue"], lw=.8)
    plt.title("Daily revenue (all countries)"); plt.ylabel("Revenue"); plt.tight_layout()
    plt.savefig(os.path.join(out, "eda_revenue_over_time.png"), dpi=120); plt.close()

    tot = data[data["country"] != "all"].groupby("country")["revenue"].sum().sort_values()
    plt.figure(figsize=(7, 4)); tot.plot.barh()
    plt.title("Total revenue by country"); plt.tight_layout()
    plt.savefig(os.path.join(out, "eda_revenue_by_country.png"), dpi=120); plt.close()

    plt.figure(figsize=(7, 4)); allc["revenue"].plot.hist(bins=40)
    plt.title("Distribution of daily revenue"); plt.tight_layout()
    plt.savefig(os.path.join(out, "eda_revenue_distribution.png"), dpi=120); plt.close()

    dow = allc.assign(dow=allc["date"].dt.dayofweek).groupby("dow")["revenue"].mean()
    plt.figure(figsize=(7, 4)); dow.plot.bar()
    plt.title("Average revenue by weekday (0=Mon)"); plt.tight_layout()
    plt.savefig(os.path.join(out, "eda_weekday_pattern.png"), dpi=120); plt.close()
    return sorted(f for f in os.listdir(out) if f.startswith("eda_"))


if __name__ == "__main__":
    print(run_eda())
