"""Performance monitoring: model error, latency, and prediction drift."""
import numpy as np
from src.app_logging import read_log


def performance_report():
    train = read_log("train.log")
    pred = read_log("predict.log")
    report = {"n_trainings": len(train), "n_predictions": len(pred)}
    if train:
        rmse = [float(r["rmse"]) for r in train]
        report["latest_rmse"] = rmse[-1]
        report["mean_rmse"] = float(np.mean(rmse))
    if pred:
        rt = [float(r["runtime_s"]) for r in pred]
        report["mean_predict_runtime_s"] = float(np.mean(rt))
        report["p95_predict_runtime_s"] = float(np.percentile(rt, 95))
    return report


def drift_check(recent, reference, threshold=0.25):
    """Flag drift when the recent mean differs from the reference mean by more
    than `threshold` (relative)."""
    ref, rec = float(np.mean(reference)), float(np.mean(recent))
    change = abs(rec - ref) / ref if ref else float("inf")
    return {"reference_mean": ref, "recent_mean": rec,
            "relative_change": change, "drift": bool(change > threshold)}
