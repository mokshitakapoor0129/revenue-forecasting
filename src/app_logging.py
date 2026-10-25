"""CSV-style logging for training and prediction events."""
import csv
import os
from datetime import datetime, timezone
from src.config import logs_dir

TRAIN_COLS = ["timestamp", "country", "model", "rmse", "mae", "runtime_s", "version"]
PRED_COLS = ["timestamp", "country", "date", "prediction", "runtime_s", "version"]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(fname, cols, row):
    path = os.path.join(logs_dir(), fname)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(cols)
        w.writerow(row)
    return path


def log_train(country, model, rmse, mae, runtime_s, version):
    return _write("train.log", TRAIN_COLS,
                  [_now(), country, model, round(rmse, 4), round(mae, 4),
                   round(runtime_s, 4), version])


def log_predict(country, date, prediction, runtime_s, version):
    return _write("predict.log", PRED_COLS,
                  [_now(), country, date, round(float(prediction), 4),
                   round(runtime_s, 4), version])


def read_log(name):
    path = os.path.join(logs_dir(), name)
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))
