"""Flask API: /train, /predict, /monitor, /logs/<name>."""
from flask import Flask, jsonify, request

from src import model as m
from src.app_logging import read_log
from src.ingest import ingest_data
from src.monitor import performance_report

app = Flask(__name__)


@app.get("/")
def index():
    return jsonify({"endpoints": ["POST /train?country=", "GET /predict?country=&date=",
                                  "GET /monitor", "GET /logs/<train|predict>"]})


@app.post("/train")
def train():
    country = (request.args.get("country") or "all").lower()
    data = ingest_data()
    if country not in set(data["country"]):
        return jsonify(error=f"unknown country '{country}'"), 400
    return jsonify(m.train_all(data, countries=[country]))


@app.get("/predict")
def predict():
    country = (request.args.get("country") or "all").lower()
    date = request.args.get("date")
    if not date:
        return jsonify(error="date (YYYY-MM-DD) is required"), 400
    try:
        return jsonify(m.predict(country, date))
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 404
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.get("/monitor")
def monitor():
    return jsonify(performance_report())


@app.get("/logs/<name>")
def logs(name):
    if name not in ("train", "predict"):
        return jsonify(error="name must be train or predict"), 404
    return jsonify(read_log(f"{name}.log"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
