import os
import unittest
from tests.base import IsolatedCase


class ModelTests(IsolatedCase):
    def test_ingest_has_all_and_countries(self):
        from src.ingest import ingest_data
        d = ingest_data()
        self.assertIn("all", set(d["country"]))
        self.assertIn("germany", set(d["country"]))
        self.assertTrue({"date", "revenue"}.issubset(d.columns))

    def test_train_saves_model_and_beats_or_reports_baseline(self):
        from src import model
        from src.config import models_dir
        from src.ingest import ingest_data
        d = ingest_data()
        res = model.train_country(d[d["country"] == "germany"], "germany")
        self.assertIn("baseline_lag7", res["metrics"])
        self.assertGreaterEqual(len(res["metrics"]), 4)  # baseline + 3 models
        self.assertTrue(os.path.exists(os.path.join(models_dir(), "germany.joblib")))
        # saved under the temp dir, not production
        self.assertTrue(models_dir().startswith(self.tmp))

    def test_predict_returns_number(self):
        from src import model
        from src.ingest import ingest_data
        d = ingest_data()
        model.train_country(d[d["country"] == "all"], "all")
        out = model.predict("all", "2018-07-15")
        self.assertIsInstance(out["prediction"], float)
        self.assertGreater(out["prediction"], 0)

    def test_predict_unknown_country_raises(self):
        from src import model
        with self.assertRaises(FileNotFoundError):
            model.predict("atlantis", "2030-01-01")

    def test_drift_check(self):
        from src.monitor import drift_check
        self.assertTrue(drift_check([200] * 5, [100] * 5)["drift"])
        self.assertFalse(drift_check([101] * 5, [100] * 5)["drift"])


if __name__ == "__main__":
    unittest.main()
