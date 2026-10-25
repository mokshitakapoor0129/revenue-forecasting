import os
import unittest
from tests.base import IsolatedCase


class LoggingTests(IsolatedCase):
    def test_train_log_written(self):
        from src.app_logging import log_train, read_log
        from src.config import logs_dir
        path = log_train("testland", "ridge", 1.5, 1.0, 0.2, "1.0")
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.startswith(self.tmp))
        rows = read_log("train.log")
        self.assertEqual(rows[-1]["country"], "testland")
        self.assertEqual(rows[-1]["model"], "ridge")

    def test_predict_log_written(self):
        from src.app_logging import log_predict, read_log
        log_predict("testland", "2030-01-01", 123.4, 0.01, "1.0")
        rows = read_log("predict.log")
        self.assertEqual(rows[-1]["date"], "2030-01-01")
        self.assertAlmostEqual(float(rows[-1]["prediction"]), 123.4, places=2)

    def test_monitor_report_reads_logs(self):
        from src.app_logging import log_predict, log_train
        from src.monitor import performance_report
        log_train("x", "ridge", 2.0, 1.0, 0.1, "1.0")
        log_predict("x", "2030-01-02", 5.0, 0.02, "1.0")
        r = performance_report()
        self.assertGreaterEqual(r["n_trainings"], 1)
        self.assertIn("mean_predict_runtime_s", r)


if __name__ == "__main__":
    unittest.main()
