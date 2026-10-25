import unittest
from tests.base import IsolatedCase


class ApiTests(IsolatedCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from src.app import app
        cls.client = app.test_client()

    def test_index(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_train_then_predict_single_country(self):
        r = self.client.post("/train?country=france")
        self.assertEqual(r.status_code, 200)
        r = self.client.get("/predict?country=france&date=2018-07-20")
        self.assertEqual(r.status_code, 200)
        self.assertIn("prediction", r.get_json())

    def test_train_then_predict_all_countries(self):
        self.assertEqual(self.client.post("/train?country=all").status_code, 200)
        r = self.client.get("/predict?country=all&date=2018-07-20")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["country"], "all")

    def test_predict_missing_date_400(self):
        self.assertEqual(self.client.get("/predict?country=all").status_code, 400)

    def test_predict_untrained_country_404(self):
        self.assertEqual(self.client.get("/predict?country=spain&date=2018-07-20").status_code, 404)

    def test_train_unknown_country_400(self):
        self.assertEqual(self.client.post("/train?country=narnia").status_code, 400)

    def test_monitor_and_logs_endpoints(self):
        self.client.post("/train?country=eire")
        self.assertEqual(self.client.get("/monitor").status_code, 200)
        r = self.client.get("/logs/train")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.get_json(), list)
        self.assertEqual(self.client.get("/logs/bogus").status_code, 404)


if __name__ == "__main__":
    unittest.main()
