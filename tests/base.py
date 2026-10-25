"""Shared fixture: every test runs against a temp directory so production
models/ and logs/ are never read or written."""
import os
import shutil
import tempfile
import unittest


class IsolatedCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="rev_test_")
        os.environ["REV_BASE_DIR"] = cls.tmp
        os.environ["REV_DATA_DIR"] = os.path.join(cls.tmp, "data")
        from src.generate_data import generate
        generate(days=150, path=os.path.join(cls.tmp, "data", "invoices.csv"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        os.environ.pop("REV_BASE_DIR", None)
        os.environ.pop("REV_DATA_DIR", None)
