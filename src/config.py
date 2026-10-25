"""Central config. Tests override paths via the env var REV_BASE_DIR so they
never touch production models or logs."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def base_dir():
    return os.environ.get("REV_BASE_DIR", PROJECT_ROOT)


def models_dir():
    p = os.path.join(base_dir(), "models")
    os.makedirs(p, exist_ok=True)
    return p


def logs_dir():
    p = os.path.join(base_dir(), "logs")
    os.makedirs(p, exist_ok=True)
    return p


def data_dir():
    return os.environ.get("REV_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))


def reports_dir():
    p = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(p, exist_ok=True)
    return p
