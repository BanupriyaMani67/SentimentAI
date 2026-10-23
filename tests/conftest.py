import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "frontend"))


@pytest.fixture(scope="session", autouse=True)
def ensure_model_exists():
    """Train the model once if the artifacts are missing (e.g. fresh clone)."""
    if not (ROOT / "model" / "sentiment_model.joblib").exists():
        subprocess.run([sys.executable, str(ROOT / "model" / "train_model.py")], check=True)


@pytest.fixture()
def client():
    from app.main import app
    with TestClient(app) as test_client:   # "with" runs the startup (model loading)
        yield test_client
