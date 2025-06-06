import matplotlib
matplotlib.use("Agg")
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from app import app
client = TestClient(app)


@pytest.fixture
def mock_csv_files(tmp_path, monkeypatch):
    # Données de test
    data = {
        "model": ["LogisticRegression", "RandomForest"],
        "auc": [0.91, 0.87],
        "accuracy": [0.88, 0.85],
        "recall": [0.9, 0.86],
        "precision": [0.92, 0.88],
        "training_time": [12.5, 15.7]
    }
    df = pd.DataFrame(data)

    # Création des fichiers CSV temporaires
    spark_path = tmp_path / "resultats_auc_spark.csv"
    rapids_path = tmp_path / "resultats_auc_rapids.csv"
    sklearn_path = tmp_path / "resultats_auc_cpu.csv"

    df.to_csv(spark_path, index=False)
    df.to_csv(rapids_path, index=False)
    df.to_csv(sklearn_path, index=False)

    # Monkeypatch des chemins dans le module app
    monkeypatch.setattr(app, "RESULTS_PATH_SPARK", str(spark_path))
    monkeypatch.setattr(app, "RESULTS_PATH_RAPIDS", str(rapids_path))
    monkeypatch.setattr(app, "RESULTS_PATH_SKLEARN", str(sklearn_path))

    return {
        "spark": spark_path,
        "rapids": rapids_path,
        "sklearn": sklearn_path,
    }


def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "compare-service running"}


def test_get_results_no_file(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    response = client.get("/results")
    assert response.status_code == 404
    assert "error" in response.json()


def test_get_results_with_files(mock_csv_files):
    response = client.get("/results")
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert any("comparisons" in item for item in json_data)


def test_get_plots_with_files(mock_csv_files):
    response = client.get("/plots")
    assert response.status_code == 200
    json_data = response.json()
    assert "training_time_scores" in json_data
    assert "auc" in json_data
    assert "auc_boxplot" in json_data
    assert "precision" in json_data
    assert "training_time_pie" in json_data
    for plot in json_data.values():
        assert isinstance(plot, str)  # base64 encodé


def test_monitor():
    response = client.get("/monitor")
    assert response.status_code == 200
    json_data = response.json()
    assert "cpu" in json_data
    assert "ram" in json_data
    assert "gpu" in json_data