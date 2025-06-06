from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest
from services.predict_service.app import app  

from app import app  # importe normalement

client = TestClient(app)

# Transaction de test avec des valeurs réalistes
sample_transaction = {
    "Time": 10000.0,
    "V1": -1.3598,
    "V2": -0.0728,
    "V3": 2.5363,
    "V4": 1.3781,
    "V5": -0.3383,
    "V6": 0.4624,
    "V7": 0.2396,
    "V8": 0.0987,
    "V9": 0.3638,
    "V10": 0.0908,
    "V11": -0.5516,
    "V12": -0.6178,
    "V13": -0.9913,
    "V14": -0.3111,
    "V15": -0.3072,
    "V16": 0.6100,
    "V17": 0.0766,
    "V18": 0.1285,
    "V19": 0.1892,
    "V20": 0.1335,
    "V21": -0.0210,
    "V22": 0.0156,
    "V23": 0.0525,
    "V24": 0.0462,
    "V25": 0.1045,
    "V26": 0.5249,
    "V27": 0.2514,
    "V28": 0.0821,
    "Amount": 149.62
}

def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "predict-service running"}

from unittest.mock import patch, MagicMock

@patch("os.path.exists", return_value=True)
@patch("pyspark.ml.classification.LogisticRegressionModel.load")
def test_predict(mock_load_model, mock_exists):
    # Mock du modèle
    mock_model = MagicMock()
    mock_row = MagicMock()
    mock_row.prediction = 1.0
    mock_row.probability = [0.1, 0.9]  # [probabilité pour 0, probabilité pour 1]

    # simulate Spark model.transform() behavior
    mock_model.transform.return_value.collect.return_value = [mock_row]
    mock_load_model.return_value = mock_model

    response = client.post("/predict/", json=sample_transaction)

    assert response.status_code == 200
    result = response.json()

    assert "prediction" in result
    assert result["prediction"] == 1
    assert "probability" in result
    assert 0.8 < result["probability"] < 1.0

def test_features():
    response = client.get("/features")
    assert response.status_code == 200
    json_data = response.json()
    assert "features" in json_data
    assert isinstance(json_data["features"], list)
    assert "Time" in json_data["features"]

def test_monitor():
    response = client.get("/monitor")
    assert response.status_code == 200
    json_data = response.json()
    assert "cpu" in json_data
    assert "ram" in json_data
    assert "gpu" in json_data  # peut être None mais doit exister

def test_summary():
    response = client.get("/summary")
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, dict)
    for key in ["spark", "sklearn", "rapids"]:
        assert key in json_data
