import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from app import app

import pytest
import requests
from unittest.mock import patch, MagicMock

class MockResponse:
    def __init__(self, json_data, status_code):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json
    
# Dictionnaire des URL à tester
services = {
    "Ingestion-service": "http://ingestion-service:8000",
    "Train-service": "http://train-service:8000",
    "Predict-service": "http://predict-service:8000",
    "Compare-service": "http://compare-service:8000",
}


def mocked_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json = json_data
            self.status_code = status_code

        def json(self):
            return self._json

    url = args[0]
    if "ingestion-service" in url:
        return MockResponse({"status": "ingestion-service running"}, 200)
    if "train-service" in url:
        return MockResponse({"status": "train-service running"}, 200)
    if "predict-service" in url:
        return MockResponse({"status": "predict-service running"}, 200)
    if "compare-service" in url:
        return MockResponse({"status": "compare-service running"}, 200)
    return MockResponse(None, 404)


@patch("requests.get", side_effect=mocked_requests_get)
def test_services_status(mock_secrets, mock_get):
    for name, url in services.items():
        res = requests.get(f"{url}/status")
        assert res.status_code == 200
        assert "running" in res.json()["status"]