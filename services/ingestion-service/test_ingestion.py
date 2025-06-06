import pytest
from fastapi.testclient import TestClient
from app import app
import os
from services.ingestion_service.app import app  # <-- Chemin correct !

client = TestClient(app)

def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ingestion-service running"}


def test_upload_file(tmp_path):
    # Créer un fichier temporaire à envoyer
    file_path = tmp_path / "test_data.csv"
    file_content = "col1,col2\n1,2\n3,4"
    file_path.write_text(file_content)

    with open(file_path, "rb") as f:
        response = client.post("/upload/", files={"file": ("test_data.csv", f, "text/csv")})

    assert response.status_code == 200
    json_data = response.json()
    assert json_data["filename"] == "test_data.csv"
    assert json_data["status"] == "uploaded"

    # Vérifie que le fichier a bien été enregistré
    saved_file = os.path.join("data", "test_data.csv")
    assert os.path.exists(saved_file)

    # Nettoyage
    os.remove(saved_file)
    if os.path.exists("data") and not os.listdir("data"):
        os.rmdir("data")


def test_monitor():
    response = client.get("/monitor")
    assert response.status_code == 200
    json_data = response.json()
    assert "cpu" in json_data
    assert "ram" in json_data
    assert "gpu" in json_data
