FROM python:3.9-slim

WORKDIR /app

# Installer uniquement ce qui est nécessaire
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers nécessaires
COPY services/ingestion-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ingestion-service/app.py ./app.py

# Copie des données (depuis le dossier racine du projet)
COPY data/ ./data/

# Lancement du service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
