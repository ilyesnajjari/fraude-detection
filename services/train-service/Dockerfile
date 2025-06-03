FROM python:3.9-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des requirements
COPY services/train-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie des scripts Python
COPY services/train-service/app.py .
COPY scripts/spark_pipeline.py .
COPY scripts/rapids_pipeline.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]