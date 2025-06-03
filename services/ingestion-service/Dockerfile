FROM python:3.9-slim

WORKDIR /app

COPY services/ingestion-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ingestion-service/app.py .
COPY data/ ./data/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
