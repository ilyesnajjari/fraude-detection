FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for matplotlib and pandas
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copie des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY app.py .

# Copie des données (c'est ici qu'on ajoute le dossier local data/)
VOLUME ["/app/data"]

# Exposition du port
EXPOSE 8000

# Commande de lancement
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
