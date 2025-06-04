FROM python:3.9-slim

WORKDIR /app

# Préparer les outils de base avant installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    curl \
    wget \
    ca-certificates \
    software-properties-common \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Installer OpenJDK 17 proprement (préférable via backports/deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Définir JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Vérification de Java
RUN java -version

# Installer les dépendances Python
COPY services/train-service/requirements-base.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements-base.txt

# Copier le code de l'application
COPY services/train-service/. .

# Ajouter les dépendances GPU conditionnellement
COPY services/train-service/requirements-gpu.txt .
RUN echo '#!/bin/bash\n\
if [ "$USE_GPU" = "true" ]; then\n\
    pip install --no-cache-dir -r requirements-gpu.txt\n\
fi' > /install_gpu.sh && chmod +x /install_gpu.sh

# Ajouter le PYTHONPATH pour les scripts
ENV PYTHONPATH="${PYTHONPATH}:/app/scripts"

# Commande de lancement FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
