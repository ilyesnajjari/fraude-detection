# Dockerfile pour frontend Streamlit

FROM python:3.9-slim

# Installer les dépendances système utiles
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Créer un dossier de travail
WORKDIR /app

# Copier les fichiers requirements.txt (contenant streamlit + autres libs)
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source frontend
COPY . .

# Exposer le port de Streamlit
EXPOSE 8501

# Commande pour lancer Streamlit en mode production
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
