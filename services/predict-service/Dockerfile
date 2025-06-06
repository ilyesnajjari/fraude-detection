FROM python:3.9-bullseye

# Évite les prompts lors des installations (Debian)
ENV DEBIAN_FRONTEND=noninteractive


# Installer Java + utilitaires nécessaires pour wget/apt
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-11-jre-headless wget procps ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Variables d'environnement pour Spark
ENV SPARK_VERSION=3.3.0
ENV HADOOP_VERSION=3

# Télécharger et installer Spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /opt && \
    ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Configuration de Spark et Java
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:/opt/spark/bin
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pyspark==3.3.0


# Copier le code applicatif
WORKDIR /app
COPY . .

# Commande de démarrage du service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
