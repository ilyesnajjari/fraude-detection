# Optimisation de la détection de fraude bancaire : Spark CPU vs RAPIDS GPU

Ce projet vise à comparer les performances de Spark (CPU), RAPIDS (GPU) et Sklearn (CPU) pour la détection de fraude bancaire sur le dataset Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).  
L’architecture repose sur des microservices Python (FastAPI), orchestrés via Docker (et Kubernetes), pour permettre l’entraînement, la prédiction et la comparaison des modèles de façon modulaire et scalable.

---

## 📁 Structure des dossiers

- **`data/`**  
  Contient le dataset d’origine (`creditcard.csv`) et les fichiers prétraités ou générés par les pipelines (`*_processed.csv`, `predictions_*.csv`).

- **`models/`**  
  Contient les modèles entraînés (dossiers par plateforme), les statistiques descriptives (`summary_stats_*.csv`) et les résultats d’évaluation (`resultats_auc_*.csv`).

- **`notebooks/`**  
  Notebooks d’exploration, de prototypage et de tests pour Spark, RAPIDS et Sklearn.

- **`services/`**  
  - **`ingest-service/`** : Service d’ingestion et de préparation des données.
  - **`train-service/`** : Service d’entraînement des modèles (Spark, RAPIDS, Sklearn).
    - `spark_pipeline.py` : Pipeline Spark MLlib.
    - `rapids_pipeline.py` : Pipeline RAPIDS cuML.
    - `sklearn_pipeline.py` : Pipeline Sklearn.
    - `app.py` : API FastAPI pour lancer les entraînements.
  - **`predict-service/`** : Service d’inférence (prédiction) en temps réel.
    - `app.py` : API FastAPI pour prédire la fraude à partir d’une transaction.
  - **`compare-service/`** : Service de comparaison et visualisation des résultats.
    - `app.py` : API FastAPI pour comparer les scores, générer des graphiques, etc.
  - **`frontend/`** : Interface utilisateur (Streamlit) pour piloter l’ensemble du pipeline, visualiser les résultats, lancer des prédictions, etc.

- **`k8s/`**  
  Fichiers de déploiement Kubernetes (optionnel).

- **`logs/`**  
  Logs d’exécution des services.

---

## 🏗️ Architecture

L’architecture est basée sur des microservices :

```
[Ingest Service] ---> [Train Service] ---> [Models/Results]
                                  |
                                  v
                        [Predict Service]
                                  |
                                  v
                            [Frontend]
                                  |
                                  v
                        [Compare Service]
```

- **Ingest Service** : Prépare et nettoie les données.
- **Train Service** : Entraîne les modèles sur Spark (CPU), RAPIDS (GPU) et Sklearn (CPU), sauvegarde les modèles et les statistiques.
- **Predict Service** : Sert les modèles pour la prédiction en temps réel via API.
- **Compare Service** : Agrège les résultats, génère des visualisations comparatives (AUC, accuracy, temps d’entraînement, etc.).
- **Frontend** : Interface Streamlit pour piloter, visualiser et comparer.

---

## 🚀 Lancement rapide (Docker)

1. **Prérequis** :  
   - Docker et Docker Compose installés
   - (Optionnel) GPU NVIDIA + drivers pour RAPIDS

2. **Lancer tous les services** :
   ```bash
   docker compose up --build
   ```
    ```bash
docker compose --profile gpu up --build
   ```

3. **Accéder à l’interface** :  
   Ouvre [http://localhost:8501](http://localhost:8501) pour accéder au frontend Streamlit.

---

## ⚙️ Fonctionnalités principales

- **Entraînement multi-plateforme** :  
  Lance l’entraînement sur Spark, RAPIDS ou Sklearn, compare les scores et les temps.

- **Prédiction en temps réel** :  
  Envoie une transaction à l’API pour obtenir une prédiction (fraude ou non).

- **Monitoring** :  
  Suivi de l’utilisation CPU/RAM/GPU de chaque service.

- **Comparaison visuelle** :  
  Visualisation des scores (AUC, accuracy, recall, precision) et des temps d’entraînement par plateforme et modèle.

- **Statistiques descriptives** :  
  Accès aux statistiques du dataset (min, max, mean, stddev) pour chaque pipeline.

---

## 📄 Fichiers importants

- **`data/creditcard.csv`** : Dataset d’origine (non versionné sur GitHub).
- **`models/summary_stats_*.csv`** : Statistiques descriptives par pipeline.
- **`models/resultats_auc_*.csv`** : Résultats d’évaluation des modèles.
- **`models/*_logistic_model/`** : Dossiers contenant les modèles sauvegardés.
- **`services/train-service/spark_pipeline.py`** : Pipeline Spark.
- **`services/train-service/rapids_pipeline.py`** : Pipeline RAPIDS.
- **`services/train-service/sklearn_pipeline.py`** : Pipeline Sklearn.
- **`services/predict-service/app.py`** : API de prédiction.
- **`services/compare-service/app.py`** : API de comparaison.
- **`frontend/app.py`** : Interface utilisateur Streamlit.

---

## 🔒 .gitignore

- Les fichiers volumineux ou sensibles (datasets, modèles, résultats CSV) sont ignorés par défaut.

---

## 👥 Auteurs

- [Ton nom/prénom ou équipe]

---

## 📚 Références

- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Spark MLlib](https://spark.apache.org/mllib/)
- [RAPIDS cuML](https://rapids.ai/)
- [Scikit-learn](https://scikit-learn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)

---

## 📝 Licence

Ce projet est sous licence MIT (ou à adapter selon ton choix).
