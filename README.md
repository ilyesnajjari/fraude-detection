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
    - `test_predict.py` : Tests unitaires pour l’API de prédiction.
  - **`compare-service/`** : Service de comparaison et visualisation des résultats.
    - `app.py` : API FastAPI pour comparer les scores, générer des graphiques, etc.
    - `test_compare.py` : Tests unitaires pour l’API de comparaison.
  - **`frontend/`** : Interface utilisateur (Streamlit) pour piloter l’ensemble du pipeline, visualiser les résultats, lancer des prédictions, etc.

- **`k8s/`**  
  Fichiers de déploiement Kubernetes :
  - `train-deployment.yaml`, `train-service.yaml`
  - `predict-deployment.yaml`, `predict-service.yaml`
  - `ingestion-deployment.yaml`, `ingestion-service.yaml`
  - `compare-deployment.yaml`, `compare-service.yaml`
  - `frontend-deployment.yaml`, `frontend-service.yaml`

- **`.github/workflows/ci.yml`**  
  Pipeline CI/CD GitHub Actions pour tests automatiques à chaque push/pull request.

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

## 📦 Gestion des images Docker spécialisées

Pour optimiser le déploiement et la modularité, chaque type de ressource (data, models, scripts) possède sa propre image Docker :

- **`Dockerfile.data`** :  
  Contient `data/`, `models/` et `scripts/`.
- **`Dockerfile.models`** :  
  Contient uniquement `models/`.
- **`Dockerfile.scripts`** :  
  Contient uniquement `scripts/`.

**Exemple de build et push pour chaque image :**
```sh
docker build -f Dockerfile.data -t ilyesnajjari/data-image:v1 .
docker push ilyesnajjari/data-image:v1

docker build -f Dockerfile.models -t ilyesnajjari/models-image:v1 .
docker push ilyesnajjari/models-image:v1

docker build -f Dockerfile.scripts -t ilyesnajjari/scripts-image:v1 .
docker push ilyesnajjari/scripts-image:v1
```

---

## ☸️ Utilisation des init containers Kubernetes

Les init containers sont utilisés pour copier les ressources (data, models, scripts) dans des volumes partagés avant le démarrage des services principaux.  
Exemple pour les modèles :

```yaml
initContainers:
  - name: models-init
    image: ilyesnajjari/models-image:v1
    command: ["sh", "-c", "ls -l /models; mkdir -p /models-dest; cp -r /models/. /models-dest/ || true"]
    volumeMounts:
      - name: models-volume
        mountPath: /models-dest
containers:
  - name: predict
    image: ilyesnajjari/predict-service:latest
    volumeMounts:
      - name: models-volume
        mountPath: /app/models
```

---

## 🔍 Commandes utiles pour le debug

- **Vérifier le contenu d’un dossier dans une image Docker :**
  ```sh
  docker run --rm -it ilyesnajjari/models-image:v1 ls -l /models
  docker run --rm -it ilyesnajjari/scripts-image:v1 ls -l /scripts
  ```

- **Accéder à un pod Kubernetes :**
  ```sh
  kubectl exec -it <pod-name> -- sh
  ```

- **Voir les logs d’un init container :**
  ```sh
  kubectl logs <pod-name> -c models-init
  ```

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

## ☸️ Déploiement Kubernetes

1. **Prérequis** :  
   - Un cluster Kubernetes local (Minikube, Docker Desktop) ou cloud (GKE, AKS, EKS)
   - Images Docker poussées sur Docker Hub

2. **Déployer tous les services** :
   ```bash
   kubectl apply -f k8s/
   ```

3. **Vérifier le déploiement** :
   ```bash
   kubectl get pods
   kubectl get services
   ```

4. **Accéder au frontend** :  
   - Récupère le port NodePort ou utilise :
     ```bash
     kubectl port-forward service/frontend-service 8501:80
     ```
   - Puis ouvre [http://localhost:8501](http://localhost:8501)

---

## 🧪 Tests unitaires

- Chaque microservice possède ses propres tests unitaires (ex : `services/predict-service/test_predict.py`).
- Les tests utilisent `pytest` et `unittest.mock` pour simuler les dépendances Spark, accès disque, etc.

**Exemple pour lancer tous les tests :**
```bash
pytest services/predict-service/test_predict.py
pytest services/compare-service/test_compare.py
```

---

## ⚙️ Intégration Continue (CI/CD)

- Une pipeline GitHub Actions (`.github/workflows/ci.yml`) exécute automatiquement les tests à chaque push ou pull request sur `main`.

**Extrait du workflow :**
```yaml
- name: Install dependencies for all services
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt || true
    for req in services/*/requirements.txt; do
      pip install -r "$req"
    done
    if [ -f frontend/requirements.txt ]; then pip install -r frontend/requirements.txt; fi

- name: Run tests
  run: |
    pytest
```

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
