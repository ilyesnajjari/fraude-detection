from fastapi import FastAPI
app = FastAPI()

import streamlit as st
import requests
import pandas as pd
import os
import time

# URLs des microservices
INGESTION_URL = os.getenv("INGESTION_URL", "http://ingestion-service:8000")
TRAIN_URL = os.getenv("TRAIN_URL", "http://train-service:8000")
PREDICT_URL = os.getenv("PREDICT_URL", "http://predict-service:8000")
COMPARE_URL = os.getenv("COMPARE_URL", "http://compare-service:8000")

HEADERS = {}

st.set_page_config(page_title="Dashboard Détection de Fraude", layout="wide")
st.title("Dashboard Détection de Fraude Bancaire")

# Vérification des microservices AVANT la navigation
services = {
    "Ingestion-service": INGESTION_URL,
    "Train-service": TRAIN_URL,
    "Predict-service": PREDICT_URL,
    "Compare-service": COMPARE_URL,
}
all_services_ok = True
status_dict = {}

with st.sidebar:
    st.header("Statut des microservices")
    for name, url in services.items():
        try:
            res = requests.get(f"{url}/status", headers=HEADERS, timeout=3)
            status = res.json().get("status", "")
            if status.startswith(name.lower().replace("-","") + " running") or "running" in status:
                st.markdown(f"<span style='color:green;font-weight:bold'>{name} : OK</span>", unsafe_allow_html=True)
                status_dict[name] = True
            else:
                st.markdown(f"<span style='color:red;font-weight:bold'>{name} : Indisponible</span>", unsafe_allow_html=True)
                all_services_ok = False
                status_dict[name] = False
        except Exception:
            st.markdown(f"<span style='color:orange;font-weight:bold'>{name} : En attente...</span>", unsafe_allow_html=True)
            all_services_ok = False
            status_dict[name] = False

if not all_services_ok:
    st.header("Statut des microservices")
    st.info("Veuillez patienter, tous les services ne sont pas encore disponibles. Rafraîchissez la page si besoin.")
    st.stop()

# Navigation sidebar (affichée seulement si tous les services sont OK)
page = st.sidebar.radio(
    "Navigation",
    [
        "Statut Services",
        "Prédiction",
        "Résultats des modèles",
        "Monitoring",
        "Lancer l'entraînement Spark"
    ]
)
if "last_page" not in st.session_state:
    st.session_state["last_page"] = page
elif st.session_state["last_page"] != page:
    for key in list(st.session_state.keys()):
        if key not in ["last_page"]:
            del st.session_state[key]
    st.session_state["last_page"] = page

if page == "Statut Services":
    st.header("1. Statut des microservices")
    
    # Variable pour suivre l'état des services
    all_services_ok = True
    
    services = {
        "Ingestion-service": INGESTION_URL,
        "Train-service": TRAIN_URL,
        "Predict-service": PREDICT_URL,
        "Compare-service": COMPARE_URL,
    }
    
    for name, url in services.items():
        try:
            res = requests.get(f"{url}/status", headers=HEADERS, timeout=3)
            status = res.json().get("status", "")
            if status.startswith(name.lower().replace("-","") + " running") or "running" in status:
                st.success(f"{name} : ✅ Validé")
            else:
                st.error(f"{name} : ❌ Non disponible")
                all_services_ok = False
        except Exception:
            st.warning(f"{name} : ⏳ En attente de réponse...")
            all_services_ok = False

    # Compte à rebours et actualisation si tous les services ne sont pas OK
    if not all_services_ok:
        st.warning("Tous les services ne sont pas encore disponibles. Veuillez réessayer plus tard.")
    else:
        st.success("✅ Tous les services sont opérationnels!")

elif page == "Prédiction":
    st.header("2. Prédiction")
    st.info("Les valeurs par défaut sont extraites du fichier CSV de statistiques (creditcard.csv).")

    # Choix de la plateforme de prédiction
    platform = st.selectbox(
        "Choisissez la plateforme de prédiction :",
        options=["spark", "sklearn", "rapids"],
        index=0
    )

    

    # Vérifie les statuts
    ingestion_ok = False
    entrainement_ok = False
    try:
        res = requests.get(f"{INGESTION_URL}/status", headers=HEADERS)
        if res.status_code == 200 and res.json().get("status", "").startswith("ingestion-service running"):
            ingestion_ok = True
    except Exception:
        pass
    try:
        res = requests.get(f"{TRAIN_URL}/status", headers=HEADERS)
        if res.status_code == 200 and res.json().get("status", "").startswith("train-service running"):
            entrainement_ok = True
    except Exception:
        pass

    if not (ingestion_ok and entrainement_ok):
        st.warning("Veuillez attendre que l'ingestion et l'entraînement soient validés avant d'accéder à la prédiction.")
    else:
        features = []
        stats_dict = {}
        mean_dict = {}
        max_wait = 30  # secondes max d'attente
        waited = 0
        with st.spinner("Chargement des features et statistiques du modèle..."):
            while waited < max_wait:
                features_ok = False
                stats_ok = False
                try:
                    features_response = requests.get(f"{PREDICT_URL}/features?platform={platform}", headers=HEADERS, timeout=3)
                    if features_response.status_code == 200:
                        features = features_response.json().get("features", [])
                        features_ok = True
                except Exception:
                    pass
                try:
                    summary_response = requests.get(f"{PREDICT_URL}/summary?platform={platform}", headers=HEADERS, timeout=3)
                    if summary_response.status_code == 200:
                        summary_json = summary_response.json()
                        if platform in summary_json:
                            summary = pd.DataFrame(summary_json[platform])
                        else:
                            summary = pd.DataFrame()
                        if not summary.empty and "summary" in summary.columns:
                            mean_row = summary[summary["summary"] == "mean"]
                        if not mean_row.empty:
                            stats_dict = mean_row.iloc[0].to_dict()
                            mean_dict = mean_row.iloc[0].to_dict()
                            stats_ok = True
                except Exception:
                    pass
                if features_ok and stats_ok:
                    break
                time.sleep(2)
                waited += 2

        if not features or not stats_dict or not mean_dict:
            st.info("⏳ Toujours en attente des features/statistiques du modèle... La page va s'actualiser.")
            time.sleep(2)
        else:
            with st.form("prediction_form"):
                st.write(f"Entrez les caractéristiques de la transaction ({platform}) :")
                inputs = {}
                for feat in features:
                    # Récupère les valeurs min, max, mean pour chaque feature
                    min_val = stats_dict.get(feat, "")
                    max_val = stats_dict.get(feat, "")
                    mean_val = mean_dict.get(feat, "")

                    # Conversion pour affichage scientifique si besoin
                    def fmt(val):
                        try:
                            val = float(val)
                            if abs(val) < 1e-3 or abs(val) > 1e4:
                                return f"{val:.2e}"
                            else:
                                return f"{val}"
                        except Exception:
                            return str(val)

                    label = f"{feat} (min: {fmt(min_val)}, max: {fmt(max_val)}, mean: {fmt(mean_val)})"

                    # Pré-remplir avec la valeur mean, en gardant la notation scientifique si besoin
                    def parse_val(val):
                        try:
                            # Si la valeur est déjà en notation scientifique sous forme de str, garde-la
                            if isinstance(val, str) and ("e" in val or "E" in val):
                                return float(val)
                            return float(val)
                        except Exception:
                            return 0.0

                    val_input = parse_val(mean_val)

                    # Affichage du champ avec label enrichi et valeur scientifique réelle
                    inputs[feat] = st.number_input(label, value=val_input, format="%.6g")
                submitted = st.form_submit_button("Prédire la fraude")
                if submitted:
                    try:
                        payload = {feat: val for feat, val in inputs.items()}
                        res = requests.post(f"{PREDICT_URL}/predict?platform={platform}", json=payload, headers=HEADERS)
                        prediction = res.json()
                        st.success(f"Prédiction ({platform}) : {'Fraude' if prediction.get('prediction', 0) == 1 else 'Non fraude'}")
                        st.write(f"Probabilité : {prediction.get('probability', 0):.2%}")
                    except Exception as e:
                        st.error(f"Erreur predict-service: {e}")

elif page == "Résultats des modèles":
    st.header("4. Résultats des modèles")
    
    # Récupération des résultats
    try:
        res = requests.get(f"{COMPARE_URL}/results", headers=HEADERS)
        results = res.json()
        
        # Extraction des métriques
        metrics_data = [r for r in results if "comparisons" not in r]
        
        if metrics_data:
            # Affichage du tableau des résultats
            df = pd.DataFrame(metrics_data)
            st.subheader("Tableau des résultats")
            st.dataframe(df)
            
            # Affichage des plots
            st.subheader("Visualisations détaillées")
            try:
                plots_res = requests.get(f"{COMPARE_URL}/plots", headers=HEADERS)
                plots = plots_res.json()
                
                if "training_time_scores" in plots:
                    st.image(f"data:image/png;base64,{plots['training_time_scores']}", caption="Temps d'entraînement par plateforme et modèle")
                if "auc" in plots:
                    st.image(f"data:image/png;base64,{plots['auc']}", caption="Score AUC par modèle et plateforme")
                if "auc_boxplot" in plots:
                    st.image(f"data:image/png;base64,{plots['auc_boxplot']}", caption="Distribution des scores AUC par plateforme")
                if "precision" in plots:
                    st.image(f"data:image/png;base64,{plots['precision']}", caption="Précision par modèle et plateforme")
                if "training_time_pie" in plots:
                    st.image(f"data:image/png;base64,{plots['training_time_pie']}", caption="Répartition du temps d'entraînement moyen par plateforme")
            except Exception as e:
                st.warning(f"Impossible de charger les graphiques détaillés : {e}")
        else:
            st.info("Aucun résultat disponible. Lancez d'abord l'entraînement des modèles.")
    except Exception as e:
        st.error(f"Erreur lors de la récupération des résultats : {e}")

elif page == "Monitoring":
    st.header("5. Monitoring des ressources")
    services = {
        "Ingestion": INGESTION_URL,
        "Entraînement": TRAIN_URL,
        "Prédiction": PREDICT_URL,
        "Comparaison": COMPARE_URL,
    }
    monitoring_loaded = False
    for name, url in services.items():
        st.subheader(f"{name} Service")
        cols = st.columns(3)
        try:
            res = requests.get(f"{url}/monitor", headers=HEADERS, timeout=5)
            monitor = res.json()
            cpu = float(monitor.get('cpu', 0))
            ram = float(monitor.get('ram', 0))
            gpu = monitor.get('gpu', None)
            cols[0].metric("CPU Utilisation (%)", f"{cpu:.1f} %")
            cols[0].progress(min(int(cpu), 100))
            cols[1].metric("RAM Utilisation (MB)", f"{ram:.1f} MB")
            ram_max = 8000  # Par exemple, 8 Go
            cols[1].progress(min(int(ram / ram_max * 100), 100))
            if gpu is not None:
                cols[2].metric("GPU Utilisation (%)", f"{gpu:.1f} %")
                cols[2].progress(min(int(gpu), 100))
            else:
                cols[2].info("GPU non détecté")
            monitoring_loaded = True
        except Exception:
            with st.spinner("Chargement des métriques..."):
                time.sleep(1)
    if not monitoring_loaded:
        st.info("⏳ Toujours en attente des métriques... La page va s'actualiser.")
        time.sleep(2)
        st.rerun()
elif page == "Lancer l'entraînement Spark":
    st.header("6. Lancer l'entraînement")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🚀 Lancer l'entraînement Spark (CPU)"):
            with st.spinner("Entraînement Spark en cours..."):
                try:
                    res = requests.get(f"{TRAIN_URL}/train?platform=spark", headers=HEADERS, timeout=600)
                    train_results = res.json()
                    if "error" in train_results.get("spark", {}):
                        st.error(f"Erreur Spark : {train_results['spark']['error']}")
                    else:
                        st.success("Entraînement Spark terminé !")
                        st.json(train_results.get("spark", {}))
                        try:
                            upload_res = requests.post(f"{COMPARE_URL}/upload-results", json=[train_results], headers=HEADERS)
                            st.info("Résultats envoyés à compare-service")
                        except Exception as e:
                            st.warning(f"Impossible d'envoyer les résultats : {e}")
                except Exception as e:
                    st.error(f"Erreur lors de l'entraînement Spark : {e}")

    with col2:
        if st.button("⚡ Lancer l'entraînement RAPIDS (GPU)"):
            with st.spinner("Entraînement RAPIDS en cours..."):
                try:
                    res = requests.get(f"{TRAIN_URL}/train?platform=rapids", headers=HEADERS, timeout=600)
                    train_results = res.json()
                    if "error" in train_results.get("rapids", {}):
                        st.error(f"Erreur RAPIDS : {train_results['rapids']['error']}")
                    else:
                        st.success("Entraînement RAPIDS terminé !")
                        st.json(train_results.get("rapids", {}))
                        try:
                            upload_res = requests.post(f"{COMPARE_URL}/upload-results", json=[train_results], headers=HEADERS)
                            st.info("Résultats envoyés à compare-service")
                        except Exception as e:
                            st.warning(f"Impossible d'envoyer les résultats : {e}")
                except Exception as e:
                    st.error(f"Erreur lors de l'entraînement RAPIDS : {e}")

    with col3:
        if st.button("🧮 Lancer l'entraînement Sklearn (CPU)"):
            with st.spinner("Entraînement Sklearn en cours..."):
                try:
                    res = requests.get(f"{TRAIN_URL}/train?platform=sklearn", headers=HEADERS, timeout=600)
                    train_results = res.json()
                    if "error" in train_results.get("sklearn", {}):
                        st.error(f"Erreur Sklearn : {train_results['sklearn']['error']}")
                    else:
                        st.success("Entraînement Sklearn terminé !")
                        st.json(train_results.get("sklearn", {}))
                        try:
                            upload_res = requests.post(f"{COMPARE_URL}/upload-results", json=[train_results], headers=HEADERS)
                            st.info("Résultats envoyés à compare-service")
                        except Exception as e:
                            st.warning(f"Impossible d'envoyer les résultats : {e}")
                except Exception as e:
                    st.error(f"Erreur lors de l'entraînement Sklearn : {e}")


