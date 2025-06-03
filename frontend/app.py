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

AUTH_TOKEN = st.secrets.get("auth_token", None)
HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}

st.set_page_config(page_title="Dashboard Détection de Fraude", layout="wide")
st.title("Dashboard Détection de Fraude Bancaire")

# Navigation sidebar
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
        placeholder = st.empty()
        for seconds in range(20, 0, -1):
            placeholder.warning(f"⏳ Actualisation dans {seconds} secondes...")
            time.sleep(1)
        placeholder.empty()
        st.experimental_rerun()
    else:
        st.success("✅ Tous les services sont opérationnels!")

elif page == "Prédiction":
    st.header("2. Prédiction")
    st.info("Les valeurs par défaut sont extraites du fichier CSV de statistiques (creditcard.csv).")
    stat_choice = st.selectbox(
        "Choisissez la statistique pour pré-remplir les champs (sauf Time et Amount qui prennent toujours la moyenne) :",
        options=["min", "max", "mean"],
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
                    features_response = requests.get(f"{PREDICT_URL}/features", headers=HEADERS, timeout=3)
                    if features_response.status_code == 200:
                        features = features_response.json().get("features", [])
                        features_ok = True
                except Exception:
                    pass
                try:
                    summary_response = requests.get(f"{PREDICT_URL}/summary", headers=HEADERS, timeout=3)
                    if summary_response.status_code == 200:
                        summary = pd.DataFrame(summary_response.json())
                        if not summary.empty and "summary" in summary.columns:
                            stat_row = summary[summary["summary"] == stat_choice]
                            if not stat_row.empty:
                                stats_dict = stat_row.iloc[0].to_dict()
                            mean_row = summary[summary["summary"] == "mean"]
                            if not mean_row.empty:
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
            st.experimental_rerun()
        else:
            with st.form("prediction_form"):
                st.write("Entrez les caractéristiques de la transaction :")
                inputs = {}
                for feat in features:
                    if feat in ["Time", "Amount"]:
                        default = float(mean_dict.get(feat, 0.0))
                        st.caption(f"{feat} (pré-rempli avec la moyenne du CSV)")
                    else:
                        default = float(stats_dict.get(feat, 0.0))
                        st.caption(f"{feat} (pré-rempli avec {stat_choice} du CSV)")
                    inputs[feat] = st.number_input(f"{feat}", value=default)
                submitted = st.form_submit_button("Prédire la fraude")
                if submitted:
                    try:
                        payload = {feat: val for feat, val in inputs.items()}
                        res = requests.post(f"{PREDICT_URL}/predict", json=payload, headers=HEADERS)
                        prediction = res.json()
                        st.success(f"Prédiction : {'Fraude' if prediction.get('prediction', 0) == 1 else 'Non fraude'}")
                        st.write(f"Probabilité : {prediction.get('probability', 0):.2%}")
                    except Exception as e:
                        st.error(f"Erreur predict-service: {e}")

elif page == "Résultats des modèles":
    st.header("4. Résultats des modèles")
    
    # Récupération des résultats
    try:
        res = requests.get(f"{COMPARE_URL}/results", headers=HEADERS)
        results = res.json()
        
        # Extraction des métriques et comparaisons
        metrics_data = [r for r in results if "comparisons" not in r]
        comparisons = next((r["comparisons"] for r in results if "comparisons" in r), None)
        
        if metrics_data:
            # Affichage du tableau des résultats
            df = pd.DataFrame(metrics_data)
            st.subheader("Tableau des résultats")
            st.dataframe(df)
            
            # Affichage des comparaisons
            if comparisons:
                st.subheader("Comparaisons des performances")
                
                # Temps d'entraînement
                if comparisons["training_times"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("⏱️ Temps d'entraînement moyen par plateforme")
                        times_df = pd.DataFrame(list(comparisons["training_times"].items()), 
                                             columns=["Platform", "Time (s)"])
                        st.bar_chart(times_df.set_index("Platform"))
                
                # Scores AUC
                if comparisons["auc_scores"]:
                    with col2:
                        st.write("📈 Scores AUC par modèle et plateforme")
                        auc_df = pd.DataFrame(list(comparisons["auc_scores"].items()),
                                           columns=["Model-Platform", "AUC"])
                        st.bar_chart(auc_df.set_index("Model-Platform"))
            
            # Graphiques détaillés
            st.subheader("Visualisations détaillées")
            try:
                plots_res = requests.get(f"{COMPARE_URL}/plots", headers=HEADERS)
                plots = plots_res.json()
                
                if "training_time" in plots:
                    st.image(f"data:image/png;base64,{plots['training_time']}", 
                            caption="Comparaison des temps d'entraînement")
                
                if "auc" in plots:
                    st.image(f"data:image/png;base64,{plots['auc']}", 
                            caption="Comparaison des scores AUC")
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
        st.experimental_rerun()

elif page == "Lancer l'entraînement Spark":
    st.header("6. Lancer l'entraînement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Lancer l'entraînement Spark (CPU)"):
            with st.spinner("Entraînement Spark en cours..."):
                try:
                    res = requests.get(f"{TRAIN_URL}/train?platform=spark", headers=HEADERS, timeout=600)
                    train_results = res.json()
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
                    st.success("Entraînement RAPIDS terminé !")
                    if "error" in train_results.get("rapids", {}):
                        st.warning(f"Erreur RAPIDS : {train_results['rapids']['error']}")
                    else:
                        st.json(train_results.get("rapids", {}))
                    try:
                        upload_res = requests.post(f"{COMPARE_URL}/upload-results", json=[train_results], headers=HEADERS)
                        st.info("Résultats envoyés à compare-service")
                    except Exception as e:
                        st.warning(f"Impossible d'envoyer les résultats : {e}")
                except Exception as e:
                    st.error(f"Erreur lors de l'entraînement RAPIDS : {e}")
