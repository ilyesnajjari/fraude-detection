from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import psutil

app = FastAPI()

RESULTS_PATH_SPARK = "/app/models/resultats_auc_spark.csv"
RESULTS_PATH_RAPIDS = "/app/models/resultats_auc_rapids.csv"
RESULTS_PATH_SKLEARN = "/app/models/resultats_auc_cpu.csv"

@app.get("/status")
def status():
    return {"status": "compare-service running"}

@app.get("/results")
def get_results():
    results = []
    df_list = []
    try:
        # Lecture des résultats Spark
        if os.path.exists(RESULTS_PATH_SPARK):
            df_spark = pd.read_csv(RESULTS_PATH_SPARK)
            df_spark["platform"] = "Spark-CPU"
            df_list.append(df_spark)
            results.extend(df_spark.to_dict(orient="records"))

        # Lecture des résultats RAPIDS
        if os.path.exists(RESULTS_PATH_RAPIDS):
            df_rapids = pd.read_csv(RESULTS_PATH_RAPIDS)
            df_rapids["platform"] = "RAPIDS-GPU"
            df_list.append(df_rapids)
            results.extend(df_rapids.to_dict(orient="records"))

        # Lecture des résultats SKLEARN (CPU)
        if os.path.exists(RESULTS_PATH_SKLEARN):
            df_sklearn = pd.read_csv(RESULTS_PATH_SKLEARN)
            df_sklearn["platform"] = "Sklearn-CPU"
            df_list.append(df_sklearn)
            results.extend(df_sklearn.to_dict(orient="records"))

        # Calcul des statistiques comparatives
        if df_list:
            df = pd.concat(df_list, ignore_index=True)

            # Nettoyage : s'assurer que 'model' et 'platform' sont bien des chaînes de caractères
            df["model"] = df["model"].apply(lambda x: str(x))
            df["platform"] = df["platform"].apply(lambda x: str(x))

            # Après avoir chargé le DataFrame df
            if "training_time" in df.columns and "training_time_scores" not in df.columns:
                df = df.rename(columns={"training_time": "training_time_scores"})

            comparisons = {
                "platforms": list(df["platform"].unique()),
                "models": list(df["model"].unique()),
            }

            numeric_metrics = ["auc", "accuracy", "recall", "precision", "training_time_scores"]
            for metric in numeric_metrics:
                if metric in df.columns:
                    metric_means = df.groupby(["model", "platform"])[metric].mean().to_dict()
                    formatted_means = {
                        f"{model}_{platform}": float(value)
                        for (model, platform), value in metric_means.items()
                    }
                    comparisons[f"{metric}_scores"] = formatted_means

            results.append({"comparisons": comparisons})
            return results
        else:
            return JSONResponse(content={"error": "Aucun résultat trouvé."}, status_code=404)
    except Exception as e:
        print("Erreur interne dans /results :", e)
        return JSONResponse(content={"error": f"Erreur interne : {str(e)}"}, status_code=500)

@app.get("/plots")
def get_comparison_plots():
    import seaborn as sns

    df_list = []

    if os.path.exists(RESULTS_PATH_SPARK):
        df_spark = pd.read_csv(RESULTS_PATH_SPARK)
        df_spark["platform"] = "Spark-CPU"
        df_list.append(df_spark)

    if os.path.exists(RESULTS_PATH_RAPIDS):
        df_rapids = pd.read_csv(RESULTS_PATH_RAPIDS)
        df_rapids["platform"] = "RAPIDS-GPU"
        df_list.append(df_rapids)

    if os.path.exists(RESULTS_PATH_SKLEARN):
        df_sklearn = pd.read_csv(RESULTS_PATH_SKLEARN)
        df_sklearn["platform"] = "Sklearn-CPU"
        df_list.append(df_sklearn)

    if not df_list:
        return {"error": "Aucun résultat à comparer"}

    df = pd.concat(df_list, ignore_index=True)
    df["model"] = df["model"].apply(lambda x: str(x))
    df["platform"] = df["platform"].apply(lambda x: str(x))

    # Après avoir chargé le DataFrame df
    if "training_time" in df.columns and "training_time_scores" not in df.columns:
        df = df.rename(columns={"training_time": "training_time_scores"})

    plots = {}
    sns.set_theme(style="whitegrid")

    # Juste avant les plots, tu peux forcer l'ordre et la lisibilité :
    platform_order = ["Spark-CPU", "Sklearn-CPU", "RAPIDS-GPU"]
    palette = {"Spark-CPU": "#4C72B0", "Sklearn-CPU": "#55A868", "RAPIDS-GPU": "#C44E52"}

    # Barplot des temps d'entraînement
    if "training_time_scores" in df.columns:
        plt.figure(figsize=(8, 5))
        model_palette = {"LogisticRegression": "#4C72B0", "RandomForest": "#55A868"}
        ax = sns.barplot(
            data=df,
            x="platform",
            y="training_time_scores",
            hue="model",
            order=platform_order,
            palette=model_palette
        )
        plt.ylabel("Temps d'entraînement (s)")
        plt.title("Temps d'entraînement moyen par plateforme et modèle")
        plt.legend(title="Modèle")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots["training_time_scores"] = base64.b64encode(buf.getvalue()).decode()
        plt.close()

    # Barplot des scores AUC
    if "auc" in df.columns:
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(
            data=df,
            x="model",
            y="auc",
            hue="platform",
            palette="Set1"
        )
        plt.ylabel("AUC")
        plt.title("Score AUC par modèle et plateforme")
        plt.legend(title="Plateforme")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots["auc"] = base64.b64encode(buf.getvalue()).decode()
        plt.close()

        # Boxplot des scores AUC par plateforme
        plt.figure(figsize=(8, 5))
        ax = sns.boxplot(
            data=df,
            x="platform",
            y="auc",
            palette="pastel"
        )
        plt.ylabel("AUC")
        plt.title("Distribution des scores AUC par plateforme")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots["auc_boxplot"] = base64.b64encode(buf.getvalue()).decode()
        plt.close()

    # Barplot de la précision
    if "precision" in df.columns:
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(
            data=df,
            x="model",
            y="precision",
            hue="platform",
            palette="Set3"
        )
        plt.ylabel("Précision")
        plt.title("Précision par modèle et plateforme")
        plt.legend(title="Plateforme")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots["precision"] = base64.b64encode(buf.getvalue()).decode()
        plt.close()

    # Pie chart (camembert) moderne de la répartition du temps d'entraînement par plateforme
    if "training_time_scores" in df.columns:
        train_times = df.groupby("platform")["training_time_scores"].mean() if "platform" in df.columns else df.groupby("model")["training_time_scores"].mean()
        plt.figure(figsize=(7, 7))
        wedges, texts, autotexts = plt.pie(
            train_times,
            labels=train_times.index,
            autopct='%1.1f%%',
            colors=[palette.get(p, "#cccccc") for p in train_times.index],
            startangle=140,
            wedgeprops=dict(width=0.4, edgecolor='w'),
            pctdistance=0.85
        )
        # Cercle central pour un effet donut moderne
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        plt.gca().add_artist(centre_circle)
        plt.title("Répartition du temps d'entraînement moyen par plateforme", fontsize=14)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots["training_time_pie"] = base64.b64encode(buf.getvalue()).decode()
        plt.close()

    return plots

@app.get("/monitor")
def monitor():
    cpu_percent = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().used / (1024 * 1024)  # en MB

    # Ajout monitoring RAPIDS/GPU si disponible
    gpu_percent = None
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_percent = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        pynvml.nvmlShutdown()
    except Exception:
        gpu_percent = None

    return {
        "cpu": cpu_percent,
        "ram": ram_usage,
        "gpu": gpu_percent
    }