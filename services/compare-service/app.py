from fastapi import FastAPI
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import psutil

app = FastAPI()
RESULTS_PATH_SPARK = "/app/models/resultats_auc.csv"
RESULTS_PATH_RAPIDS = "/app/models/resultats_auc_rapids.csv"

@app.get("/status")
def status():
    return {"status": "compare-service running"}

@app.get("/results")
def get_results():
    results = []
    df_list = []
    
    # Lecture des résultats Spark
    if os.path.exists(RESULTS_PATH_SPARK):
        try:
            df_spark = pd.read_csv(RESULTS_PATH_SPARK)
            df_spark["platform"] = "Spark-CPU"
            df_list.append(df_spark)
            results.extend(df_spark.to_dict(orient="records"))
        except Exception as e:
            print("Erreur lecture Spark :", e)
    
    # Lecture des résultats RAPIDS
    if os.path.exists(RESULTS_PATH_RAPIDS):
        try:
            df_rapids = pd.read_csv(RESULTS_PATH_RAPIDS)
            df_rapids["platform"] = "RAPIDS-GPU"
            df_list.append(df_rapids)
            results.extend(df_rapids.to_dict(orient="records"))
        except Exception as e:
            print("Erreur lecture RAPIDS :", e)
    
    # Calcul des statistiques comparatives
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        comparisons = {
            "training_times": df.groupby("platform")["training_time"].mean().to_dict() if "training_time" in df.columns else {},
            "auc_scores": df.groupby(["model", "platform"])["auc"].mean().to_dict() if "auc" in df.columns else {},
            "platforms": list(df["platform"].unique()),
            "models": list(df["model"].unique())
        }
        results.append({"comparisons": comparisons})
    
    return results

@app.get("/plots")
def get_comparison_plots():
    df_list = []
    
    if os.path.exists(RESULTS_PATH_SPARK):
        df_spark = pd.read_csv(RESULTS_PATH_SPARK)
        df_spark["platform"] = "Spark-CPU"
        df_list.append(df_spark)
    
    if os.path.exists(RESULTS_PATH_RAPIDS):
        df_rapids = pd.read_csv(RESULTS_PATH_RAPIDS)
        df_rapids["platform"] = "RAPIDS-GPU"
        df_list.append(df_rapids)
    
    if not df_list:
        return {"error": "Aucun résultat à comparer"}
    
    df = pd.concat(df_list, ignore_index=True)
    plots = {}
    
    # Plot des temps d'entraînement
    if "training_time" in df.columns:
        plt.figure(figsize=(10, 6))
        train_times = df.groupby("platform")["training_time"].mean()
        plt.bar(train_times.index, train_times.values)
        plt.ylabel("Temps d'entraînement (s)")
        plt.title("Temps d'entraînement moyen par plateforme")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots["training_time"] = base64.b64encode(buf.getvalue()).decode()
        plt.close()
    
    # Plot des AUC
    if "auc" in df.columns:
        plt.figure(figsize=(12, 6))
        plt.bar(df["model"] + " (" + df["platform"] + ")", df["auc"])
        plt.ylabel("AUC")
        plt.xticks(rotation=45)
        plt.title("Comparaison des AUC")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots["auc"] = base64.b64encode(buf.getvalue()).decode()
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

