from fastapi import FastAPI
from spark_pipeline import load_data, preprocess, run_all_models, save_stats_and_model, save_results
from rapids_pipeline import run_rapids_models
import psutil

app = FastAPI()

DATA_PATH = "./data/creditcard.csv"

@app.get("/status")
def status():
    return {"status": "train-service running"}

@app.get("/train")
def train_models(platform: str = None):
    results = {}
    
    if platform in [None, "spark"]:
        # Spark training
        spark, df = load_data(DATA_PATH)
        df_prepared = preprocess(df)
        train, test = df_prepared.randomSplit([0.8, 0.2], seed=42)
        results_spark, lr_model = run_all_models(train, test)
        save_stats_and_model(df, lr_model)
        save_results(results_spark)
        spark.stop()
        results["spark"] = results_spark

    if platform in [None, "rapids"]:
        # RAPIDS training (GPU)
        try:
            results_rapids = run_rapids_models(DATA_PATH)
            results["rapids"] = results_rapids
        except Exception as e:
            results["rapids"] = {"error": str(e)}

    return results

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

    return {"cpu": cpu_percent, "ram": ram_usage, "gpu": gpu_percent}
