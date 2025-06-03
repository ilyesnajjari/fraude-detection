from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler
import psutil
import pandas as pd

app = FastAPI()
spark = SparkSession.builder.appName("FraudDetectionPredict").getOrCreate()

@app.get("/status")
def status():
    return {"status": "predict-service running"}

MODEL_PATH = "./models/spark_logistic_model"
model = LogisticRegressionModel.load(MODEL_PATH)

# Classe Transaction avec toutes les features
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict/")
def predict(transaction: Transaction):
    # Liste complète des colonnes/features
    columns = [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
    ]
    data = [[getattr(transaction, col) for col in columns]]
    df = spark.createDataFrame(data, schema=columns)

    assembler = VectorAssembler(inputCols=columns, outputCol="features")
    df = assembler.transform(df)

    pred = model.transform(df).collect()[0]

    return {
        "prediction": int(pred.prediction),
        "probability": float(pred.probability[1])
    }

@app.get("/features")
def get_features():
    return {
        "features": [
            "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
            "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
            "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
        ]
    }

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

@app.get("/summary")
def get_summary():
    try:
        df = pd.read_csv("./models/summary_stats.csv")
        # On retourne tout le CSV sous forme de liste de dicts
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}
