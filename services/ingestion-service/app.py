from fastapi import FastAPI, UploadFile, File
import pandas as pd
import os
import psutil

app = FastAPI()

@app.get("/status")
def status():
    return {"status": "ingestion-service running"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    os.makedirs("data", exist_ok=True)  # pour éviter l'erreur si data/ n'existe pas
    with open(f"data/{file.filename}", "wb") as f:
        f.write(contents)
    return {"filename": file.filename, "status": "uploaded"}


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
