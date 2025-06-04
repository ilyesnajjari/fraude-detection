from fastapi import FastAPI
from spark_pipeline import load_data, preprocess, run_all_models, save_results as spark_save_results
from rapids_pipeline import run_rapids_models
from sklearn_pipeline import load_data as skl_load_data, preprocess as skl_preprocess, run_all_models as skl_run_all_models, save_results as skl_save_results, save_stats_and_model as skl_save_stats_and_model
import psutil
import os
import pandas as pd

app = FastAPI()
DATA_PATH = "./data/creditcard.csv"

@app.get("/status")
def status():
    return {"status": "train-service running"}

@app.get("/train")
def train_models(platform: str = None):
    results = {}

    if platform in [None, "spark"]:
        try:
            spark, df = load_data(DATA_PATH)
            df_prepared = preprocess(df)
            train, test = df_prepared.randomSplit([0.8, 0.2], seed=42)

            results_spark, lr_model = run_all_models(train, test)
            spark_save_results(results_spark)

            # ➕ Sauvegarde des résultats spark dans un CSV
            spark_metrics_df = pd.DataFrame([
                {"model": model, **metrics} for model, metrics in results_spark.items()
            ])
            spark_metrics_df.to_csv("./models/resultats_auc_spark.csv", index=False)

            # Sauvegarde des prédictions (Logistic Regression uniquement ici)
            y_test = test.select("Class").toPandas()
            y_pred = lr_model.transform(test).select("prediction").toPandas()
            y_proba = lr_model.transform(test).select("probability").toPandas()["probability"].apply(lambda x: x[1])
            df_preds = pd.DataFrame({
                "actual": y_test["Class"],
                "pred_lr": y_pred["prediction"],
                "proba_lr": y_proba
            })
            df_preds.to_csv("./data/predictions_spark.csv", index=False)

            # Sauvegarde des features prétraitées
            df_prepared.toPandas().to_csv("./data/creditcard_spark_processed.csv", index=False)

            # ➕ Sauvegarde des summary stats
            summary_df = df.describe().toPandas().set_index("summary")
            column_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
            summary_df = summary_df.loc[["count", "mean", "stddev", "min", "max"]][column_order]
            summary_df.to_csv("./models/summary_stats_spark.csv")

            spark.stop()
            results["spark"] = results_spark
        except Exception as e:
            results["spark"] = {"error": str(e)}



    if platform in [None, "rapids"]:
        try:
            results_rapids, preds_df, processed_df = run_rapids_models(DATA_PATH)

            # ➕ Sauvegarde des résultats rapids dans un CSV
            rapids_metrics_df = pd.DataFrame([
                {"model": model, **metrics} for model, metrics in results_rapids.items()
            ])
            rapids_metrics_df.to_csv("./models/resultats_auc_rapids.csv", index=False)

            # Sauvegarde des prédictions
            preds_df.to_csv("./data/predictions_rapids.csv", index=False)

            # Sauvegarde des features prétraitées
            processed_df.to_csv("./data/creditcard_rapids_processed.csv", index=False)

            # ➕ Sauvegarde des summary stats
            import cudf
            df_cu = cudf.read_csv(DATA_PATH)
            summary_df = df_cu.describe().to_pandas()
            summary_df = summary_df.set_index("summary")
            column_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
            summary_df = summary_df.loc[["count", "mean", "std", "min", "max"]].rename(index={"std": "stddev"})[column_order]
            summary_df.to_csv("./models/summary_stats_rapids.csv")

            results["rapids"] = results_rapids
        except Exception as e:
            results["rapids"] = {"error": str(e)}


    if platform in [None, "sklearn"]:
        try:
            os.makedirs("./models", exist_ok=True)
            df = skl_load_data(DATA_PATH)
            df = df.astype(float)
            X, y, scaler = skl_preprocess(df)

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            results_sklearn, lr_model, rf_model = skl_run_all_models(X_train, X_test, y_train, y_test)

            skl_save_stats_and_model(df, scaler, lr_model)
            skl_save_results(results_sklearn)

            # ➕ Sauvegarde des résultats sklearn dans un CSV
            sklearn_metrics_df = pd.DataFrame([
                {"model": model, **metrics} for model, metrics in results_sklearn.items()
            ])
            sklearn_metrics_df.to_csv("./models/resultats_auc_cpu.csv", index=False)

            # Sauvegarde des prédictions
            df_preds = pd.DataFrame({
                "actual": y_test,
                "pred_lr": lr_model.predict(X_test),
                "proba_lr": lr_model.predict_proba(X_test)[:, 1],
                "pred_rf": rf_model.predict(X_test),
                "proba_rf": rf_model.predict_proba(X_test)[:, 1]
            })
            df_preds.to_csv("./data/predictions_sklearn.csv", index=False)

            # Sauvegarde des features prétraitées
            X_scaled_df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, X.shape[1] + 1)])
            X_scaled_df["Class"] = y.values
            X_scaled_df.to_csv("./data/creditcard_sklearn_processed.csv", index=False)

            # ➕ Sauvegarde des summary stats
            summary_df = df.describe().T[["count", "mean", "std", "min", "max"]].rename(columns={"std": "stddev"}).T
            summary_df.index.name = "summary"
            column_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
            summary_df = summary_df[column_order]
            summary_df.to_csv("./models/summary_stats_sklearn.csv")

            results["sklearn"] = results_sklearn
        except Exception as e:
            results["sklearn"] = {"error": str(e)}


    return results

@app.get("/monitor")
def monitor():
    cpu_percent = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().used / (1024 * 1024)

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
