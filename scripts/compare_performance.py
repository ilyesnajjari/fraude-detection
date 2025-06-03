import matplotlib.pyplot as plt
import pandas as pd
import os

spark_csv = "../models/resultats_auc.csv"
rapids_csv = "../models/resultats_auc_rapids.csv"

# Lecture des résultats
df_list = []
if os.path.exists(spark_csv):
    df_spark = pd.read_csv(spark_csv)
    df_spark["platform"] = "Spark-CPU"
    df_list.append(df_spark)
if os.path.exists(rapids_csv):
    df_rapids = pd.read_csv(rapids_csv)
    df_rapids["platform"] = "RAPIDS-GPU"
    df_list.append(df_rapids)

if not df_list:
    print("Aucun résultat à comparer.")
    exit()

df = pd.concat(df_list, ignore_index=True)

# Comparaison des temps d'entraînement et de prédiction (si dispo)
if "training_time" in df.columns:
    train_times = df.groupby("platform")["training_time"].mean()
    labels = train_times.index.tolist()
    values = train_times.values.tolist()
    plt.bar(labels, values, width=0.3, label='Train Time')
    plt.ylabel("Temps d'entraînement (s)")
    plt.title("Temps d'entraînement moyen par plateforme")
    plt.legend()
    plt.show()

# Comparaison des AUC
if "auc" in df.columns:
    plt.bar(df["model"] + " (" + df["platform"] + ")", df["auc"])
    plt.ylabel("AUC")
    plt.xticks(rotation=45)
    plt.title("Comparaison des AUC")
    plt.tight_layout()
    plt.show()
