import cudf
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
from cuml.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from cuml.model_selection import train_test_split
import pandas as pd
import time
import os

def run_rapids_models(data_path, results_path="../models/resultats_auc_rapids.csv"):
    print("📥 Chargement des données RAPIDS GPU depuis :", data_path)
    df = cudf.read_csv(data_path)

    features = [col for col in df.columns if col != "Class"]
    X = df[features]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # Logistic Regression RAPIDS
    start = time.time()
    lr = cuLogisticRegression()
    lr.fit(X_train, y_train)
    train_time_lr = time.time() - start

    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]

    results["RAPIDS LogisticRegression"] = {
        "auc": float(roc_auc_score(y_test, y_proba_lr)),
        "accuracy": float(accuracy_score(y_test, y_pred_lr)),
        "recall": float(recall_score(y_test, y_pred_lr)),
        "precision": float(precision_score(y_test, y_pred_lr)),
        "training_time": train_time_lr
    }

    # Random Forest RAPIDS
    start = time.time()
    rf = cuRandomForestClassifier(n_estimators=20)
    rf.fit(X_train, y_train)
    train_time_rf = time.time() - start

    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    results["RAPIDS RandomForest"] = {
        "auc": float(roc_auc_score(y_test, y_proba_rf)),
        "accuracy": float(accuracy_score(y_test, y_pred_rf)),
        "recall": float(recall_score(y_test, y_pred_rf)),
        "precision": float(precision_score(y_test, y_pred_rf)),
        "training_time": train_time_rf
    }

    # Sauvegarde des résultats dans un CSV compatible
    all_metrics = []
    for model, metrics in results.items():
        row = {"model": model}
        row.update(metrics)
        all_metrics.append(row)
    df_results = pd.DataFrame(all_metrics)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df_results.to_csv(results_path, index=False)
    print(f"✅ Résultats RAPIDS sauvegardés dans {results_path}")

    return results


# Si tu veux exécuter ce script en standalone :
if __name__ == "__main__":
    DATA_PATH = "../data/creditcard.csv"
    run_rapids_models(DATA_PATH)
