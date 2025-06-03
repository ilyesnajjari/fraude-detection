import cudf
from cuml.linear_model import LogisticRegression
from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from cuml.model_selection import train_test_split
import pandas as pd

def run_rapids_models(data_path, results_path="../models/resultats_auc_rapids.csv"):
    df = cudf.read_csv(data_path)
    features = [col for col in df.columns if col not in ("Class",)]
    X = df[features]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]
    results["RAPIDS LogisticRegression"] = {
        "auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=20)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    results["RAPIDS RandomForest"] = {
        "auc": float(roc_auc_score(y_test, y_proba_rf)),
        "accuracy": float(accuracy_score(y_test, y_pred_rf)),
        "recall": float(recall_score(y_test, y_pred_rf)),
        "precision": float(precision_score(y_test, y_pred_rf)),
    }

    # Save results
    all_metrics = []
    for model, metrics in results.items():
        row = {"model": model}
        row.update(metrics)
        all_metrics.append(row)
    df_results = pd.DataFrame(all_metrics)
    df_results.to_csv(results_path, index=False)
    return results