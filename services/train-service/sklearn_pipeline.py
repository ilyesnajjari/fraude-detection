import pandas as pd
import numpy as np
import os
import shutil
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    feature_cols = [c for c in df.columns if c not in ("Class",)]
    X = df[feature_cols]
    y = df["Class"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def run_all_models(X_train, X_test, y_train, y_test):
    results = {}

    # Logistic Regression
    start = time.time()
    lr = LogisticRegression(max_iter=5000, solver='lbfgs', random_state=42)
    lr.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]
    results["LogisticRegression"] = {
        "auc": roc_auc_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "training_time": train_time
    }

    # Random Forest
    start = time.time()
    rf = RandomForestClassifier(n_estimators=20, random_state=42)
    rf.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    results["RandomForest"] = {
        "auc": roc_auc_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "training_time": train_time
    }

    return results, lr, rf

def save_stats_and_model(df, scaler, lr_model, model_dir="../models/sklearn_logistic_model", stats_path="../models/summary_stats_sklearn.csv"):
    # S'assurer que le dossier existe
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    # Save statistics
    desc_pd = df.describe()
    desc_pd.to_csv(stats_path, index=False)
    # Save model (removing old one if exists)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(lr_model, os.path.join(model_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

def save_results(results, path="../models/resultats_auc_cpu.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    all_metrics = []
    for model, metrics in results.items():
        row = {"model": model}
        row.update(metrics)
        all_metrics.append(row)
    df_results = pd.DataFrame(all_metrics)
    df_results.to_csv(path, index=False)

# Exemple d'utilisation dans un script principal :
if __name__ == "__main__":
    DATA_PATH = "../data/creditcard.csv"
    df = load_data(DATA_PATH)
    X, y, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results, lr_model, rf_model = run_all_models(X_train, X_test, y_train, y_test)
    save_stats_and_model(df, scaler, lr_model)
    save_results(results)
    print("Résultats et modèles sauvegardés dans ../models/")