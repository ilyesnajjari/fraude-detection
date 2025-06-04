from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import time
import os
import shutil
import pandas as pd

def load_data(path):
    spark = SparkSession.builder.appName("FraudDetectionTrain").getOrCreate()
    df = spark.read.csv(path, header=True, inferSchema=True)
    return spark, df

def preprocess(df):
    feature_cols = [c for c in df.columns if c not in ("Class",)]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_prepared = assembler.transform(df).select("features", "Class")
    return df_prepared

def run_all_models(train, test):
    results = {}

    # Logistic Regression
    start = time.time()
    lr = LogisticRegression(featuresCol="features", labelCol="Class")
    lr_model = lr.fit(train)
    lr_pred = lr_model.transform(test)
    auc = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderROC").evaluate(lr_pred)
    acc = MulticlassClassificationEvaluator(labelCol="Class", metricName="accuracy").evaluate(lr_pred)
    recall = MulticlassClassificationEvaluator(labelCol="Class", metricName="recallByLabel").evaluate(lr_pred)
    precision = MulticlassClassificationEvaluator(labelCol="Class", metricName="precisionByLabel").evaluate(lr_pred)
    results["LogisticRegression"] = {
        "auc": auc,
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "training_time": time.time() - start
    }

    # Random Forest
    start = time.time()
    rf = RandomForestClassifier(featuresCol="features", labelCol="Class", numTrees=20)
    rf_model = rf.fit(train)
    rf_pred = rf_model.transform(test)
    auc = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderROC").evaluate(rf_pred)
    acc = MulticlassClassificationEvaluator(labelCol="Class", metricName="accuracy").evaluate(rf_pred)
    recall = MulticlassClassificationEvaluator(labelCol="Class", metricName="recallByLabel").evaluate(rf_pred)
    precision = MulticlassClassificationEvaluator(labelCol="Class", metricName="precisionByLabel").evaluate(rf_pred)
    results["RandomForest"] = {
        "auc": auc,
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "training_time": time.time() - start
    }

    # Pour sauvegarde ultérieure
    return results, lr_model

def save_stats_and_model(df, model, stats_path="../models/summary_stats.csv", model_path="../models/spark_logistic_model"):
    """Save dataset statistics and trained model"""
    desc_pd = df.describe().toPandas()
    desc_pd.to_csv(stats_path, index=False)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    model.save(model_path)

def save_results(results, path="../models/resultats_auc_spark.csv"):
    """Save model results to CSV"""
    all_metrics = []
    for model, metrics in results.items():
        row = {"model": model}
        row.update(metrics)
        all_metrics.append(row)
    df_results = pd.DataFrame(all_metrics)
    df_results.to_csv(path, index=False)

def save_preprocessed_data(predictions_df, processed_df, pred_path="./data/predictions_spark.csv", processed_path="./data/creditcard_spark_processed.csv"):
    """Save predictions and preprocessed features to CSV for microservices"""
    predictions_df.toPandas().to_csv(pred_path, index=False)
    processed_df.toPandas().to_csv(processed_path, index=False)

# Exemple d’utilisation
if __name__ == "__main__":
    DATA_PATH = "../data/creditcard.csv"
    spark, df = load_data(DATA_PATH)
    df_prepared = preprocess(df)
    train, test = df_prepared.randomSplit([0.8, 0.2], seed=42)

    results, lr_model, rf_model = run_all_models(train, test)

    save_stats_and_model(df, lr_model)
    save_results(results)
    save_preprocessed_data(pred_df, processed_df)

    spark.stop()
