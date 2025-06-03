from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import time

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

    # XGBoost possible (nécessite pyspark-xgboost et configuration JVM)
    # from sparkxgb import XGBoostClassifier
    # start = time.time()
    # xgb = XGBoostClassifier(featuresCol="features", labelCol="Class")
    # xgb_model = xgb.fit(train)
    # xgb_pred = xgb_model.transform(test)
    # auc = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderROC").evaluate(xgb_pred)
    # acc = MulticlassClassificationEvaluator(labelCol="Class", metricName="accuracy").evaluate(xgb_pred)
    # recall = MulticlassClassificationEvaluator(labelCol="Class", metricName="recallByLabel").evaluate(xgb_pred)
    # precision = MulticlassClassificationEvaluator(labelCol="Class", metricName="precisionByLabel").evaluate(xgb_pred)
    # results["XGBoost"] = {
    #     "auc": auc,
    #     "accuracy": acc,
    #     "recall": recall,
    #     "precision": precision,
    #     "training_time": time.time() - start
    # }

    return results