import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from app import app

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd

client = TestClient(app)

class TestTrainModels(unittest.TestCase):
    @patch("app.skl_save_results")
    @patch("app.skl_save_stats_and_model")
    @patch("app.skl_run_all_models")
    @patch("app.skl_preprocess")
    @patch("app.skl_load_data")
    @patch("app.run_rapids_models")
    @patch("app.spark_save_results")
    @patch("app.run_all_models")
    @patch("app.preprocess")
    @patch("app.load_data")
    def test_train_all_platforms(
        self,
        mock_load_data,
        mock_preprocess,
        mock_run_all_models,
        mock_spark_save_results,
        mock_run_rapids_models,
        mock_skl_load_data,
        mock_skl_preprocess,
        mock_skl_run_all_models,
        mock_skl_save_stats_and_model,
        mock_skl_save_results
    ):
        # Mock Spark data loading and processing
        mock_spark = MagicMock()
        mock_df = MagicMock()
        mock_load_data.return_value = (mock_spark, mock_df)
        mock_preprocess.return_value = mock_df
        mock_df.randomSplit.return_value = ("train_set", "test_set")
        mock_run_all_models.return_value = ({"logistic_regression": {"auc": 0.95}}, MagicMock())
        mock_spark_save_results.return_value = None

        # Mock Spark DataFrame describe
        mock_df.describe.return_value.toPandas.return_value = \
            pd.DataFrame({"summary": ["count", "mean", "stddev", "min", "max"]})

        # Mock Rapids models
        mock_run_rapids_models.return_value = (
            {"logistic_regression": {"auc": 0.96}},
            MagicMock(),  # preds_df
            MagicMock()   # processed_df
        )
        # to_csv mocks for preds_df and processed_df
        mock_run_rapids_models.return_value[1].to_csv = MagicMock()
        mock_run_rapids_models.return_value[2].to_csv = MagicMock()

        # Mock sklearn data loading and processing
        mock_df_skl = MagicMock()
        mock_skl_load_data.return_value = mock_df_skl
        mock_df_skl.astype.return_value = mock_df_skl
        mock_skl_preprocess.return_value = ("X", "y", "scaler")
        mock_skl_run_all_models.return_value = (
            {"logistic_regression": {"auc": 0.97}},
            MagicMock(),  # lr_model
            MagicMock()   # rf_model
        )
        # Mock predict and predict_proba for sklearn models
        mock_skl_run_all_models.return_value[1].predict.return_value = [0, 1]
        mock_skl_run_all_models.return_value[1].predict_proba.return_value = [[0.3, 0.7], [0.6, 0.4]]
        mock_skl_run_all_models.return_value[2].predict.return_value = [1, 0]
        mock_skl_run_all_models.return_value[2].predict_proba.return_value = [[0.2, 0.8], [0.7, 0.3]]

        mock_skl_save_stats_and_model.return_value = None
        mock_skl_save_results.return_value = None

        response = client.get("/train")
        self.assertEqual(response.status_code, 200)

        json_response = response.json()

        self.assertIn("spark", json_response)
        self.assertIn("rapids", json_response)
        self.assertIn("sklearn", json_response)

    def test_status_endpoint(self):
        response = client.get("/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "train-service running"})

    def test_monitor_endpoint(self):
        response = client.get("/monitor")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("cpu", data)
        self.assertIn("ram", data)
        self.assertIn("gpu", data)


if __name__ == "__main__":
    unittest.main()
