import os
import numpy as np
import pandas as pd
import pickle
import json
import logging
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from dvclive import Live

# ------------------ LOGGING ------------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'model_evaluation.log'))

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ------------------ LOAD PARAMS ------------------
def load_params(params_path: str):
    try:
        if not os.path.exists(params_path):
            logger.warning("params.yaml not found, using default values")
            return {"model_building": {"n_estimators": 100, "random_state": 42}}

        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        logger.debug("Params loaded successfully")
        return params

    except Exception as e:
        logger.error("Error loading params: %s", e)
        raise


# ------------------ LOAD MODEL ------------------
def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)

        logger.debug("Model loaded from %s", file_path)
        return model

    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise


# ------------------ LOAD DATA ------------------
def load_data(file_path: str):
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df

    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise


# ------------------ EVALUATION ------------------
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba)
        }

        logger.debug("Model evaluation completed")
        return metrics

    except Exception as e:
        logger.error("Evaluation error: %s", e)
        raise


# ------------------ SAVE METRICS ------------------
def save_metrics(metrics, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.debug("Metrics saved to %s", file_path)

    except Exception as e:
        logger.error("Error saving metrics: %s", e)
        raise


# ------------------ MAIN ------------------
def main():
    try:
        params = load_params('params.yaml')['model_building']

        model = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(model, X_test, y_test)

        # ---------------- DVCLive FIXED ----------------
        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy", metrics["accuracy"])
            live.log_metric("precision", metrics["precision"])
            live.log_metric("recall", metrics["recall"])
            live.log_metric("auc", metrics["auc"])

            live.log_params(params)

        # Save metrics locally
        save_metrics(metrics, 'reports/metrics.json')

        logger.info("Model evaluation completed successfully!")

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()