import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# Logs directory
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logger setup
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'model_building.log'))

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str):
    """Load YAML params safely."""
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


def load_data(file_path: str):
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded: %s shape %s", file_path, df.shape)
        return df
    except Exception as e:
        logger.error("Data loading error: %s", e)
        raise


def train_model(X_train, y_train, params):
    try:
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            random_state=params['random_state']
        )

        clf.fit(X_train, y_train)
        logger.debug("Model trained successfully")

        return clf

    except Exception as e:
        logger.error("Training error: %s", e)
        raise


def save_model(model, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(model, f)

        logger.debug("Model saved at %s", file_path)

    except Exception as e:
        logger.error("Model saving error: %s", e)
        raise


def main():
    try:
        params = load_params('params.yaml')['model_building']

        train_data = load_data('./data/processed/train_tfidf.csv')

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        model = train_model(X_train, y_train, params)

        save_model(model, 'models/model.pkl')

        logger.info("Model training pipeline completed!")

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()