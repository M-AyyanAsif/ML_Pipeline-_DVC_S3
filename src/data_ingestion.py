import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# Ensure logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'data_ingestion.log'))

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from YAML or use default."""
    try:
        if not os.path.exists(params_path):
            logger.warning("params.yaml not found, using default parameters")
            return {"data_ingestion": {"test_size": 0.2}}

        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        logger.debug('Parameters loaded from %s', params_path)
        return params

    except Exception as e:
        logger.error('Error loading params: %s', e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug('Preprocessing completed')
        return df
    except Exception as e:
        logger.error('Error in preprocessing: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, save_dir: str):
    try:
        raw_path = os.path.join(save_dir, 'raw')
        os.makedirs(raw_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_path, 'test.csv'), index=False)

        logger.debug('Data saved to %s', raw_path)
    except Exception as e:
        logger.error('Error saving data: %s', e)
        raise


def main():
    try:
        # Load params (safe)
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']

        # Load data
        url = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        df = load_data(url)

        # Preprocess
        df = preprocess_data(df)

        # Split
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

        # Save
        save_data(train_data, test_data, 'data')

        logger.info("Data ingestion completed successfully!")

    except Exception as e:
        logger.error('Pipeline failed: %s', e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()