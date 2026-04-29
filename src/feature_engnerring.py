import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# Logs folder
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logger setup
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'feature_engineering.log'))

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load params or use default."""
    try:
        if not os.path.exists(params_path):
            logger.warning("params.yaml not found, using default value")
            return {"feature_engineering": {"max_features": 100}}

        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        logger.debug('Params loaded from %s', params_path)
        return params

    except Exception as e:
        logger.error('Error loading params: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise


def apply_tfidf(train_data, test_data, max_features):
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text']
        y_train = train_data['target']
        X_test = test_data['text']
        y_test = test_data['target']

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train.values

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test.values

        logger.debug("TF-IDF applied successfully")
        return train_df, test_df

    except Exception as e:
        logger.error('TF-IDF error: %s', e)
        raise


def save_data(df, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Saved to %s', file_path)
    except Exception as e:
        logger.error('Save error: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, './data/processed/train_tfidf.csv')
        save_data(test_df, './data/processed/test_tfidf.csv')

        logger.info("Feature engineering completed successfully!")

    except Exception as e:
        logger.error('Pipeline failed: %s', e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()