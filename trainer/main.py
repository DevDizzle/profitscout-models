import os
import logging
import argparse
import pandas as pd
from joblib import dump
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from google.cloud import bigquery, storage

logging.basicConfig(level=logging.INFO)

def main(args):
    """Main training script."""
    bq_client = bigquery.Client(project=args.project_id)

    # 1. Load Data from BigQuery
    logging.info(f"Loading data from table: {args.source_table}")
    # This query joins features with metadata and calculates the target variable
    query = f"""
    WITH features AS (
        SELECT * FROM `{args.source_table}`
    ),
    metadata AS (
        SELECT symbol as ticker, sector, industry FROM `{args.metadata_table}`
    )
    SELECT
        f.*,
        m.sector,
        m.industry,
        -- Calculate the target variable: 1 if max_close_30d is > 14% gain, else 0
        IF(f.max_close_30d > (f.adj_close_on_call_date * 1.14), 1, 0) as target
    FROM features f
    LEFT JOIN metadata m ON f.ticker = m.ticker
    WHERE f.adj_close_on_call_date IS NOT NULL AND f.max_close_30d IS NOT NULL
    """
    df = bq_client.query(query).to_dataframe()
    logging.info(f"Loaded {len(df)} rows.")

    # 2. Feature and Target Preparation
    # Drop non-feature columns
    features_to_drop = [
        'ticker', 'quarter_end_date', 'earnings_call_date',
        'adj_close_on_call_date', 'max_close_30d', 'target'
    ]
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['sector', 'industry'], dummy_na=True)
    
    X = df.drop(columns=features_to_drop)
    y = df['target']

    # 3. Train Model
    logging.info("Splitting data and training XGBoost model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    logging.info(f"Model accuracy on test set: {score:.4f}")

    # 4. Save Model to GCS
    # Vertex AI sets this environment variable to a GCS path for you
    model_directory = os.environ.get('AIP_MODEL_DIR', f'gs://{args.model_bucket}/models/')
    model_filename = 'model.joblib'
    gcs_path = os.path.join(model_directory, model_filename)
    
    storage_client = storage.Client()
    bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    logging.info(f"Saving model to {gcs_path}")
    dump(model, model_filename)
    blob.upload_from_filename(model_filename)
    logging.info("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', type=str, required=True, help='Google Cloud project ID')
    parser.add_argument('--source-table', type=str, required=True, help='BigQuery table with features')
    parser.add_argument('--metadata-table', type=str, required=True, help='BigQuery table with stock metadata')
    parser.add_argument('--model-bucket', type=str, required=True, help='GCS bucket to save model to')
    
    known_args, _ = parser.parse_known_args()
    main(known_args)