#!/usr/bin/env python3
"""
ProfitScout Batch Prediction Script
===================================
This script loads a trained model and its associated preprocessing
artifacts to run batch predictions on new data.
"""
import argparse
import json
import logging
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from google.cloud import bigquery, storage

# --- Configuration ---
logging.basicConfig(level=logging.INFO)

# --- Feature Definitions (must match trainer) ---
EMB_COLS = [
    "key_financial_metrics_embedding", "key_discussion_points_embedding",
    "sentiment_tone_embedding", "short_term_outlook_embedding",
    "forward-looking_signals_embedding"
]
STATIC_NUM_COLS = [
    "sentiment_score", "sma_20", "ema_50", "rsi_14", "adx_14", "sma_20_delta",
    "ema_50_delta", "rsi_14_delta", "adx_14_delta", "eps_surprise"
]
ENGINEERED_COLS = [
    "price_sma20_ratio", "ema50_sma20_ratio", "rsi_centered", "adx_log1p",
    "sent_rsi", "eps_surprise_isnull", "cos_fin_disc", "cos_fin_tone",
    "cos_disc_short", "cos_short_fwd"
]


# --- Modular Functions ---

def load_artifacts(model_dir: str) -> tuple:
    """Loads all model and preprocessing artifacts from GCS."""
    if not model_dir.startswith("gs://"):
        raise ValueError(f"model_dir must be a GCS path. Got: {model_dir}")

    logging.info(f"Loading artifacts from {model_dir}")
    storage_client = storage.Client()
    bucket_name, blob_prefix = model_dir.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)

    # Load XGBoost model
    local_xgb_path = "xgb_model.json"
    bucket.blob(os.path.join(blob_prefix, "xgb_model.json")).download_to_filename(local_xgb_path)
    bst = xgb.Booster()
    bst.load_model(local_xgb_path)
    logging.info("XGBoost model loaded.")

    # Load all other artifacts
    local_artifacts_path = "training_artifacts.joblib"
    bucket.blob(os.path.join(blob_prefix, "training_artifacts.joblib")).download_to_filename(local_artifacts_path)
    artifacts = joblib.load(local_artifacts_path)
    logging.info("Preprocessing artifacts loaded.")
    
    return bst, artifacts

def load_data(project_id: str, source_table: str) -> pd.DataFrame:
    """Loads prediction data from a BigQuery table."""
    logging.info(f"Loading prediction data from {project_id}.{source_table}")
    bq_client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{project_id}.{source_table}`"
    df = bq_client.query(query).to_dataframe()
    logging.info(f"Loaded {len(df):,} rows for prediction.")
    return df

def preprocess_for_inference(df: pd.DataFrame, artifacts: dict) -> np.ndarray:
    """Applies transformations to new data using loaded artifacts."""
    logging.info("Applying inference transformations...")

    # Type correction
    for c in EMB_COLS:
        if c in df.columns and df[c].dtype == 'object' and df[c].notna().any():
             df[c] = df[c].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) and x.startswith('[') else x)

    # Apply Winsorizing and Imputation using maps from artifacts
    for col, (lo, hi) in artifacts["winsor_map"].items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    
    df["eps_surprise"] = df["eps_surprise"].fillna(df["industry_sector"].map(artifacts["sector_mean_map"])).fillna(0.0)

    # Apply PCA transformation using loaded PCA models
    X_pca_list = []
    for col in EMB_COLS:
        if col not in df.columns: continue
        
        valid_embeddings = df[col].dropna()
        if valid_embeddings.empty: 
            # If no valid embeddings, create a placeholder of NaNs
            X_pca_list.append(np.full((len(df), artifacts["pca_map"][col].n_components_), np.nan))
            continue
        
        A = np.vstack(valid_embeddings.values)
        pca = artifacts["pca_map"][col]
        
        transformed_col = np.full((len(df), pca.n_components_), np.nan)
        transformed_col[valid_embeddings.index] = pca.transform(A)
        X_pca_list.append(transformed_col)

    X_pca = np.hstack(X_pca_list)

    # Assemble feature matrix
    for c in EMB_COLS:
        base = c.replace("_embedding", "")
        ENGINEERED_COLS.extend([f"{base}_norm", f"{base}_mean", f"{base}_std"])

    final_num_cols = [c for c in (STATIC_NUM_COLS + ENGINEERED_COLS) if c in df.columns]
    X_num = df[final_num_cols].values
    
    feature_names = X_pca.shape[1] + len(final_num_cols)
    X_all = np.hstack([X_pca, X_num]).astype(np.float32)

    # Apply Imputer and Scaler
    X_all_imputed = artifacts["imputer"].transform(X_all)
    X_all_scaled = artifacts["scaler"].transform(X_all_imputed)
    
    return X_all_imputed, X_all_scaled, feature_names

def make_predictions(bst, artifacts: dict, X_imputed: np.ndarray, X_scaled: np.ndarray, feature_names: list) -> pd.Series:
    """Generates final predictions using the model blend."""
    logging.info("Generating predictions...")
    
    # Predict with both models
    dpred = xgb.DMatrix(X_imputed, feature_names=feature_names)
    xgb_raw = bst.predict(dpred)
    log_raw = artifacts["logreg"].predict_proba(X_scaled)[:, 1]
    
    # Blend, Calibrate, and Apply Threshold
    blend_raw = 0.5 * xgb_raw + 0.5 * log_raw
    blend_calibrated = artifacts["calibrator"].transform(blend_raw)
    
    predictions = (blend_calibrated >= artifacts["threshold"]).astype(int)
    
    # Return calibrated probability and final prediction
    return pd.DataFrame({
        "calibrated_probability": blend_calibrated,
        "prediction": predictions
    })

def save_predictions(df: pd.DataFrame, project_id: str, destination_table: str):
    """Saves the predictions back to a BigQuery table."""
    logging.info(f"Saving {len(df):,} predictions to {project_id}.{destination_table}")
    bq_client = bigquery.Client(project=project_id)
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE", # Overwrite table with new predictions
        schema=[
            bigquery.SchemaField("ticker", "STRING"),
            bigquery.SchemaField("quarter_end_date", "DATE"),
            bigquery.SchemaField("earnings_call_date", "DATE"),
            bigquery.SchemaField("calibrated_probability", "FLOAT64"),
            bigquery.SchemaField("prediction", "INTEGER"),
        ]
    )
    
    job = bq_client.load_table_from_dataframe(df, destination_table, job_config=job_config)
    job.result() # Wait for the job to complete
    logging.info("✅ Predictions saved successfully.")


def main():
    """Main orchestrator function for batch prediction."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', type=str, required=True)
    parser.add_argument('--source-table', type=str, required=True, help="Table with data to predict on.")
    parser.add_argument('--destination-table', type=str, required=True, help="Table to write predictions to.")
    parser.add_argument('--model-dir', type=str, required=True, help="GCS directory of saved model artifacts.")
    args = parser.parse_args()

    # Execute Pipeline
    bst, artifacts = load_artifacts(args.model_dir)
    df_new_data = load_data(args.project_id, args.source_table)
    
    if df_new_data.empty:
        logging.info("No data to predict on. Exiting.")
        return

    X_imp, X_scl, f_names = preprocess_for_inference(df_new_data, artifacts)
    df_predictions = make_predictions(bst, artifacts, X_imp, X_scl, f_names)

    # Combine original identifiers with predictions for saving
    df_results = pd.concat([
        df_new_data[['ticker', 'quarter_end_date', 'earnings_call_date']].reset_index(drop=True),
        df_predictions
    ], axis=1)

    save_predictions(df_results, args.project_id, args.destination_table)


if __name__ == "__main__":
    main()