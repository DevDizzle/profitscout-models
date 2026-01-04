#!/usr/bin/env python3
"""
ProfitScout Batch Prediction Script
Loads LONG and SHORT models from GCS, generates technical features, and predicts High Gamma opportunities.
Outputs Top 50 CALLs and Top 50 PUTs.
"""
import argparse
import logging
import os
import tempfile
import urllib.parse
import pandas as pd
import pandas_ta_classic as ta
import xgboost as xgb
import numpy as np
import json
from google.cloud import bigquery, storage

logging.basicConfig(level=logging.INFO)

FEATURE_NAMES = [
    # Trend
    "sma_10", "sma_20", "sma_50", "sma_200",
    "ema_10", "ema_50",
    "adx_14", 
    "macd_12_26_9", "macdh_12_26_9", "macds_12_26_9",
    # Momentum
    "rsi_14", "rsi_5",
    "stochk_14_3_3", "stochd_14_3_3",
    "roc_1", "roc_3", "roc_5",
    # Volatility
    "atrr_14", "atr_ratio",
    "bbl_20_2.0", "bbu_20_2.0", "bbb_20_2.0", "bbp_20_2.0",
    # Volume
    "obv", "rvol_20", "volume_delta",
    # Structure/Ratios
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    "dist_from_high", "dist_from_low", "close_loc"
]

def load_threshold(model_path: str) -> float:
    """Download and load threshold from GCS path."""
    if not model_path.startswith("gs://"):
        return 0.5
        
    bucket_name = urllib.parse.urlparse(model_path).netloc
    prefix = urllib.parse.urlparse(model_path).path.lstrip("/")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(os.path.join(prefix, "threshold.json"))
    
    if not blob.exists():
        logging.warning("threshold.json not found at %s, defaulting to 0.5", model_path)
        return 0.5
    
    with tempfile.TemporaryDirectory() as tmp:
        local_path = os.path.join(tmp, "threshold.json")
        blob.download_to_filename(local_path)
        
        with open(local_path, "r") as f:
            data = json.load(f)
            return float(data.get("threshold", 0.5))

def load_model(model_path: str):
    """Download and load XGBoost model from GCS path."""
    if not model_path.startswith("gs://"):
        raise ValueError(f"model-path must start with gs://, got {model_path}")
        
    bucket_name = urllib.parse.urlparse(model_path).netloc
    prefix = urllib.parse.urlparse(model_path).path.lstrip("/")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(os.path.join(prefix, "model.json"))
    
    if not blob.exists():
        raise FileNotFoundError(f"model.json not found at {model_path}")
    
    with tempfile.TemporaryDirectory() as tmp:
        local_path = os.path.join(tmp, "model.json")
        blob.download_to_filename(local_path)
        
        bst = xgb.Booster()
        bst.load_model(local_path)
        return bst

def load_raw_price_data(project_id: str, source_table: str) -> pd.DataFrame:
    """Loads raw OHLCV data from BigQuery."""
    logging.info("Loading raw price data from %s.%s", project_id, source_table)
    bq = bigquery.Client(project=project_id)
    
    query = f"""
        SELECT ticker, date, open, high, low, adj_close as close, volume
        FROM `{project_id}.{source_table}`
        WHERE adj_close IS NOT NULL
        ORDER BY ticker, date
    """
    df = bq.query(query).to_dataframe()
    
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
    }, inplace=True)
    
    df["date"] = pd.to_datetime(df["date"])
    logging.info("Loaded %d rows", len(df))
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates technical indicators using pandas_ta."""
    logging.info("Engineering features...")
    
    results = []
    grouped = df.groupby("ticker")
    
    for ticker, group in grouped:
        if len(group) < 50:
            continue
            
        g = group.copy().set_index("date").sort_index()
        
        # --- Indicators (Must match training exactly) ---
        g.ta.sma(length=10, append=True)
        g.ta.sma(length=20, append=True)
        g.ta.sma(length=50, append=True)
        g.ta.sma(length=200, append=True)
        g.ta.ema(length=10, append=True)
        g.ta.ema(length=50, append=True)
        g.ta.adx(length=14, append=True)
        g.ta.macd(fast=12, slow=26, signal=9, append=True)
        g.ta.rsi(length=14, append=True)
        g.ta.rsi(length=5, append=True)
        g.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        g.ta.roc(length=1, append=True)
        g.ta.roc(length=3, append=True)
        g.ta.roc(length=5, append=True)

        g.ta.atr(length=14, append=True)
        g.ta.atr(length=3, append=True)
        if "ATRr_3" in g.columns and "ATRr_14" in g.columns:
             g["atr_ratio"] = g["ATRr_3"] / g["ATRr_14"]
        else:
             g["atr_ratio"] = 1.0

        g.ta.bbands(length=20, std=2, append=True)
        g.ta.obv(append=True)
        
        vol_sma = g["Volume"].rolling(window=20).mean()
        g["RVOL_20"] = g["Volume"] / vol_sma
        g["volume_delta"] = g["Volume"].pct_change()

        if "SMA_20" in g.columns:
            g["price_vs_sma20"] = g["Close"] / g["SMA_20"]
        if "SMA_50" in g.columns:
            g["price_vs_sma50"] = g["Close"] / g["SMA_50"]
        if "SMA_200" in g.columns:
            g["price_vs_sma200"] = g["Close"] / g["SMA_200"]
        
        g["dist_from_high"] = (g["High"] - g["Close"]) / g["Close"]
        g["dist_from_low"]  = (g["Close"] - g["Low"]) / g["Close"]
        
        denom = (g["High"] - g["Low"])
        g["close_loc"] = (g["Close"] - g["Low"]) / denom
        g.loc[denom == 0, "close_loc"] = 0.5

        g["ticker"] = ticker
        
        # Return all days where we have valid features
        results.append(g)

    if not results:
        return pd.DataFrame()
        
    full_df = pd.concat(results).reset_index()
    full_df.columns = [c.lower() for c in full_df.columns]
    
    # Cast features to float32 to ensure DMatrix compatibility
    for col in FEATURE_NAMES:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce').astype('float32')
        
    # Replace infs with nan
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Keep rows only where we have all features
    full_df.dropna(subset=FEATURE_NAMES, inplace=True)
    
    # Filter for the latest date per ticker for inference
    full_df = full_df.sort_values("date").groupby("ticker").tail(1)
    
    return full_df

def save_predictions(df: pd.DataFrame, project_id: str, destination_table: str):
    """Saves predictions to BigQuery with contract_type."""
    logging.info("Saving %d predictions to %s", len(df), destination_table)
    
    # Output columns
    out_df = df[["ticker", "date", "close", "prob", "contract_type", "prediction"]].copy()
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE", 
        schema=[
            bigquery.SchemaField("ticker", "STRING"),
            bigquery.SchemaField("date", "DATE"),
            bigquery.SchemaField("close", "FLOAT64"),
            bigquery.SchemaField("prob", "FLOAT64"),
            bigquery.SchemaField("contract_type", "STRING"),
            bigquery.SchemaField("prediction", "INTEGER"), # 1 if > threshold, 0 otherwise
        ],
    )
    
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{destination_table}" if destination_table.count(".") == 1 else destination_table
    
    client.load_table_from_dataframe(out_df, table_ref, job_config=job_config).result()
    logging.info("Save complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--source-table", required=True)
    parser.add_argument("--destination-table", required=True)
    parser.add_argument("--model-base-dir", required=True, help="Base GCS path for models (e.g. .../production/model)")
    
    # Compat flags
    parser.add_argument("--top-k-features", default="0")
    parser.add_argument("--auto-prune", default="false")
    
    args, _ = parser.parse_known_args()
    
    # 1. Load Data & Engineer Features
    raw_df = load_raw_price_data(args.project_id, args.source_table)
    feat_df = engineer_features(raw_df)
    
    if feat_df.empty:
        logging.warning("No valid features generated.")
        return
        
    all_predictions = []
    
    # 2. Iterate Directions
    for direction, subfolder, contract_type in [
        ("LONG", "long", "CALL"),
        ("SHORT", "short", "PUT")
    ]:
        model_path = os.path.join(args.model_base_dir, subfolder)
        logging.info("Processing %s direction from %s...", direction, model_path)
        
        try:
            bst = load_model(model_path)
            threshold = load_threshold(model_path)
            logging.info("Loaded %s model. Threshold: %f", direction, threshold)
            
            dmatrix = xgb.DMatrix(feat_df[FEATURE_NAMES])
            probs = bst.predict(dmatrix)
            
            # Create a view for this direction
            dir_df = feat_df.copy()
            dir_df["prob"] = probs
            dir_df["contract_type"] = contract_type
            
            # Mark strict sniper prediction
            dir_df["prediction"] = (probs > threshold).astype(int)
            
            # Take Top 10 by probability, regardless of threshold
            # (Users want to see the best available, even if confidence is lower than ideal)
            top_10_df = dir_df.sort_values("prob", ascending=False).head(10).copy()
            
            logging.info("Direction %s: Selected top %d tickers.", direction, len(top_10_df))
            all_predictions.append(top_10_df)
            
        except Exception as e:
            logging.error("Failed to process %s model: %s", direction, e)
            continue
            
    if not all_predictions:
        logging.warning("No predictions generated for either direction.")
        return

    # 3. Consolidate & Save
    final_df = pd.concat(all_predictions)
    
    # Sort by probability descending to show best setups at top (mixed Calls and Puts)
    final_df.sort_values("prob", ascending=False, inplace=True)
    
    save_predictions(final_df, args.project_id, args.destination_table)

if __name__ == "__main__":
    main()