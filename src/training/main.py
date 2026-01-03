#!/usr/bin/env python3
"""
ProfitScout Training Script - High Gamma Options Strategy
Target: Next Day Price Move > 0.5 * ATR(14)
"""
import argparse
import logging
import os
import tempfile
import urllib.parse
import gc
import sys
import hypertune
import pandas as pd
import pandas_ta_classic as ta
import xgboost as xgb
from google.cloud import aiplatform, bigquery, storage
from sklearn.metrics import (
    brier_score_loss,
    average_precision_score
)

logging.basicConfig(level=logging.INFO)

# ───────────────────────────── Config ─────────────────────────────

FEATURE_NAMES = [
    # Trend
    "sma_10", "sma_20", "sma_50", "sma_200",
    "ema_10", "ema_50",
    "adx_14", 
    "macd_12_26_9", "macdh_12_26_9", "macds_12_26_9",
    # Momentum
    "rsi_14", "rsi_5",  # Added RSI 5
    "stochk_14_3_3", "stochd_14_3_3",
    "roc_1", "roc_3", "roc_5", # Added ROC
    # Volatility
    "atrr_14", "atr_ratio", # Added ATR Ratio
    "bbl_20_2.0", "bbu_20_2.0", "bbb_20_2.0", "bbp_20_2.0",
    # Volume
    "obv", "rvol_20", "volume_delta", # Added Volume Delta
    # Structure/Ratios
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    "dist_from_high", "dist_from_low", "close_loc" # Added Close Location
]

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
    sys.stdout.flush()
    
    results = []
    grouped = df.groupby("ticker")
    processed_count = 0
    total_tickers = len(grouped)
    
    for ticker, group in grouped:
        if len(group) < 200:
            continue
            
        g = group.copy().set_index("date").sort_index()
        
        # --- Trend ---
        g.ta.sma(length=10, append=True)
        g.ta.sma(length=20, append=True)
        g.ta.sma(length=50, append=True)
        g.ta.sma(length=200, append=True)
        g.ta.ema(length=10, append=True)
        g.ta.ema(length=50, append=True)
        g.ta.adx(length=14, append=True)
        g.ta.macd(fast=12, slow=26, signal=9, append=True)

        # --- Momentum ---
        g.ta.rsi(length=14, append=True)
        g.ta.rsi(length=5, append=True) # Fast RSI
        g.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        g.ta.roc(length=1, append=True) # 1-day change
        g.ta.roc(length=3, append=True) # 3-day change
        g.ta.roc(length=5, append=True) # 5-day change

        # --- Volatility ---
        g.ta.atr(length=14, append=True)
        g.ta.atr(length=3, append=True) # Fast ATR
        # ATR Ratio: Fast / Slow (Rising volatility check)
        # pandas_ta names them ATRr_14 and ATRr_3 usually
        if "ATRr_3" in g.columns and "ATRr_14" in g.columns:
             g["atr_ratio"] = g["ATRr_3"] / g["ATRr_14"]
        else:
             g["atr_ratio"] = 1.0

        g.ta.bbands(length=20, std=2, append=True)

        # --- Volume ---
        g.ta.obv(append=True)
        # RVOL
        vol_sma = g["Volume"].rolling(window=20).mean()
        g["RVOL_20"] = g["Volume"] / vol_sma
        # Volume Delta
        g["volume_delta"] = g["Volume"].pct_change()

        # --- Ratios & Structure ---
        if "SMA_20" in g.columns:
            g["price_vs_sma20"] = g["Close"] / g["SMA_20"]
        if "SMA_50" in g.columns:
            g["price_vs_sma50"] = g["Close"] / g["SMA_50"]
        if "SMA_200" in g.columns:
            g["price_vs_sma200"] = g["Close"] / g["SMA_200"]
        
        g["dist_from_high"] = (g["High"] - g["Close"]) / g["Close"]
        g["dist_from_low"]  = (g["Close"] - g["Low"]) / g["Close"]
        
        # Close Location Value (CLV) - Where in the High-Low range did we close?
        # ((C - L) - (H - C)) / (H - L) -> Ranges from -1 (Low) to 1 (High)
        # Or simpler: (C - L) / (H - L) -> 0 to 1
        denom = (g["High"] - g["Low"])
        g["close_loc"] = (g["Close"] - g["Low"]) / denom
        g.loc[denom == 0, "close_loc"] = 0.5 # Handle flat days

        # --- Target Generation: High Gamma ---
        g["next_close"] = g["Close"].shift(-1)
        atr_col = "ATRr_14" if "ATRr_14" in g.columns else "ATR_14"
        
        if atr_col in g.columns:
            threshold = 0.5 * g[atr_col]
            g["target"] = ((g["next_close"] - g["Close"]) > threshold).astype(int)
        else:
            g["target"] = 0
            
        g["ticker"] = ticker
        results.append(g)
        
        processed_count += 1
        if processed_count % 100 == 0:
            logging.info("Processed %d/%d tickers", processed_count, total_tickers)
            sys.stdout.flush()

    if not results:
        return pd.DataFrame()
        
    full_df = pd.concat(results).reset_index()
    del results
    gc.collect()
    
    # Clean up column names to lowercase match
    full_df.columns = [c.lower() for c in full_df.columns]
    
    # Explicitly cast features to float32 FIRST
    # This ensures any overflows (float64 -> float32 -> inf) are caught in the subsequent clean
    for col in FEATURE_NAMES:
        # errors='coerce' turns non-numeric junk into NaNs
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce').astype('float32')

    # Handle Infinite values (e.g. from volume delta 0 division or float32 overflow)
    import numpy as np
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Ensure all required features exist (fill NaNs for very start of history if needed, or drop)
    # We drop because indicators need warmup
    full_df.dropna(subset=FEATURE_NAMES + ["target"], inplace=True)
    
    return full_df

def validate_data(X, y, name="dataset"):
    """Checks for NaNs, Infs, or object types."""
    import numpy as np
    logging.info("Validating %s...", name)
    
    # Check X for NaNs
    if X.isnull().values.any():
        null_cols = X.columns[X.isnull().any()].tolist()
        logging.error("%s contains NaNs in columns: %s", name, null_cols)
        raise ValueError(f"{name} contains NaNs!")
        
    # Check X for Infs
    if np.isinf(X.values).any():
        inf_cols = X.columns[np.isinf(X.values).any(axis=0)].tolist()
        logging.error("%s contains Infs in columns: %s", name, inf_cols)
        raise ValueError(f"{name} contains Infs!")
    
    # Check types
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        raise ValueError(f"{name} contains non-numeric columns: {non_numeric}")

    # Check y
    if y.isnull().values.any():
        raise ValueError(f"{name} target contains NaNs!")
    if np.isinf(y.values).any():
        raise ValueError(f"{name} target contains Infs!")
        
    logging.info("%s validation passed. Shape: %s", name, X.shape)

def train_and_evaluate(
    X_train, y_train, X_val, y_val, params
):
    """Trains XGBoost and evaluates."""
    
    pos_frac = y_train.mean()
    suggested_scale = (1 - pos_frac) / pos_frac
    logging.info("Training set positive fraction: %.4f (Suggested scale_pos_weight: %.2f)", pos_frac, suggested_scale)
    
    scale = params["scale_pos_weight"]
    if scale == 0.0: 
        scale = suggested_scale

    validate_data(X_train, y_train, "Train Set")
    validate_data(X_val, y_val, "Validation Set")

    clf = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=params["learning_rate"],
        max_depth=params["xgb_max_depth"],
        min_child_weight=params["xgb_min_child_weight"],
        subsample=params["xgb_subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        reg_alpha=params["alpha"],
        reg_lambda=params["reg_lambda"],
        scale_pos_weight=scale,
        tree_method="hist",
        early_stopping_rounds=params["early_stopping_rounds"],
        eval_metric="aucpr",
        random_state=42,
        n_jobs=2  # Limit cores to prevent potential segfaults/instability
    )
    
    logging.info("Starting XGBoost training...")
    import sys
    sys.stdout.flush()
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )
    
    logging.info("Training finished.")
    y_prob = clf.predict_proba(X_val)[:, 1]
    
    pr_auc = average_precision_score(y_val, y_prob)
    brier = brier_score_loss(y_val, y_prob)
    
    results = pd.DataFrame({"prob": y_prob, "true": y_val})
    results.sort_values("prob", ascending=False, inplace=True)
    top_100 = results.head(100)
    prec_at_100 = top_100["true"].mean()
    threshold_at_100 = float(top_100["prob"].min())
    
    metrics = {
        "pr_auc": float(pr_auc),
        "brier": float(brier),
        "prec_at_100": float(prec_at_100),
        "threshold_at_100": threshold_at_100,
        "best_iteration": int(clf.best_iteration)
    }
    
    return clf, metrics

def save_feature_importance(model, feature_names, model_dir: str):
    """Saves feature importance to GCS."""
    if not model_dir.startswith("gs://"):
        logging.warning("AIP_MODEL_DIR not GCS, skipping feature importance upload.")
        return

    bucket_name = urllib.parse.urlparse(model_dir).netloc
    prefix = urllib.parse.urlparse(model_dir).path.lstrip("/")
    
    # Get importance
    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)
    
    with tempfile.TemporaryDirectory() as tmp:
        local_path = os.path.join(tmp, "feature_importance.csv")
        fi_df.to_csv(local_path, index=False)
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(os.path.join(prefix, "feature_importance.csv"))
        blob.upload_from_filename(local_path)
        logging.info("Feature importance saved to %s", blob.public_url)
        
        # Also log top 20 to console
        logging.info("Top 20 Features:\n%s", fi_df.head(20).to_string())

def save_artifacts(model, model_dir: str):
    """Saves model to GCS."""
    if not model_dir.startswith("gs://"):
        logging.error("AIP_MODEL_DIR must be GCS.")
        return
        
    bucket_name = urllib.parse.urlparse(model_dir).netloc
    prefix = urllib.parse.urlparse(model_dir).path.lstrip("/")
    
    with tempfile.TemporaryDirectory() as tmp:
        local_path = os.path.join(tmp, "model.json")
        model.save_model(local_path)
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(os.path.join(prefix, "model.json"))
        blob.upload_from_filename(local_path)
        logging.info("Model saved to %s", blob.public_url)

def save_threshold(threshold: float, model_dir: str):

    """Saves threshold to GCS."""

    if not model_dir.startswith("gs://"):

        return



    bucket_name = urllib.parse.urlparse(model_dir).netloc

    prefix = urllib.parse.urlparse(model_dir).path.lstrip("/")

    

    import json

    with tempfile.TemporaryDirectory() as tmp:

        local_path = os.path.join(tmp, "threshold.json")

        with open(local_path, "w") as f:

            json.dump({"threshold": threshold}, f)

        

        client = storage.Client()

        bucket = client.bucket(bucket_name)

        blob = bucket.blob(os.path.join(prefix, "threshold.json"))

        blob.upload_from_filename(local_path)

        logging.info("Threshold saved to %s", blob.public_url)



def promote_model(model_dir: str, prod_dir: str = "gs://profitscout-lx6bb-pipeline-artifacts/production/model"):

    """Copies artifacts to a stable production location."""

    if not model_dir.startswith("gs://"):

        return

        

    logging.info("Promoting model to %s...", prod_dir)

    client = storage.Client()

    

    # Parse source

    src_bucket_name = urllib.parse.urlparse(model_dir).netloc

    src_prefix = urllib.parse.urlparse(model_dir).path.lstrip("/")

    src_bucket = client.bucket(src_bucket_name)

    

    # Parse dest

    dst_bucket_name = urllib.parse.urlparse(prod_dir).netloc

    dst_prefix = urllib.parse.urlparse(prod_dir).path.lstrip("/")

    dst_bucket = client.bucket(dst_bucket_name)

    

    # Copy files

    for filename in ["model.json", "threshold.json", "feature_importance.csv"]:

        src_blob = src_bucket.blob(os.path.join(src_prefix, filename))

        if src_blob.exists():

            dst_blob = dst_bucket.blob(os.path.join(dst_prefix, filename))

            src_bucket.copy_blob(src_blob, dst_bucket, dst_blob.name)

            logging.info("Promoted %s", filename)



def main():

    parser = argparse.ArgumentParser()


    parser.add_argument("--project-id", required=True)
    parser.add_argument("--source-table", required=True)
    parser.add_argument("--experiment-name", default="profitscout-high-gamma")
    parser.add_argument("--run-name", default=f"run-{pd.Timestamp.now():%Y%m%d-%H%M%S}")
    
    hp = parser.add_argument
    hp("--xgb-max-depth", dest="xgb_max_depth", type=int, default=6)
    hp("--learning-rate", dest="learning_rate", type=float, default=0.03)
    hp("--xgb-min-child-weight", dest="xgb_min_child_weight", type=int, default=10)
    hp("--xgb-subsample", dest="xgb_subsample", type=float, default=0.8)
    hp("--colsample-bytree", dest="colsample_bytree", type=float, default=0.8)
    hp("--gamma", type=float, default=0.1)
    hp("--alpha", type=float, default=0.001)
    hp("--reg-lambda", dest="reg_lambda", type=float, default=1.0)
    hp("--scale-pos-weight", dest="scale_pos_weight", type=float, default=0.0) 

    args = parser.parse_args()
    
    params = vars(args)
    params["early_stopping_rounds"] = 50

    try:
        aiplatform.init(project=args.project_id, experiment=args.experiment_name)
        aiplatform.start_run(run=args.run_name)
        aiplatform.log_params(params)
    except Exception as e:
        logging.warning("Vertex AI init failed: %s", e)

    raw_df = load_raw_price_data(args.project_id, args.source_table)
    
    df = engineer_features(raw_df)
    logging.info("Feature Matrix shape: %s", df.shape)
    logging.info("Target distribution:\n%s", df["target"].value_counts(normalize=True))
    sys.stdout.flush()

    del raw_df
    gc.collect()

    df.sort_values("date", inplace=True)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["target"]
    X_val = val_df[FEATURE_NAMES]
    y_val = val_df["target"]
    
    logging.info("Train shape: %s, Val shape: %s", X_train.shape, X_val.shape)

    model, metrics = train_and_evaluate(X_train, y_train, X_val, y_val, params)
    
    for k, v in metrics.items():
        logging.info("%s: %f", k, v)
        try:
             aiplatform.log_metrics({k: v})
        except Exception:
             # Ignore if no active run or other logging error
             pass

    try:
        ht = hypertune.HyperTune()
        ht.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="pr_auc",
            metric_value=metrics["pr_auc"],
            global_step=1
        )
    except:
        pass

    model_dir = os.getenv("AIP_MODEL_DIR")
    if model_dir:
        save_artifacts(model, model_dir)
        save_feature_importance(model, FEATURE_NAMES, model_dir)
        if "threshold_at_100" in metrics:
            save_threshold(metrics["threshold_at_100"], model_dir)
            
        # Promote to stable production path
        promote_model(model_dir)


if __name__ == "__main__":
    import traceback
    import sys
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)