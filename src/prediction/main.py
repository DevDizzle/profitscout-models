#!/usr/bin/env python3
"""
ProfitScout Batch Prediction Script
Loads model + preprocessing artifacts from GCS and writes batch predictions to BigQuery.
"""

# --- stdlib / deps -----------------------------------------------------------
import argparse
import json
import logging
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from google.cloud import bigquery, storage
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)

# --- Feature Definitions (must match trainer) ---
EMB_COLS = [
    "key_financial_metrics_embedding",
    "key_discussion_points_embedding",
    "sentiment_tone_embedding",
    "short_term_outlook_embedding",
    "forward_looking_signals_embedding",
]
STATIC_NUM_COLS = [
    "sentiment_score",
    "sma_20",
    "ema_50",
    "rsi_14",
    "adx_14",
    "sma_20_delta",
    "ema_50_delta",
    "rsi_14_delta",
    "adx_14_delta",
    "eps_surprise",
]
ENGINEERED_COLS = [
    "price_sma20_ratio",
    "ema50_sma20_ratio",
    "rsi_centered",
    "adx_log1p",
    "sent_rsi",
    "eps_surprise_isnull",
    "cos_fin_disc",
    "cos_fin_tone",
    "cos_disc_short",
    "cos_short_fwd",
]


# --- Modular Functions ---

def load_artifacts(model_dir: str):
    """Download model + artifacts from GCS."""
    bucket_name, blob_prefix = model_dir.replace("gs://", "").split("/", 1)
    bucket = storage.Client().bucket(bucket_name)

    # ---- Booster ---------------------------------------------------------
    model_blob = bucket.blob(f"{blob_prefix}/xgb_model.json")
    if model_blob.exists():
        local = "xgb_model.json"
        model_blob.download_to_filename(local)
        bst = xgb.Booster();  bst.load_model(local)
    elif bucket.blob(f"{blob_prefix}/xgb_model.ubj").exists():
        local = "xgb_model.ubj"
        bucket.blob(f"{blob_prefix}/xgb_model.ubj").download_to_filename(local)
        bst = xgb.Booster();  bst.load_model(local)
    else:
        local = "xgb_model.joblib"
        bucket.blob(f"{blob_prefix}/xgb_model.joblib").download_to_filename(local)
        bst = joblib.load(local)

    # ---- preprocessing artifacts ----------------------------------------
    bucket.blob(f"{blob_prefix}/training_artifacts.joblib") \
          .download_to_filename("training_artifacts.joblib")
    artifacts = joblib.load("training_artifacts.joblib")

    # ---- PCA map --------------------------------------------------------
    bucket.blob(f"{blob_prefix}/pca_map.joblib") \
          .download_to_filename("pca_map.joblib")
    pca_map = joblib.load("pca_map.joblib")
    artifacts["pca_map"] = pca_map  # inject into artifacts

    return bst, artifacts


def load_data(project_id: str, source_table: str) -> pd.DataFrame:
    """Loads prediction data from a BigQuery table."""
    fq_table = source_table if source_table.count(".") >= 2 else f"{project_id}.{source_table}"
    logging.info(f"Loading prediction data from {fq_table}")
    df = bigquery.Client(project=project_id).query(f"SELECT * FROM `{fq_table}`").to_dataframe()
    logging.info(f"Loaded {len(df):,} rows for prediction.")
    return df


def preprocess_for_inference(df: pd.DataFrame, artifacts: dict):
    """Applies transformations to new data using loaded artifacts."""
    logging.info("Applying inference transformations…")

    # --- Type correction for embeddings ---
    for c in EMB_COLS:
        if c in df.columns and df[c].dtype == "object" and df[c].notna().any():
            df[c] = df[c].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) and x.startswith("[") else x)

    # --- Winsorize ---
    for col, (lo, hi) in artifacts["winsor_map"].items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # --- Fill missing EPS surprise ---
    df["eps_surprise"] = (
        df["eps_surprise"].fillna(df["industry_sector"].map(artifacts["sector_mean_map"]))
        .fillna(0.0)
    )

    # --- PCA for embeddings ---
    X_pca_list = []
    pca_feature_names: list[str] = []
    for col in EMB_COLS:
        if col not in artifacts["pca_map"]:
            logging.warning(f"No PCA model found for {col}; skipping")
            continue

        pca = artifacts["pca_map"][col]
        n_comp = pca.n_components_
        pca_feature_names.extend(f"{col}_pc{i}" for i in range(n_comp))

        if col not in df.columns or df[col].dropna().empty:
            X_pca_list.append(np.full((len(df), n_comp), np.nan))
            continue

        valid = df[col].dropna()
        transformed = np.full((len(df), n_comp), np.nan, dtype=np.float32)
        transformed[valid.index] = pca.transform(np.vstack(valid.values))
        X_pca_list.append(transformed)

    X_pca = np.hstack(X_pca_list) if X_pca_list else np.empty((len(df), 0), np.float32)

    # --- numeric features ---
    engineered = ENGINEERED_COLS.copy()
    for c in EMB_COLS:
        base = c.replace("_embedding", "")
        engineered.extend([f"{base}_norm", f"{base}_mean", f"{base}_std"])

    final_num_cols = [c for c in (STATIC_NUM_COLS + engineered) if c in df.columns]
    X_num = df[final_num_cols].to_numpy(dtype=np.float32)

    feature_names = pca_feature_names + final_num_cols
    X_all = np.hstack([X_pca, X_num]).astype(np.float32)

    # --- Impute & scale (LogReg path) ---
    X_imp = artifacts["imputer"].transform(X_all)
    X_scl = artifacts["scaler"].transform(X_imp)

    return X_imp, X_scl, feature_names


def make_predictions(bst, artifacts: dict, X_imp: np.ndarray, X_scl: np.ndarray, feature_names: list):
    """Generates calibrated blended predictions."""
    logging.info("Generating predictions…")

    dpred = xgb.DMatrix(X_imp, feature_names=feature_names)
    xgb_raw = bst.predict(dpred)
    log_raw = artifacts["logreg"].predict_proba(X_scl)[:, 1]

    w = artifacts.get("blend_weight", 0.5)
    blend_raw = w * xgb_raw + (1 - w) * log_raw
    prob = artifacts["calibrator"].transform(blend_raw)
    preds = (prob >= artifacts["threshold"]).astype(int)

    return pd.DataFrame({
        "calibrated_probability": prob,
        "prediction": preds,
    })


def save_predictions(df: pd.DataFrame, project_id: str, destination_table: str):
    """Saves the predictions back to a BigQuery table."""
    logging.info(f"Saving {len(df):,} predictions to {project_id}.{destination_table}")
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("ticker", "STRING"),
            bigquery.SchemaField("quarter_end_date", "DATE"),
            bigquery.SchemaField("earnings_call_date", "DATE"),
            bigquery.SchemaField("calibrated_probability", "FLOAT64"),
            bigquery.SchemaField("prediction", "INTEGER"),
        ],
    )
    bigquery.Client(project=project_id).load_table_from_dataframe(df, destination_table, job_config=job_config).result()
    logging.info("✅ Predictions saved successfully.")


def main() -> None:
    """Entry‑point for batch prediction."""
    parser = argparse.ArgumentParser()

    # ───────── required parameters ─────────
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--source-table", required=True)
    parser.add_argument("--destination-table", required=True)
    parser.add_argument("--model-dir", required=True)

    # ───────── optional / ignored in inference ─────────

    parser.add_argument("--top-k-features", default="0")
    parser.add_argument("--auto-prune",    default="false")
    parser.add_argument("--metric-tol",    default="0.002")
    parser.add_argument("--prune-step",    default="25")

    # If any *unknown* CLI flags are added later, ignore them gracefully.
    args, _ = parser.parse_known_args()

    # ------------------------ run pipeline ------------------------
    bst, artifacts   = load_artifacts(args.model_dir)
    df_source        = load_data(args.project_id, args.source_table)

    if df_source.empty:
        logging.info("No rows to predict on – exiting.")
        return

    X_imp, X_scl, feat_names = preprocess_for_inference(df_source, artifacts)
    df_preds = make_predictions(bst, artifacts, X_imp, X_scl, feat_names)

    df_out = pd.concat(
        [
            df_source[["ticker", "quarter_end_date", "earnings_call_date"]].reset_index(drop=True),
            df_preds,
        ],
        axis=1,
    )
    save_predictions(df_out, args.project_id, args.destination_table)


if __name__ == "__main__":
    main()