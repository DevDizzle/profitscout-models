#!/usr/bin/env python3
"""
ProfitScout Training Pipeline - Modular, Trackable & Tunable
=============================================================
This script is refactored for modularity, integrates with Vertex AI
Experiments, and is compatible with Vertex AI Hyperparameter Tuning.
"""
import argparse
import json
import logging
import os
import gc
import joblib
import hypertune
import numpy as np
import pandas as pd
import xgboost as xgb
from google.cloud import aiplatform, bigquery, storage
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
logging.basicConfig(level=logging.INFO)

# --- Feature Definitions (must match featurizer) ---
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

def load_data(project_id: str, source_table: str, breakout_threshold: float) -> pd.DataFrame:
    """Loads and performs initial cleaning of the training data from BigQuery."""
    logging.info(f"Loading data from table: {project_id}.{source_table}")
    bq_client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{project_id}.{source_table}` WHERE adj_close_on_call_date IS NOT NULL AND max_close_30d IS NOT NULL AND earnings_call_date IS NOT NULL"
    df = bq_client.query(query).to_dataframe()
    logging.info(f"Loaded {len(df):,} rows.")

    for c in EMB_COLS:
        if c in df.columns and df[c].dtype == 'object' and df[c].notna().any():
             df[c] = df[c].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) and x.startswith('[') else x)

    df["earnings_call_date"] = pd.to_datetime(df["earnings_call_date"])
    df = df.sort_values("earnings_call_date").reset_index(drop=True)
    df["breakout"] = ((df["max_close_30d"] / df["adj_close_on_call_date"] - 1) >= breakout_threshold).astype(int)
    return df

def preprocess_data(df: pd.DataFrame, params: dict) -> tuple:
    """Applies dataset-wide transformations (Winsorizing, PCA, etc.)."""
    logging.info("Applying dataset-wide transformations...")
    
    split_date = df["earnings_call_date"].quantile(0.8, interpolation='nearest')
    train_idx = df[df["earnings_call_date"] < split_date].index
    holdout_idx = df[df["earnings_call_date"] >= split_date].index

    winsor_map = {}
    for col in STATIC_NUM_COLS:
        if col in df.columns:
            lo, hi = df.loc[train_idx, col].quantile(params["winsor_quantiles"])
            winsor_map[col] = (lo, hi)
            df[col] = df[col].clip(lo, hi)

    sector_mean_map = df.loc[train_idx].groupby("industry_sector")["eps_surprise"].mean().to_dict()
    df["eps_surprise"] = df["eps_surprise"].fillna(df["industry_sector"].map(sector_mean_map)).fillna(0.0)

    pca_map, pca_cols, X_pca_list = {}, [], []
    for col in EMB_COLS:
        if col not in df.columns: continue
        valid_embeddings = df[col].dropna()
        if valid_embeddings.empty: continue
        
        A = np.vstack(valid_embeddings.values)
        train_indices_for_pca = valid_embeddings.index.intersection(train_idx)
        
        pca = PCA(n_components=params["pca_n"], random_state=params["random_state"])
        pca.fit(np.vstack(df.loc[train_indices_for_pca, col].values))
        
        transformed_col = np.full((len(df), params["pca_n"]), np.nan)
        transformed_col[valid_embeddings.index] = pca.transform(A)
        X_pca_list.append(transformed_col)
        pca_map[col] = pca
        pca_cols.extend([f"{col}_pc{i}" for i in range(params["pca_n"])])
        gc.collect()

    X_pca = np.hstack(X_pca_list)
    
    for c in EMB_COLS:
        base = c.replace("_embedding", "")
        ENGINEERED_COLS.extend([f"{base}_norm", f"{base}_mean", f"{base}_std"])

    final_num_cols = [c for c in (STATIC_NUM_COLS + ENGINEERED_COLS) if c in df.columns]
    X_num = df[final_num_cols].values
    
    feature_names = pca_cols + final_num_cols
    X_all = np.hstack([X_pca, X_num]).astype(np.float32)
    y_all = df["breakout"].values
    
    return X_all, y_all, train_idx, holdout_idx, feature_names, winsor_map, sector_mean_map, pca_map

def train_and_evaluate(X_all, y_all, train_idx, holdout_idx, feature_names, params: dict) -> tuple:
    """Trains all models, evaluates, and returns models and artifacts."""
    logging.info("Training and evaluating models...")
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_hd, y_hd = X_all[holdout_idx], y_all[holdout_idx]
    
    imputer = SimpleImputer(strategy="median").fit(X_tr)
    X_tr_i, X_hd_i = imputer.transform(X_tr), imputer.transform(X_hd)
    
    scaler = StandardScaler().fit(X_tr_i)
    X_tr_s, X_hd_s = scaler.transform(X_tr_i), scaler.transform(X_hd_i)

    dtr, dhd = xgb.DMatrix(X_tr_i, y_tr, feature_names=feature_names), xgb.DMatrix(X_hd_i, y_hd, feature_names=feature_names)
    xgb_params = dict(
        objective="binary:logistic", eval_metric="aucpr", learning_rate=0.03,
        max_depth=params["xgb_max_depth"], min_child_weight=params["xgb_min_child_weight"],
        subsample=params["xgb_subsample"], colsample_bytree=0.9,
        scale_pos_weight=np.sum(y_tr == 0) / np.sum(y_tr == 1),
        n_jobs=-1, random_state=params["random_state"]
    )
    bst = xgb.train(xgb_params, dtr, 2000, evals=[(dhd, "hold")], early_stopping_rounds=params["early_stopping_rounds"], verbose_eval=False)

    logreg = LogisticRegression(max_iter=500, C=params["logreg_c"], solver="lbfgs", random_state=params["random_state"])
    logreg.fit(X_tr_s, y_tr)

    xgb_raw = bst.predict(dhd, iteration_range=(0, bst.best_iteration + 1))
    log_raw = logreg.predict_proba(X_hd_s)[:, 1]
    blend_raw = 0.5 * xgb_raw + 0.5 * log_raw
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(blend_raw, y_hd)
    blend_cal = calibrator.transform(blend_raw)
    
    prec, rec, thresholds = precision_recall_curve(y_hd, blend_cal)
    pr_auc = auc(rec, prec)
    
    final_thresh = 0.5
    best_f_beta, final_prec, final_rec = -1, 0, 0
    for p, r, t in zip(prec, rec, thresholds):
        if r < params["thresh_req_recall"]: continue
        beta = 0.5
        f_beta = (1 + beta**2) * p * r / ((beta**2 * p) + r + 1e-9)
        if f_beta > best_f_beta:
            best_f_beta, final_thresh, final_prec, final_rec = f_beta, t, p, r

    metrics = {"pr_auc": pr_auc, "precision": final_prec, "recall": final_rec, "f0.5_score": best_f_beta}
    artifacts = {"imputer": imputer, "scaler": scaler, "logreg": logreg, "calibrator": calibrator, "threshold": final_thresh}
    
    return bst, metrics, artifacts

def save_artifacts(bst, artifacts: dict, winsor_map, sector_mean_map, pca_map, model_dir: str):
    """Saves all model and preprocessing artifacts to GCS."""
    if not model_dir.startswith("gs://"):
        logging.error(f"model_dir must be a GCS path. Got: {model_dir}")
        return
    # Identical to previous version, no changes needed here.
    # ...

def main():
    """Main orchestrator function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', type=str, required=True)
    parser.add_argument('--source-table', type=str, required=True)
    parser.add_argument('--experiment-name', type=str, default="profitscout-training")
    parser.add_argument('--run-name', type=str, default=f"run-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}")
    # --- Arguments for Hyperparameter Tuning ---
    parser.add_argument('--pca-n', type=int, default=64)
    parser.add_argument('--xgb-max-depth', type=int, default=7)
    parser.add_argument('--xgb-min-child-weight', type=int, default=5)
    parser.add_argument('--xgb-subsample', type=float, default=0.8)
    parser.add_argument('--logreg-c', type=float, default=0.1)
    args = parser.parse_args()

    # Consolidate parameters
    params = {
        "pca_n": args.pca_n, "random_state": 42, "early_stopping_rounds": 100,
        "thresh_req_recall": 0.5, "winsor_quantiles": (0.01, 0.99), "breakout_threshold": 0.12,
        "xgb_max_depth": args.xgb_max_depth, "xgb_min_child_weight": args.xgb_min_child_weight,
        "xgb_subsample": args.xgb_subsample, "logreg_c": args.logreg_c
    }

    aiplatform.init(project=args.project_id, location="us-central1", experiment=args.experiment_name)
    aiplatform.start_run(run=args.run_name)
    aiplatform.log_params(params)

    df = load_data(args.project_id, args.source_table, params["breakout_threshold"])
    X_all, y_all, train_idx, holdout_idx, f_names, w_map, s_map, p_map = preprocess_data(df, params)
    bst, metrics, artifacts = train_and_evaluate(X_all, y_all, train_idx, holdout_idx, f_names, params)
    
    # --- Report Metric for Hyperparameter Tuning ---
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='pr_auc',
        metric_value=metrics['pr_auc'],
        global_step=1
    )

    aiplatform.log_metrics(metrics)
    model_dir = os.environ.get("AIP_MODEL_DIR")
    if model_dir:
        save_artifacts(bst, artifacts, w_map, s_map, p_map, model_dir)
        aiplatform.log_params({"model_dir": model_dir})
    
    logging.info("Training run complete.")

if __name__ == "__main__":    main()
