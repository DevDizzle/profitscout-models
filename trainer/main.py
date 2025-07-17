#!/usr/bin/env python3
"""
ProfitScout training script – Vertex AI‑ready, HPO‑friendly.

• Accepts either hyphen or underscore CLI flags (Vertex passes underscores).
• Saves model + preprocessing artifacts back to GCS.
"""

import argparse
import gc
import json
import logging
import os
import tempfile
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
from sklearn.metrics import auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler

# ───────────────── Config ─────────────────
logging.basicConfig(level=logging.INFO)

EMB_COLS = [
    "key_financial_metrics_embedding",
    "key_discussion_points_embedding",
    "sentiment_tone_embedding",
    "short_term_outlook_embedding",
    "forward_looking_signals_embedding", 
]
STATIC_NUM_COLS = [
    "sentiment_score", "sma_20", "ema_50", "rsi_14", "adx_14",
    "sma_20_delta", "ema_50_delta", "rsi_14_delta", "adx_14_delta",
    "eps_surprise",
]
ENGINEERED_COLS = [
    "price_sma20_ratio", "ema50_sma20_ratio", "rsi_centered", "adx_log1p",
    "sent_rsi", "eps_surprise_isnull", "cos_fin_disc", "cos_fin_tone",
    "cos_disc_short", "cos_short_fwd",
]
for c in EMB_COLS:
    base = c.replace("_embedding", "")
    ENGINEERED_COLS.extend([f"{base}_norm", f"{base}_mean", f"{base}_std"])

# ───────────────── Helpers ─────────────────
def load_data(project_id: str, source_table: str, breakout_threshold: float) -> pd.DataFrame:
    """Load & basic‑clean BigQuery training data."""
    logging.info("Loading data from %s.%s", project_id, source_table)
    bq = bigquery.Client(project=project_id)
    df = bq.query(
        f"""
        SELECT *
        FROM `{project_id}.{source_table}`
        WHERE adj_close_on_call_date IS NOT NULL
          AND max_close_30d          IS NOT NULL
          AND earnings_call_date     IS NOT NULL
        """
    ).to_dataframe()
    logging.info("Loaded %d rows", len(df))  # Fixed format

    # convert embedding JSON strings → numpy arrays
    for col in EMB_COLS:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: np.array(json.loads(x)) if isinstance(x, str) and x.startswith("[") else x
            )

    df["earnings_call_date"] = pd.to_datetime(df["earnings_call_date"])
    df.sort_values("earnings_call_date", inplace=True)
    df["breakout"] = (
        (df["max_close_30d"] / df["adj_close_on_call_date"] - 1) >= breakout_threshold
    ).astype(int)
    return df


def preprocess_data(df: pd.DataFrame, params: dict):
    """Winsorize, fill, PCA‑reduce embeddings, return arrays + metadata."""
    split_date = df["earnings_call_date"].quantile(0.8, interpolation="nearest")
    train_idx = df[df["earnings_call_date"] < split_date].index
    holdout_idx = df[df["earnings_call_date"] >= split_date].index

    # winsorize numeric features
    for col in STATIC_NUM_COLS:
        if col in df.columns:
            lo, hi = df.loc[train_idx, col].quantile(params["winsor_quantiles"])
            df[col] = df[col].clip(lo, hi)

    # fill eps_surprise by sector mean
    sector_mean = (
        df.loc[train_idx]
        .groupby("industry_sector")["eps_surprise"]
        .mean()
        .to_dict()
    )
    df["eps_surprise"] = (
        df["eps_surprise"]
        .fillna(df["industry_sector"].map(sector_mean))
        .fillna(0.0)
    )

    # PCA on embeddings
    pca_cols, X_pca_list, pca_map = [], [], {}
    for col in EMB_COLS:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if valid.empty:
            continue

        pca = PCA(n_components=params["pca_n"], random_state=params["random_state"])
        train_ids = train_idx.intersection(valid.index)
        pca.fit(np.vstack(df.loc[train_ids, col].values))
        logging.info(f"PCA for {col}: explained variance ratio sum = {pca.explained_variance_ratio_.sum():.2f}")  # Added for debugging
        transformed = np.full((len(df), params["pca_n"]), np.nan)
        transformed[valid.index] = pca.transform(np.vstack(valid.values))
        X_pca_list.append(transformed)
        pca_map[col] = pca
        pca_cols.extend(f"{col}_pc{i}" for i in range(params["pca_n"]))
        gc.collect()

    X_pca = np.hstack(X_pca_list) if X_pca_list else np.empty((len(df), 0))

    final_num_cols = [c for c in STATIC_NUM_COLS + ENGINEERED_COLS if c in df.columns]
    missing_cols = set(STATIC_NUM_COLS + ENGINEERED_COLS) - set(df.columns)
    if missing_cols:
        logging.warning(f"Missing columns: {missing_cols} – proceeding without them")
    X_num = df[final_num_cols].to_numpy(dtype=np.float32)

    X_all = np.hstack([X_pca, X_num]).astype(np.float32)
    y_all = df["breakout"].values

    feature_names = pca_cols + final_num_cols
    return X_all, y_all, train_idx, holdout_idx, feature_names, pca_map


def train_and_evaluate(X_all, y_all, train_idx, holdout_idx, feature_names, params):
    """Train XGBoost + LogReg ensemble and compute holdout PR‑AUC."""
    # Split train into sub-train and validation for XGBoost early stopping
    split_point = int(len(train_idx) * 0.8)
    sub_train_idx = train_idx[:split_point]
    val_idx = train_idx[split_point:]

    X_subtr, y_subtr = X_all[sub_train_idx], y_all[sub_train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]  # Full train for LogReg and calibrator
    X_hd, y_hd = X_all[holdout_idx], y_all[holdout_idx]

    blend_weight = params.get("blend_weight", 0.5)

    # impute + scale for LogReg (fit on full train)
    imputer = SimpleImputer(strategy="median").fit(X_tr)
    scaler = StandardScaler().fit(imputer.transform(X_tr))

    X_tr_scaled = scaler.transform(imputer.transform(X_tr))
    X_hd_scaled = scaler.transform(imputer.transform(X_hd))

    # XGBoost with validation for early stopping
    dsubtr = xgb.DMatrix(X_subtr, y_subtr, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, y_val, feature_names=feature_names)
    bst = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "learning_rate": 0.03,
            "max_depth": params["xgb_max_depth"],
            "min_child_weight": params["xgb_min_child_weight"],
            "subsample": params["xgb_subsample"],
            "colsample_bytree": 0.9,
            "scale_pos_weight": (y_subtr == 0).sum() / max((y_subtr == 1).sum(), 1),
            "n_jobs": -1,
            "random_state": params["random_state"],
        },
        dsubtr,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=params["early_stopping_rounds"],
        verbose_eval=False,
    )

    # Logistic regression (fit on full train)
    logreg = LogisticRegression(
        max_iter=500, C=params["logreg_c"], solver="lbfgs", random_state=params["random_state"]
    ).fit(X_tr_scaled, y_tr)

    # Compute blends on full train for calibration
    dtr = xgb.DMatrix(X_tr, y_tr, feature_names=feature_names)
    xgb_raw_tr = bst.predict(dtr, iteration_range=(0, bst.best_iteration + 1))
    log_raw_tr = logreg.predict_proba(X_tr_scaled)[:, 1]
    blend_raw_tr = blend_weight * xgb_raw_tr + (1 - blend_weight) * log_raw_tr
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(blend_raw_tr, y_tr)

    # Blend on holdout and calibrate for evaluation
    dhd = xgb.DMatrix(X_hd, y_hd, feature_names=feature_names)
    xgb_raw_hd = bst.predict(dhd, iteration_range=(0, bst.best_iteration + 1))
    log_raw_hd = logreg.predict_proba(X_hd_scaled)[:, 1]
    blend_raw_hd = blend_weight * xgb_raw_hd + (1 - blend_weight) * log_raw_hd
    blend_cal_hd = calibrator.transform(blend_raw_hd)

    prec, rec, _ = precision_recall_curve(y_hd, blend_cal_hd)
    pr_auc = auc(rec, prec)

    metrics = {"pr_auc": pr_auc}
    artifacts = dict(
        imputer=imputer,
        scaler=scaler,
        logreg=logreg,
        calibrator=calibrator,
        threshold=0.5,
    )
    return bst, metrics, artifacts


def save_artifacts(bst, artifacts, pca_map, model_dir: str):
    """Save model + preprocessing objects to GCS under model_dir/."""
    if not model_dir.startswith("gs://"):
        logging.error("model_dir must be a GCS path, got %s", model_dir)
        return

    # dump files to a temp dir first
    with tempfile.TemporaryDirectory() as tmp:
        joblib.dump(bst,        os.path.join(tmp, "xgb_model.joblib"))
        joblib.dump(artifacts,  os.path.join(tmp, "pipeline_artifacts.joblib"))
        joblib.dump(pca_map,    os.path.join(tmp, "pca_map.joblib"))

        # upload to GCS
        bucket_name, *path_parts = model_dir[5:].split("/", 1)
        prefix = path_parts[0] if path_parts else ""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        for fname in os.listdir(tmp):
            blob = bucket.blob(f"{prefix}/{fname}".lstrip("/"))
            blob.upload_from_filename(os.path.join(tmp, fname))
            logging.info("Uploaded %s to gs://%s/%s", fname, bucket_name, blob.name)


# ───────────────── Main ─────────────────
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-id",   required=True)
    parser.add_argument("--source-table", required=True)
    parser.add_argument("--experiment-name", default="profitscout-training")
    parser.add_argument("--run-name",
                        default=f"run-{pd.Timestamp.now():%Y%m%d-%H%M%S}")

    # hyper‑parameters – hyphen + underscore forms
    parser.add_argument("--pca-n",                  dest="pca_n", type=int,   default=64)
    parser.add_argument("--pca_n",                  dest="pca_n", type=int,   default=argparse.SUPPRESS)
    parser.add_argument("--xgb-max-depth",          dest="xgb_max_depth", type=int, default=7)
    parser.add_argument("--xgb_max_depth",          dest="xgb_max_depth", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--xgb-min-child-weight",   dest="xgb_min_child_weight", type=int, default=5)
    parser.add_argument("--xgb_min_child_weight",   dest="xgb_min_child_weight", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--xgb-subsample",          dest="xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_subsample",          dest="xgb_subsample", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--logreg-c",               dest="logreg_c", type=float, default=0.1)
    parser.add_argument("--logreg_c",               dest="logreg_c", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--blend-weight",           dest="blend_weight", type=float, default=0.5)
    parser.add_argument("--blend_weight",           dest="blend_weight", type=float, default=argparse.SUPPRESS)

    args = parser.parse_args()

    params = dict(
        pca_n=args.pca_n,
        random_state=42,
        early_stopping_rounds=100,
        thresh_req_recall=0.5,
        winsor_quantiles=(0.01, 0.99),
        breakout_threshold=0.12,
        xgb_max_depth=args.xgb_max_depth,
        xgb_min_child_weight=args.xgb_min_child_weight,
        xgb_subsample=args.xgb_subsample,
        logreg_c=args.logreg_c,
        blend_weight=args.blend_weight,
    )

    # ── optional experiment tracking ─────────────────────────────
    experiment_ok = False
    try:
        aiplatform.init(
            project=args.project_id,
            location="us-central1",
            experiment=args.experiment_name,
        )
        aiplatform.start_run(run=args.run_name)
        aiplatform.log_params(params)
        experiment_ok = True
    except Exception as e:
        logging.warning("Skipping Vertex AI experiment tracking: %s", e)
    # ─────────────────────────────────────────────────────────────

    df = load_data(args.project_id, args.source_table, params["breakout_threshold"])
    X_all, y_all, train_idx, holdout_idx, f_names, pca_map = preprocess_data(df, params)
    bst, metrics, artifacts = train_and_evaluate(
        X_all, y_all, train_idx, holdout_idx, f_names, params
    )

    # report to HPT
    hypertune.HyperTune().report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="pr_auc",
        metric_value=metrics["pr_auc"],
        global_step=1,
    )
    if experiment_ok:
        aiplatform.log_metrics(metrics)

    model_dir = os.environ.get("AIP_MODEL_DIR")
    if model_dir:
        save_artifacts(bst, artifacts, pca_map, model_dir)
        if experiment_ok:  # Guarded to avoid error if init failed
            aiplatform.log_params({"model_dir": model_dir})

    logging.info("Training run complete.")


if __name__ == "__main__":
    main()