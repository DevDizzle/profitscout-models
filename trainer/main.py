#!/usr/bin/env python3
"""
ProfitScout training script – Vertex AI-ready, HPO-friendly.

• Accepts either hyphen or underscore CLI flags (Vertex passes underscores).
• Saves model + preprocessing artifacts back to GCS.
"""

import argparse
import gc
import json
import logging
import os
import tempfile
import urllib.parse
import joblib
import hypertune
import numpy as np
import pandas as pd
import xgboost as xgb
from google.cloud import aiplatform, bigquery, storage
from imblearn.over_sampling import SMOTE
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
    "qa_summary_embedding",
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

LDA_COLS = [f"lda_topic_{i}" for i in range(10)]
FINBERT_COLS = []
for sec in ["key_financial_metrics", "key_discussion_points", "sentiment_tone", "short_term_outlook", "forward_looking_signals", "qa_summary"]:
    FINBERT_COLS.extend([f"{sec}_pos_prob", f"{sec}_neg_prob", f"{sec}_neu_prob"])

# ───────────────── Helpers ─────────────────
def load_data(project_id: str, source_table: str, breakout_threshold: float) -> pd.DataFrame:
    """Load & basic-clean BigQuery training data."""
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
    logging.info("Loaded %d rows", len(df))

    # convert embedding JSON strings/lists → numpy arrays with error handling
    invalid_count = {col: 0 for col in EMB_COLS}
    for col in EMB_COLS:
        if col in df.columns:
            def parse_emb(x):
                try:
                    if isinstance(x, str) and x.startswith("["):
                        arr = np.array(json.loads(x), dtype=np.float32)
                    elif isinstance(x, (list, np.ndarray)):
                        arr = np.array(x, dtype=np.float32)
                    else:
                        return np.nan
                    if arr.size == 0 or arr.ndim != 1:  # Ensure non-empty 1D vector
                        invalid_count[col] += 1
                        return np.nan
                    return arr
                except Exception:
                    invalid_count[col] += 1
                    return np.nan
            df[col] = df[col].apply(parse_emb)
    for col, count in invalid_count.items():
        if count > 0:
            logging.warning("Invalid/empty embeddings in %s: %d rows set to NaN", col, count)

    df["earnings_call_date"] = pd.to_datetime(df["earnings_call_date"])
    df.sort_values("earnings_call_date", inplace=True)
    df["breakout"] = (
        (df["max_close_30d"] / df["adj_close_on_call_date"] - 1) >= breakout_threshold
    ).astype(int)
    logging.info("Breakout rate: %.2f%%", 100 * df["breakout"].mean())
    return df


def preprocess_data(df: pd.DataFrame, params: dict):
    """Winsorize, fill, PCA-reduce embeddings, return arrays + metadata."""
    split_date = df["earnings_call_date"].quantile(0.8, interpolation="nearest")
    train_idx = df[df["earnings_call_date"] < split_date].index
    holdout_idx = df[df["earnings_call_date"] >= split_date].index

    # winsorize numeric features
    num_cols_to_winsor = STATIC_NUM_COLS + ENGINEERED_COLS + LDA_COLS + FINBERT_COLS
    for col in num_cols_to_winsor:
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
            logging.warning(f"No valid embeddings for {col}, skipping PCA")
            continue

        # Ensure consistent embedding dimension (filter outliers if any)
        emb_lengths = valid.apply(lambda x: x.shape[0])
        if len(emb_lengths.unique()) > 1:
            common_len = emb_lengths.mode()[0]
            logging.warning(f"Inconsistent embedding lengths in {col}; keeping only length {common_len}")
            valid = valid[emb_lengths == common_len]
        if valid.empty:
            continue

        train_ids = train_idx.intersection(valid.index)
        n_train = len(train_ids)
        if n_train == 0:
            logging.warning(f"No valid training embeddings for {col}, skipping PCA")
            continue

        stacked_train = np.vstack(df.loc[train_ids, col].values)
        emb_dim = stacked_train.shape[1]
        pca_n = min(params["pca_n"], min(n_train, emb_dim))
        if pca_n == 0:
            logging.warning(f"Zero components for {col}, skipping PCA")
            continue

        pca = PCA(n_components=pca_n, random_state=params["random_state"])
        pca.fit(stacked_train)
        evr_sum = pca.explained_variance_ratio_.sum()
        logging.info(f"PCA for {col}: {pca.n_components_} components, explained variance ratio sum = {evr_sum:.2f}")
        if evr_sum < 0.8:
            logging.warning(f"Low explained variance ({evr_sum:.2f}) for {col} – consider more components or better features")
        if pca.n_components_ < pca_n:
            logging.warning(f"PCA for {col} reduced to {pca.n_components_} components (low rank/samples)")

        # Transform all valid (including holdout)
        transformed = np.full((len(df), pca.n_components_), np.nan)
        valid_idx_pos = df.index.get_indexer(valid.index)
        transformed[valid_idx_pos] = pca.transform(np.vstack(valid.values))
        X_pca_list.append(transformed)
        pca_map[col] = pca
        pca_cols.extend(f"{col}_pc{i}" for i in range(pca.n_components_))
        gc.collect()

    X_pca = np.hstack(X_pca_list) if X_pca_list else np.empty((len(df), 0))

    final_num_cols = [c for c in STATIC_NUM_COLS + ENGINEERED_COLS + LDA_COLS + FINBERT_COLS if c in df.columns]
    missing_cols = set(STATIC_NUM_COLS + ENGINEERED_COLS + LDA_COLS + FINBERT_COLS) - set(df.columns)
    if missing_cols:
        logging.warning(f"Missing columns: {missing_cols} – proceeding without them")
    X_num = df[final_num_cols].to_numpy(dtype=np.float32)

    X_all = np.hstack([X_pca, X_num]).astype(np.float32)
    y_all = df["breakout"].values

    feature_names = pca_cols + final_num_cols
    logging.info(f"Feature matrix shape: {X_all.shape}, Train/Holdout: {len(train_idx)}/{len(holdout_idx)}")
    logging.info(f"Train breakout rate: {y_all[train_idx].mean():.2%}, Holdout: {y_all[holdout_idx].mean():.2%}")

    imputer = SimpleImputer(strategy="constant", fill_value=0)
    imputer.fit(X_all[train_idx])
    X_all = imputer.transform(X_all)
    if np.isnan(X_all).any():
        logging.warning("NaNs remain after imputation – check for all-NaN columns in features")

    return X_all, y_all, train_idx, holdout_idx, feature_names, pca_map, imputer


def train_and_evaluate(X_all, y_all, train_idx, holdout_idx, feature_names, params, imputer):
    """Train XGBoost + LogReg ensemble and compute holdout PR-AUC."""
    try:
        # Split train into sub-train and validation for XGBoost early stopping
        split_point = int(len(train_idx) * 0.8)
        sub_train_idx = train_idx[:split_point]
        val_idx = train_idx[split_point:]

        X_subtr, y_subtr = X_all[sub_train_idx], y_all[sub_train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]  # Full train for LogReg and calibrator
        X_hd, y_hd = X_all[holdout_idx], y_all[holdout_idx]

        blend_weight = params.get("blend_weight", 0.5)

        # Scale for LogReg (fit on full train; no need for imputer now)
        scaler = StandardScaler().fit(X_tr)

        X_tr_scaled = scaler.transform(X_tr)
        X_hd_scaled = scaler.transform(X_hd)

        # Apply SMOTE if enabled for XGBoost sub-train
        if params["use_smote"]:
            logging.info("Applying SMOTE to sub-train for XGBoost")
            smote = SMOTE(random_state=params["random_state"])
            X_subtr_res, y_subtr_res = smote.fit_resample(X_subtr, y_subtr)
        else:
            X_subtr_res, y_subtr_res = X_subtr, y_subtr

        # XGBoost with validation for early stopping
        dsubtr = xgb.DMatrix(X_subtr_res, y_subtr_res, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, y_val, feature_names=feature_names)
        bst = xgb.train(
            {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "learning_rate": params["learning_rate"],
                "max_depth": params["xgb_max_depth"],
                "min_child_weight": params["xgb_min_child_weight"],
                "subsample": params["xgb_subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "gamma": params["gamma"],
                "scale_pos_weight": params["scale_pos_weight"],  # Now tunable
                "alpha": params["alpha"],  # New: L1 reg
                "lambda": params["reg_lambda"],  # New: L2 reg
                "tree_method": "hist",  # Faster for high-dim features
                "n_jobs": -1,
                "random_state": params["random_state"],
            },
            dsubtr,
            num_boost_round=3000,
            evals=[(dval, "val")],
            early_stopping_rounds=params["early_stopping_rounds"],
            verbose_eval=False,
        )

        # Apply SMOTE if enabled for LogReg train
        if params["use_smote"]:
            logging.info("Applying SMOTE to train for LogReg")
            smote = SMOTE(random_state=params["random_state"])
            X_tr_scaled_res, y_tr_res = smote.fit_resample(X_tr_scaled, y_tr)
        else:
            X_tr_scaled_res, y_tr_res = X_tr_scaled, y_tr

        # Logistic regression (fit on full train, potentially oversampled)
        logreg = LogisticRegression(
            max_iter=1000, C=params["logreg_c"], penalty='elasticnet', l1_ratio=0.3, solver='saga', random_state=params["random_state"]
        ).fit(X_tr_scaled_res, y_tr_res)

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
            threshold=params["thresh_req_recall"],  # Updated to use param (e.g., for post-processing thresholds if needed)
        )
        return bst, metrics, artifacts
    except Exception as e:
        logging.error("Error during training/evaluation: %s", e)
        raise


def save_artifacts(bst, artifacts, pca_map, model_dir: str):
    """Save model + preprocessing objects to GCS under model_dir/."""
    if not model_dir.startswith("gs://"):
        logging.error("model_dir must be a GCS path, got %s", model_dir)
        return

    # parse GCS URI safely
    parsed = urllib.parse.urlparse(model_dir)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/")

    # dump files to a temp dir first
    with tempfile.TemporaryDirectory() as tmp:
        joblib.dump(bst,        os.path.join(tmp, "xgb_model.joblib"))
        joblib.dump(artifacts,  os.path.join(tmp, "pipeline_artifacts.joblib"))
        joblib.dump(pca_map,    os.path.join(tmp, "pca_map.joblib"))

        # upload to GCS
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        for fname in os.listdir(tmp):
            blob = bucket.blob(os.path.join(prefix, fname).lstrip("/"))
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

    # hyper-parameters – consolidated aliases
    parser.add_argument(("--pca-n", "--pca_n"), dest="pca_n", type=int, default=128)
    parser.add_argument(("--xgb-max-depth", "--xgb_max_depth"), dest="xgb_max_depth", type=int, default=5)
    parser.add_argument(("--xgb-min-child-weight", "--xgb_min_child_weight"), dest="xgb_min_child_weight", type=int, default=2)
    parser.add_argument(("--xgb-subsample", "--xgb_subsample"), dest="xgb_subsample", type=float, default=0.9)
    parser.add_argument(("--logreg-c", "--logreg_c"), dest="logreg_c", type=float, default=0.1)
    parser.add_argument(("--blend-weight", "--blend_weight"), dest="blend_weight", type=float, default=0.6)
    parser.add_argument(("--learning-rate", "--learning_rate"), dest="learning_rate", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument(("--colsample-bytree", "--colsample_bytree"), dest="colsample_bytree", type=float, default=0.9)
    parser.add_argument(("--scale-pos-weight", "--scale_pos_weight"), dest="scale_pos_weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument(("--reg-lambda", "--reg_lambda"), dest="reg_lambda", type=float, default=1.0)
    parser.add_argument(("--use-smote", "--use_smote"), dest="use_smote", type=str, default="false")
    # TODO: Add --feature-selection flag/logic post-HPO for final model refinement

    args = parser.parse_args()

    params = dict(
        pca_n=args.pca_n,
        random_state=42,
        early_stopping_rounds=150,
        thresh_req_recall=0.5,
        winsor_quantiles=(0.01, 0.99),
        breakout_threshold=0.12,
        xgb_max_depth=args.xgb_max_depth,
        xgb_min_child_weight=args.xgb_min_child_weight,
        xgb_subsample=args.xgb_subsample,
        logreg_c=args.logreg_c,
        blend_weight=args.blend_weight,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        colsample_bytree=args.colsample_bytree,
        scale_pos_weight=args.scale_pos_weight,
        alpha=args.alpha,
        reg_lambda=args.reg_lambda,
        use_smote=(args.use_smote.lower() == 'true'),
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

    try:
        df = load_data(args.project_id, args.source_table, params["breakout_threshold"])
        X_all, y_all, train_idx, holdout_idx, f_names, pca_map, imputer = preprocess_data(df, params)
        bst, metrics, artifacts = train_and_evaluate(
            X_all, y_all, train_idx, holdout_idx, f_names, params, imputer
        )

        # Always log PR-AUC for visibility
        logging.info(f"Holdout PR-AUC: {metrics['pr_auc']:.4f}")

        # report to HPT if in HPO context
        try:
            hypertune.HyperTune().report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="pr_auc",
                metric_value=metrics["pr_auc"],
                global_step=1,
            )
        except Exception as hypertune_err:
            logging.info("Not in HPO context, skipping hypertune report: %s", hypertune_err)

        if experiment_ok:
            aiplatform.log_metrics(metrics)

        model_dir = os.environ.get("AIP_MODEL_DIR")
        if model_dir:
            save_artifacts(bst, artifacts, pca_map, model_dir)
            if experiment_ok:  # Guarded to avoid error if init failed
                aiplatform.log_params({"model_dir": model_dir})

        logging.info("Training run complete.")
    except Exception as e:
        logging.error("Training failed: %s", e)
        raise


if __name__ == "__main__":
    main()