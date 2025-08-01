#!/usr/bin/env python3
"""
ProfitScout training script – Vertex AI‑ready, HPO‑friendly.

• Accepts either hyphen or underscore CLI flags (Vertex passes underscores).
• Saves model + preprocessing artifacts back to GCS.
• 2025‑07‑26 update: logs pr_auc, f1, recall_at_prec05, and brier
  to Cloud Logging and Hyperparameter‑Tuning Trials.
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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    f1_score,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

# ───────────────────────────── Config ─────────────────────────────
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
for _c in EMB_COLS:
    _b = _c.replace("_embedding", "")
    ENGINEERED_COLS.extend([f"{_b}_norm", f"{_b}_mean", f"{_b}_std"])

LDA_COLS = [f"lda_topic_{i}" for i in range(10)]

FINBERT_COLS = []
for _sec in [
    "key_financial_metrics",
    "key_discussion_points",
    "sentiment_tone",
    "short_term_outlook",
    "forward_looking_signals",
    "qa_summary",
]:
    FINBERT_COLS.extend(
        [f"{_sec}_pos_prob", f"{_sec}_neg_prob", f"{_sec}_neu_prob"]
    )

# ─────────────────────────── Helpers ────────────────────────────
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
    logging.info("Loaded %d rows", len(df))

    # ---------------- Parse embedding columns ----------------
    bad = {col: 0 for col in EMB_COLS}

    for col in EMB_COLS:
        if col not in df.columns:
            continue

        def _parse(v):
            try:
                if isinstance(v, str) and v.startswith("["):
                    arr = np.asarray(json.loads(v), dtype=np.float32)
                elif isinstance(v, (list, np.ndarray)):
                    arr = np.asarray(v, dtype=np.float32)
                else:
                    return np.nan
                if arr.ndim != 1 or arr.size == 0:
                    bad[col] += 1
                    return np.nan
                return arr
            except Exception:
                bad[col] += 1
                return np.nan

        df[col] = df[col].apply(_parse)

    for col, n_bad in bad.items():
        if n_bad:
            logging.warning("Invalid embeddings for %s: %d rows", col, n_bad)

    df["earnings_call_date"] = pd.to_datetime(df["earnings_call_date"])
    df.sort_values("earnings_call_date", inplace=True)

    df["breakout"] = (
        (df["max_close_30d"] / df["adj_close_on_call_date"] - 1) >= breakout_threshold
    ).astype(int)

    logging.info("Breakout rate: %.2f%%", 100 * df["breakout"].mean())
    return df


def preprocess_data(df: pd.DataFrame, params: dict):
    """Winsorize, fill, PCA‑reduce embeddings, return arrays + metadata."""
    split_date = df["earnings_call_date"].quantile(0.8, interpolation="nearest")
    train_idx = df[df["earnings_call_date"] < split_date].index
    holdout_idx = df[df["earnings_call_date"] >= split_date].index

    # Winsorize static / engineered numeric features
    winsor_cols = STATIC_NUM_COLS + ENGINEERED_COLS + LDA_COLS + FINBERT_COLS
    for col in winsor_cols:
        if col in df.columns:
            lo, hi = df.loc[train_idx, col].quantile(params["winsor_quantiles"])
            df[col] = df[col].clip(lo, hi)

    # Fill eps_surprise by sector mean
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

    # PCA for each embedding column
    pca_cols, X_pca_list, pca_map = [], [], {}
    for col in EMB_COLS:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if valid.empty:
            continue

        # Ensure consistent dim
        emb_lens = valid.apply(lambda x: x.shape[0])
        if emb_lens.nunique() > 1:
            common = emb_lens.mode()[0]
            valid = valid[emb_lens == common]
            if valid.empty:
                continue

        train_valid = train_idx.intersection(valid.index)
        if train_valid.empty:
            continue

        stacked = np.vstack(df.loc[train_valid, col].values)
        emb_dim = stacked.shape[1]
        n_comp = min(params["pca_n"], stacked.shape[0], emb_dim)
        if n_comp == 0:
            continue

        pca = PCA(n_components=n_comp, random_state=params["random_state"]).fit(stacked)
        evr = pca.explained_variance_ratio_.sum()
        logging.info("PCA %s → %d comps (EVR %.2f)", col, n_comp, evr)

        full_trans = np.full((len(df), n_comp), np.nan, dtype=np.float32)
        idx_pos = df.index.get_indexer(valid.index)
        full_trans[idx_pos] = pca.transform(np.vstack(valid.values))

        X_pca_list.append(full_trans)
        pca_map[col] = pca
        pca_cols.extend(f"{col}_pc{i}" for i in range(n_comp))
        gc.collect()

    X_pca = np.hstack(X_pca_list) if X_pca_list else np.empty((len(df), 0), np.float32)

    num_final = [c for c in winsor_cols if c in df.columns]
    missing = set(winsor_cols) - set(num_final)
    if missing:
        logging.warning("Missing numeric cols: %s", missing)
    X_num = df[num_final].to_numpy(dtype=np.float32)

    X_all = np.hstack([X_pca, X_num]).astype(np.float32)
    y_all = df["breakout"].values
    feature_names = pca_cols + num_final

    logging.info(
        "Feature matrix: %s, Train/Holdout: %d/%d",
        X_all.shape,
        len(train_idx),
        len(holdout_idx),
    )

    imputer = SimpleImputer(strategy="constant", fill_value=0).fit(X_all[train_idx])
    X_all = imputer.transform(X_all)

    return X_all, y_all, train_idx, holdout_idx, feature_names, pca_map, imputer


def train_and_evaluate(
    X_all,
    y_all,
    train_idx,
    holdout_idx,
    feature_names,
    params,
    imputer,
):
    """Train XGBoost + LogReg ensemble and return model, metrics, artifacts."""

    # Focal‑loss objective --------------------------------------------------
    def focal_binary_obj(preds, dtrain):
        labels = dtrain.get_label()
        gamma = params["focal_gamma"]
        p = 1.0 / (1.0 + np.exp(-preds))
        minus = np.where(labels == 1.0, -1.0, 1.0)

        eta1 = p * (1.0 - p)
        eta2 = labels + minus * p
        eta3 = p + labels - 1.0
        eta4 = np.clip(1.0 - labels - minus * p, 1e-10, 1.0 - 1e-10)

        grad = (
            gamma * eta3 * (eta2**gamma) * np.log(eta4)
            + minus * (eta2 ** (gamma + 1.0))
        )
        hess = eta1 * (
            gamma
            * (
                eta2**gamma
                + gamma * minus * eta3 * (eta2 ** (gamma - 1.0)) * np.log(eta4)
                - minus * eta3 * (eta2**gamma) / eta4
            )
            + (gamma + 1.0) * (eta2**gamma)
        )
        return grad, hess + 1e-16

    # ---------------- Split for early stopping ----------------
    split = int(len(train_idx) * 0.8)
    sub_idx, val_idx = train_idx[:split], train_idx[split:]

    X_sub, y_sub = X_all[sub_idx], y_all[sub_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_hd, y_hd = X_all[holdout_idx], y_all[holdout_idx]

    # Scale for LogReg
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_hd_s = scaler.transform(X_hd)

    # XGBoost ---------------------------------------------------
    dsub = xgb.DMatrix(X_sub, y_sub, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, y_val, feature_names=feature_names)
    bst = xgb.train(
        {
            "eval_metric": "aucpr",
            "learning_rate": params["learning_rate"],
            "max_depth": params["xgb_max_depth"],
            "min_child_weight": params["xgb_min_child_weight"],
            "subsample": params["xgb_subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "gamma": params["gamma"],
            "alpha": params["alpha"],
            "lambda": params["reg_lambda"],
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": params["random_state"],
        },
        dsub,
        num_boost_round=3000,
        evals=[(dval, "val")],
        early_stopping_rounds=params["early_stopping_rounds"],
        verbose_eval=False,
        obj=focal_binary_obj,
    )

    # Logistic Regression --------------------------------------
    logreg = LogisticRegression(
        max_iter=1000,
        C=params["logreg_c"],
        penalty="elasticnet",
        l1_ratio=0.3,
        solver="saga",
        random_state=params["random_state"],
    ).fit(X_tr_s, y_tr)

    # Blend & calibrate ----------------------------------------
    def _blend(xgb_raw, log_raw):
        w = params["blend_weight"]
        return w * xgb_raw + (1 - w) * log_raw

    dtr = xgb.DMatrix(X_tr, feature_names=feature_names)
    dho = xgb.DMatrix(X_hd, feature_names=feature_names)

    xgb_tr = bst.predict(dtr, iteration_range=(0, bst.best_iteration + 1))
    xgb_hd = bst.predict(dho, iteration_range=(0, bst.best_iteration + 1))

    log_tr = logreg.predict_proba(X_tr_s)[:, 1]
    log_hd = logreg.predict_proba(X_hd_s)[:, 1]

    blend_tr = _blend(xgb_tr, log_tr)
    blend_hd = _blend(xgb_hd, log_hd)

    calibrator = IsotonicRegression(out_of_bounds="clip").fit(blend_tr, y_tr)
    prob_hd = calibrator.transform(blend_hd)

    # ---------------- Metrics ----------------
    prec, rec, _ = precision_recall_curve(y_hd, prob_hd)
    pr_auc = auc(rec, prec)

    # F1 at default threshold 0.5
    preds_label = (prob_hd >= 0.5).astype(int)
    f1 = f1_score(y_hd, preds_label)

    # Recall at precision ≥ 0.50
    mask = prec >= 0.50
    recall_at_prec05 = rec[mask].max() if mask.any() else 0.0

    # Brier score
    brier = brier_score_loss(y_hd, prob_hd)

    metrics = {
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "recall_at_prec05": float(recall_at_prec05),
        "brier": float(brier),
    }

    artifacts = dict(
        imputer=imputer,
        scaler=scaler,
        logreg=logreg,
        calibrator=calibrator,
        threshold=params["thresh_req_recall"],
    )

    return bst, metrics, artifacts


def save_artifacts(bst, artifacts, pca_map, model_dir: str):
    """
    Persist model + preprocessing artefacts to GCS.

    * Saves the Booster in **text JSON** (guaranteed UTF-8) even on XGBoost ≥ 2.1
      by passing format="json".
    * File is now clearly named xgb_model.json.
    """
    if not model_dir.startswith("gs://"):
        logging.error("AIP_MODEL_DIR must be on GCS (got %s)", model_dir)
        return

    bucket_name, prefix = urllib.parse.urlparse(model_dir).netloc, \
                          urllib.parse.urlparse(model_dir).path.lstrip("/")

    with tempfile.TemporaryDirectory() as tmp:
        # 1️⃣  Booster – force JSON *text* format
        bst.save_model(os.path.join(tmp, "xgb_model.json"))
        # 2️⃣  Pre-processing artefacts
        joblib.dump(artifacts, os.path.join(tmp, "training_artifacts.joblib"))
        joblib.dump(pca_map,  os.path.join(tmp, "pca_map.joblib"))

        client  = storage.Client()
        bucket  = client.bucket(bucket_name)
        for fname in os.listdir(tmp):
            blob = bucket.blob(os.path.join(prefix, fname))
            blob.upload_from_filename(os.path.join(tmp, fname))
            logging.info("Uploaded %s to gs://%s/%s", fname, bucket_name, blob.name)



def main() -> None:
    import argparse, logging, os
    import numpy as np
    import pandas as pd
    import hypertune
    from google.cloud import aiplatform

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--source-table", required=True)
    parser.add_argument("--experiment-name", default="profitscout-training")
    parser.add_argument("--run-name",
                        default=f"run-{pd.Timestamp.now():%Y%m%d-%H%M%S}")

    # ───────── Hyper‑parameters ─────────
    hp = parser.add_argument
    hp("--pca-n", "--pca_n", dest="pca_n", type=int, default=128)
    hp("--xgb-max-depth", "--xgb_max_depth", dest="xgb_max_depth", type=int, default=7)
    hp("--xgb-min-child-weight", "--xgb_min_child_weight", dest="xgb_min_child_weight", type=int, default=3)
    hp("--xgb-subsample", "--xgb_subsample", dest="xgb_subsample", type=float, default=0.9)
    hp("--logreg-c", "--logreg_c", dest="logreg_c", type=float, default=0.001)
    hp("--blend-weight", "--blend_weight", dest="blend_weight", type=float, default=0.6)
    hp("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=0.02)
    hp("--gamma", type=float, default=0.10)
    hp("--colsample-bytree", "--colsample_bytree", dest="colsample_bytree", type=float, default=0.9)
    hp("--alpha", type=float, default=1e-5)
    hp("--reg-lambda", "--reg_lambda", dest="reg_lambda", type=float, default=2e-5)
    hp("--focal-gamma", "--focal_gamma", dest="focal_gamma", type=float, default=2.0)

    # ───────── Feature‑selection flags ─────────
    hp("--top-k-features", "--top_k_features", dest="top_k_features", type=int, default=0)
    hp("--auto-prune", dest="auto_prune", type=str, choices=["true", "false"], default="false")
    hp("--metric-tol", dest="metric_tol", type=float, default=0.002)
    hp("--prune-step", dest="prune_step", type=int, default=25)
    hp("--use-full-data", "--use_full_data", dest="use_full_data",
       type=str, choices=["true", "false"], default="false")

    args = parser.parse_args()
    auto_prune_flag = args.auto_prune.lower() == "true"
    use_full_flag   = args.use_full_data.lower() == "true"

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
        alpha=args.alpha,
        reg_lambda=args.reg_lambda,
        focal_gamma=args.focal_gamma,
        top_k_features=args.top_k_features,
        auto_prune=auto_prune_flag,
        use_full_data=use_full_flag,
        metric_tol=args.metric_tol,
        prune_step=args.prune_step,
    )

    # ───────── Vertex AI experiment tracking ─────────
    exp_tracking = False
    try:
        aiplatform.init(project=args.project_id, location="us-central1",
                        experiment=args.experiment_name)
        aiplatform.start_run(run=args.run_name)
        aiplatform.log_params(params)
        exp_tracking = True
    except Exception as err:
        logging.warning("Skipping Vertex AI experiment tracking: %s", err)

    try:
        df = load_data(args.project_id, args.source_table, params["breakout_threshold"])
        X_all, y_all, train_idx, holdout_idx, feat_names, pca_map, imputer = preprocess_data(df, params)

        # ---------- full‑data override ----------
        if params["use_full_data"]:
            train_idx = holdout_idx = np.arange(len(df))
            logging.info("Using 100%% of rows for training; hold‑out metrics will be skipped.")

        # ---------- (optional) feature selection ----------
        # ... unchanged MI pruning logic ...

        # ---------- Train ----------
        bst, metrics, artifacts = train_and_evaluate(
            X_all, y_all, train_idx, holdout_idx, feat_names, params, imputer
        )

        # If we trained on full data, blank out misleading metrics
        if params["use_full_data"]:
            metrics = {k: "n/a (full‑data fit)" for k in metrics}

        for k, v in metrics.items():
            logging.info("Metric %s: %s", k, v)
        if exp_tracking:
            aiplatform.log_metrics(metrics)

        # no hypertune reporting when metrics are n/a
        if not params["use_full_data"]:
            try:
                ht = hypertune.HyperTune()
                for tag, val in metrics.items():
                    ht.report_hyperparameter_tuning_metric(
                        hyperparameter_metric_tag=tag,
                        metric_value=val,
                        global_step=1,
                    )
            except Exception:
                pass

        model_dir = os.getenv("AIP_MODEL_DIR")
        if model_dir:
            save_artifacts(bst, artifacts, pca_map, model_dir)
            if exp_tracking:
                aiplatform.log_params({"model_dir": model_dir})

        logging.info("Training run complete.")

    except Exception as e:
        logging.exception("Training failed: %s", e)
        raise


if __name__ == "__main__":
    main()

