#!/usr/bin/env python3
"""
ProfitScout weekly retraining job
• Trains on breakout_features up to --train-end (inclusive)
• Saves artefacts to the --out GCS folder
• Logs metrics + GCS path into profit_scout.model_ckpt
"""

import os, gc, json, joblib, argparse, datetime
import numpy as np, pandas as pd
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_recall_curve, auc, precision_score, recall_score, confusion_matrix
)
import xgboost as xgb
from google.cloud import bigquery, storage

# Optional: GPU check
import subprocess
try:
    print("📡 Checking GPU availability via nvidia-smi …")
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print("nvidia-smi failed:", e)

# ─── Cmd-line args ───────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--train-end", required=False, help="YYYY-MM-DD (inclusive)")
ap.add_argument("--out",       required=True, help="gs://bucket/folder")
args = ap.parse_args()

# Default to today if not provided
if args.train_end:
    TRAIN_END = datetime.date.fromisoformat(args.train_end)
else:
    TRAIN_END = datetime.date.today()

OUT_GCS_URI = os.path.join(args.out.rstrip("/"), TRAIN_END.isoformat())

# ─── Config ──────────────────────────────────────────────────────
PROJECT    = "profitscout-lx6bb"
SRC_TABLE  = f"{PROJECT}.profit_scout.breakout_features"
CKPT_TABLE = f"{PROJECT}.profit_scout.model_ckpt"

EMB_COLS = [
    "key_financial_metrics_embedding",
    "key_discussion_points_embedding",
    "sentiment_tone_embedding",
    "short-term_outlook_embedding",
    "forward-looking_signals_embedding",
]
NUM_COLS = [
    "adj_close_30d", "max_close_30d", "sentiment_score",
    "sma_20", "ema_50", "rsi_14", "adx_14",
    "sma_20_delta", "ema_50_delta", "rsi_14_delta", "adx_14_delta",
    "eps_surprise",
]
PCA_N, RANDOM, EARLY_STOP = 256, 42, 100
EMB_DIM = 768
BEST_CFG = (7, 5, 0.8)

bq  = bigquery.Client()
gcs = storage.Client()

# ─── 1) Load & clean data ────────────────────────────────────────
print(f"\n📥 Loading training data from BigQuery up to {TRAIN_END} …")
qry = f"""
SELECT *
FROM `{SRC_TABLE}`
WHERE earnings_call_date <= DATE('{TRAIN_END}')
"""
df = bq.query(qry).to_dataframe()
print("✅ Rows loaded:", len(df))

df["eps_missing"] = df["eps_missing"].fillna(False).astype(int)
df["industry_sector"] = (
    df["sector"].fillna("UNK_SEC") + "_" + df["industry"].fillna("UNK_IND")
)
def winsorise(a, q=(1, 99)):
    lo, hi = np.nanpercentile(a, q); return a.clip(lo, hi)
for col in NUM_COLS:
    df[col] = winsorise(df[col])

df["eps_surprise_isnull"] = df["eps_surprise"].isna().astype(int)
sec_mean = df.groupby("industry_sector")["eps_surprise"].transform("mean")
df["eps_surprise"] = df["eps_surprise"].fillna(sec_mean).fillna(0.0)

df["earnings_call_date"] = pd.to_datetime(df["earnings_call_date"])
df = df.sort_values("earnings_call_date").dropna(subset=EMB_COLS)
for c in EMB_COLS:
    df = df[df[c].apply(lambda v: isinstance(v, (list, np.ndarray)) and len(v) == EMB_DIM)]

# ─── [NEW] Filter: Only train on records with target present ─────
df = df.dropna(subset=["adj_close_30d", "max_close_30d"])

print("✅ Rows after embedding and target validation:", len(df))

# ─── 2) Labels & features ────────────────────────────────────────
print("🔧 Engineering features …")
df["future_return"] = df["max_close_30d"] / df["adj_close_30d"] - 1
df["breakout"] = (df["future_return"] >= 0.12).astype(int)

df["price_sma20_ratio"] = df["adj_close_30d"] / df["sma_20"]
df["ema50_sma20_ratio"] = df["ema_50"] / df["sma_20"]
df["rsi_centered"]      = df["rsi_14"] - 50
df["adx_log1p"]         = np.log1p(df["adx_14"])
df["sent_rsi"]          = df["sentiment_score"] * df["rsi_centered"]

NUM_FEATS = NUM_COLS + [
    "eps_surprise_isnull", "price_sma20_ratio", "ema50_sma20_ratio",
    "rsi_centered", "adx_log1p", "sent_rsi"
]

# ─── 3) PCA on embeddings ────────────────────────────────────────
print("🔍 Running PCA + similarity features …")
pca_list, pca_cols, scalar_cols = [], [], []
for col in EMB_COLS:
    A = np.vstack(df[col])
    pca = PCA(n_components=PCA_N, random_state=RANDOM).fit(A)
    pcs = pca.transform(A)
    pca_list.append(pcs)
    pca_cols += [f"{col}_pc{i}" for i in range(PCA_N)]
    df[f"{col}_norm"] = norm(A, 1)
    df[f"{col}_mean"] = A.mean(1)
    df[f"{col}_std"]  = A.std(1)
    scalar_cols += [f"{col}_{s}" for s in ["norm", "mean", "std"]]
    gc.collect()

def safe_cosine(u, v):
    num = (u * v).sum(1); den = norm(u, 1) * norm(v, 1)
    return num / np.where(den == 0, 1, den)

E = [np.vstack(df[c]) for c in EMB_COLS]
df["cos_fin_disc"]   = safe_cosine(E[0], E[1])
df["cos_fin_tone"]   = safe_cosine(E[0], E[2])
df["cos_disc_short"] = safe_cosine(E[1], E[3])
df["cos_short_fwd"]  = safe_cosine(E[3], E[4])
sim_cols = ["cos_fin_disc", "cos_fin_tone", "cos_disc_short", "cos_short_fwd"]
df[sim_cols] = df[sim_cols].fillna(0.0)

X_emb = np.hstack(pca_list)
X_num = df[NUM_FEATS + scalar_cols + sim_cols].values
feature_names = pca_cols + NUM_FEATS + scalar_cols + sim_cols
X_all = np.hstack([X_emb, X_num]).astype(np.float32)
y_all = df["breakout"].values
dates = df["earnings_call_date"].values

# ─── 4) Train/test split ─────────────────────────────────────────
cut = np.quantile(dates, 0.8)
train = dates < cut
X_tr, y_tr = X_all[train], y_all[train]
X_hd, y_hd = X_all[~train], y_all[~train]

# ─── 5) XGBoost training ─────────────────────────────────────────
print("🚀 Training XGBoost on GPU …")
dtr = xgb.DMatrix(X_tr, y_tr, feature_names=feature_names)
dhd = xgb.DMatrix(X_hd, y_hd, feature_names=feature_names)

md, mcw, ss = BEST_CFG
params = dict(
    objective="binary:logistic", eval_metric="aucpr",
    learning_rate=0.03, max_depth=md, min_child_weight=mcw,
    subsample=ss, colsample_bytree=0.9,
    scale_pos_weight=(1 - y_tr.mean()) / y_tr.mean(),
    n_jobs=-1, random_state=RANDOM,
    tree_method="gpu_hist", gpu_id=0
)
bst = xgb.train(
    params, dtr, 2000, evals=[(dhd, "hold")],
    early_stopping_rounds=EARLY_STOP, verbose_eval=False
)

# ─── 6) Logistic + calibration ───────────────────────────────────
print("⚖️  Calibrating blended classifier …")
imp    = SimpleImputer(strategy="median").fit(X_tr)
scaler = StandardScaler().fit(imp.transform(X_tr))
logreg = LogisticRegression(max_iter=500, C=0.1, solver="lbfgs")
logreg.fit(scaler.transform(imp.transform(X_tr)), y_tr)

xgb_raw   = bst.predict(dhd, iteration_range=(0, bst.best_iteration + 1))
log_raw   = logreg.predict_proba(scaler.transform(imp.transform(X_hd)))[:, 1]
blend_raw = 0.5 * xgb_raw + 0.5 * log_raw

iso = IsotonicRegression(out_of_bounds="clip").fit(blend_raw, y_hd)
blend_cal = iso.transform(blend_raw)

# ─── 7) Metrics ──────────────────────────────────────────────────
precisions, recalls, _ = precision_recall_curve(y_hd, blend_cal)
pr_auc_val = auc(recalls, precisions)

records = []
for thr in np.arange(0.50, 0.96, 0.05):
    mask = blend_cal >= thr
    if not mask.any(): continue
    prec = precision_score(y_hd, mask)
    rec  = recall_score(y_hd, mask)
    cov  = mask.mean()
    pnl  = df.loc[~train].loc[mask, "max_close_30d"].values / df.loc[~train].loc[mask, "adj_close_30d"].values - 1
    records.append((thr, prec, rec, cov, pnl.mean(), pnl.min()))
pd.DataFrame(
    records, columns=["thr", "precision", "recall", "coverage", "avg_pnl", "min_pnl"]
).to_csv("threshold_report.csv", index=False)

# ─── 8) Save to GCS & BQ ─────────────────────────────────────────
print("💾 Uploading model artefacts to GCS …")
joblib.dump(
    {"imputer": imp, "scaler": scaler, "logreg": logreg,
     "calibrator": iso, "threshold": 0.80},
    "blend_meta.joblib"
)
bst.save_model("xgb_model.json")

bucket_name, *path_parts = OUT_GCS_URI.replace("gs://", "").split("/", 1)
bucket  = gcs.bucket(bucket_name)
prefix  = path_parts[0] if path_parts else ""
for fname in ["xgb_model.json", "blend_meta.joblib", "threshold_report.csv"]:
    bucket.blob(f"{prefix}/{fname}").upload_from_filename(fname)
print("✅ Upload complete:", OUT_GCS_URI)

mask80 = blend_cal >= 0.80
row = {
    "model_version"   : OUT_GCS_URI.rsplit("/", 1)[-1],
    "trained_on_thru" : TRAIN_END.isoformat(),
    "pr_auc"          : float(pr_auc_val),
    "precision_0p80"  : float(precision_score(y_hd, mask80)) if mask80.any() else None,
    "coverage_0p80"   : float(mask80.mean()),
    "gcs_path"        : OUT_GCS_URI + "/"
}
errors = bq.insert_rows_json(CKPT_TABLE, [row])
if errors:
    raise RuntimeError(f"BQ insert failed: {errors}")

print("🎉 ✅ Weekly training job complete")
