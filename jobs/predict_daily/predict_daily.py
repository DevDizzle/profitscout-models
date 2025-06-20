#!/usr/bin/env python3
"""
ProfitScout daily prediction job (28-day rolling window, deduped)
• Loads latest model artefacts from model_ckpt
• Scores all new breakout_features rows from the last 28 days
• Appends only new results to profit_scout.predictions_daily
"""

import datetime, tempfile, joblib, os, sys
import numpy as np, pandas as pd
from numpy.linalg import norm
import xgboost as xgb
from google.cloud import bigquery, storage

# ─── Project / table names ───────────────────────────────────────
PROJECT      = "profitscout-lx6bb"
FEAT_TABLE   = f"{PROJECT}.profit_scout.breakout_features"
CKPT_TABLE   = f"{PROJECT}.profit_scout.model_ckpt"
PRED_TABLE   = f"{PROJECT}.profit_scout.predictions_daily"

# ─── Column lists (must match training) ──────────────────────────
EMB_COLS = [
    "key_financial_metrics_embedding",
    "key_discussion_points_embedding",
    "sentiment_tone_embedding",
    "short-term_outlook_embedding",
    "forward-looking_signals_embedding",
]

NUM_FEATS = [
    "adj_close_30d", "max_close_30d", "sentiment_score",
    "sma_20", "ema_50", "rsi_14", "adx_14",
    "sma_20_delta", "ema_50_delta", "rsi_14_delta", "adx_14_delta",
    "eps_surprise",
    "eps_surprise_isnull", "price_sma20_ratio", "ema50_sma20_ratio",
    "rsi_centered", "adx_log1p", "sent_rsi",
]

SCALAR_COLS = [f"{c}_{s}" for c in EMB_COLS for s in ["norm", "mean", "std"]]
SIM_COLS    = ["cos_fin_disc", "cos_fin_tone", "cos_disc_short", "cos_short_fwd"]

# ─── GCP clients ─────────────────────────────────────────────────
bq  = bigquery.Client()
gcs = storage.Client()

print("🔹 [1/7] Locating latest model artefacts...")
# ─── 1. locate latest model artefacts ────────────────────────────
row = (
    bq.query(
        f"SELECT gcs_path, model_version "
        f"FROM `{CKPT_TABLE}` ORDER BY model_version DESC LIMIT 1"
    )
    .to_dataframe()
    .iloc[0]
)
gcs_path, model_ver = row["gcs_path"], row["model_version"]
bucket_name, prefix = gcs_path.replace("gs://", "").split("/", 1)
bucket = gcs.bucket(bucket_name)

tmp = tempfile.mkdtemp()
for fname in ("xgb_model.json", "blend_meta.joblib"):
    bucket.blob(f"{prefix}{fname}").download_to_filename(f"{tmp}/{fname}")

bst   = xgb.Booster()
bst.load_model(f"{tmp}/xgb_model.json")
meta  = joblib.load(f"{tmp}/blend_meta.joblib")
imp, scaler, logreg, iso, pca_models = (
    meta["imputer"],
    meta["scaler"],
    meta["logreg"],
    meta["calibrator"],
    meta["pca_models"],    # list of 5 PCA objects (saved during training)
)
print("✅ Model artefacts loaded.")

print("🔹 [2/7] Fetching new feature rows from past 28 days...")
# ─── 2. fetch last 28 days’ new feature rows ──────────────────────
today = datetime.date.today()
start_date = today - datetime.timedelta(days=27)
df = bq.query(
    f"SELECT * FROM `{FEAT_TABLE}` "
    f"WHERE earnings_call_date BETWEEN DATE('{start_date}') AND DATE('{today}')"
).to_dataframe()

if df.empty:
    print("No new earnings-call rows in the last 28 days.")
    sys.exit(0)
print(f"✅ Loaded {len(df)} feature rows from past 28 days.")

# ─── Deduplicate: only predict on rows not already in predictions_daily ──
existing = bq.query(
    f"SELECT ticker, earnings_call_date FROM `{PRED_TABLE}` "
    f"WHERE earnings_call_date BETWEEN DATE('{start_date}') AND DATE('{today}')"
).to_dataframe()
existing_set = set(zip(existing.ticker, pd.to_datetime(existing.earnings_call_date).dt.date))

# Mask for not already predicted
mask = [
    (row.ticker, row.earnings_call_date if isinstance(row.earnings_call_date, datetime.date)
     else pd.to_datetime(row.earnings_call_date).date()) not in existing_set
    for row in df.itertuples()
]
df = df[mask]

if df.empty:
    print("No new predictions to make (all already scored).")
    sys.exit(0)
print(f"✅ {len(df)} feature rows to score (not already predicted).")

print("🔹 [3/7] Rebuilding engineered features...")
# ─── 3. rebuild engineered features (same logic as training) ────
df["eps_surprise_isnull"] = df["eps_surprise"].isna().astype(int)
df["price_sma20_ratio"]   = df["adj_close_30d"] / df["sma_20"]
df["ema50_sma20_ratio"]   = df["ema_50"] / df["sma_20"]
df["rsi_centered"]        = df["rsi_14"] - 50
df["adx_log1p"]           = np.log1p(df["adx_14"])
df["sent_rsi"]            = df["sentiment_score"] * df["rsi_centered"]

# embedding scalar metrics
for col in EMB_COLS:
    A = np.vstack(df[col])
    df[f"{col}_norm"] = norm(A, 1)
    df[f"{col}_mean"] = A.mean(1)
    df[f"{col}_std"]  = A.std(1)

# cosine similarities
E = [np.vstack(df[c]) for c in EMB_COLS]
df["cos_fin_disc"]   = (E[0] * E[1]).sum(1) / np.maximum(norm(E[0], 1) * norm(E[1], 1), 1)
df["cos_fin_tone"]   = (E[0] * E[2]).sum(1) / np.maximum(norm(E[0], 1) * norm(E[2], 1), 1)
df["cos_disc_short"] = (E[1] * E[3]).sum(1) / np.maximum(norm(E[1], 1) * norm(E[3], 1), 1)
df["cos_short_fwd"]  = (E[3] * E[4]).sum(1) / np.maximum(norm(E[3], 1) * norm(E[4], 1), 1)
print("✅ Engineered features rebuilt.")

print("🔹 [4/7] Building model feature matrix...")
# ─── 4. build model feature matrix ──────────────────────────────
# PCA on embeddings using saved components
pcs_list, pca_cols = [], []
for col, pca in zip(EMB_COLS, pca_models):
    A   = np.vstack(df[col])
    pcs = pca.transform(A)
    pcs_list.append(pcs)
    pca_cols += [f"{col}_pc{i}" for i in range(pcs.shape[1])]

X_emb = np.hstack(pcs_list)
X_num = df[NUM_FEATS + SCALAR_COLS + SIM_COLS].values
X_all = np.hstack([X_emb, X_num]).astype(np.float32)
print(f"✅ Feature matrix built. Shape: {X_all.shape}")

print("🔹 [5/7] Predicting probabilities...")
# ─── 5. predict probabilities ───────────────────────────────────
xgb_raw   = bst.predict(xgb.DMatrix(X_all))
log_raw   = logreg.predict_proba(scaler.transform(imp.transform(X_all)))[:, 1]
blend_raw = 0.5 * xgb_raw + 0.5 * log_raw
prob      = iso.transform(blend_raw)
print("✅ Probabilities predicted.")

print("🔹 [6/7] Appending to predictions table...")
# ─── 6. append to predictions table ─────────────────────────────
out = pd.DataFrame({
    "insert_time"        : datetime.datetime.utcnow().isoformat(timespec="seconds"),
    "model_version"      : model_ver,
    "ticker"             : df["ticker"],
    "earnings_call_date" : df["earnings_call_date"],
    "probability"        : prob.astype(float),
})
errors = bq.insert_rows_json(PRED_TABLE, out.to_dict("records"))
if errors:
    raise RuntimeError(f"BigQuery insert errors: {errors}")

print(f"✅ Inserted {len(out)} prediction rows for window {start_date} to {today} using model {model_ver}")
print("🏁 [7/7] Daily prediction job complete.")
