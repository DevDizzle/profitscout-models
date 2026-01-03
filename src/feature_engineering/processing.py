# ───────────────────────── imports ─────────────────────────
import os
import logging
import pandas as pd
import pandas_ta as ta
import numpy as np
from google.cloud import bigquery

# ───────────────────────── init ────────────────────────────
PROJECT_ID = os.environ.get("PROJECT_ID")
bq_client = bigquery.Client()
logging.basicConfig(level=logging.INFO)

# ─────────────────────── helpers ───────────────────────────

def get_price_history(ticker: str, target_date: str, days: int = 300) -> pd.DataFrame:
    try:
        table = os.environ.get("PRICE_TABLE", "profitscout-lx6bb.profit_scout.price_data")
        q = (
            f"SELECT date, open, high, low, adj_close, volume "
            f"FROM `{table}` "
            f"WHERE ticker = @t AND date <= @d "
            f"ORDER BY date DESC LIMIT @limit"
        )
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("t", "STRING", ticker),
                bigquery.ScalarQueryParameter("d", "DATE", target_date),
                bigquery.ScalarQueryParameter("limit", "INTEGER", days),
            ]
        )
        df = bq_client.query(q, job_config=job_config).to_dataframe()
        
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("date").set_index("date")
        df.rename(columns={
            "open": "Open", "high": "High", "low": "Low", "adj_close": "Close", "volume": "Volume"
        }, inplace=True)
        
        return df
    except Exception as e:
        logging.error("get_price_history failed: %s", e)
        return pd.DataFrame()


def generate_technical_features(df: pd.DataFrame) -> dict:
    if len(df) < 50:
        return {}

    data = df.copy()

    # --- Trend ---
    data.ta.sma(length=10, append=True)
    data.ta.sma(length=20, append=True)
    data.ta.sma(length=50, append=True)
    data.ta.sma(length=200, append=True)
    data.ta.ema(length=10, append=True)
    data.ta.ema(length=50, append=True)
    data.ta.adx(length=14, append=True)
    data.ta.macd(fast=12, slow=26, signal=9, append=True)

    # --- Momentum ---
    data.ta.rsi(length=14, append=True)
    data.ta.rsi(length=5, append=True)
    data.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    data.ta.roc(length=1, append=True)
    data.ta.roc(length=3, append=True)
    data.ta.roc(length=5, append=True)

    # --- Volatility ---
    data.ta.atr(length=14, append=True)
    data.ta.atr(length=3, append=True)
    if "ATRr_3" in data.columns and "ATRr_14" in data.columns:
         data["atr_ratio"] = data["ATRr_3"] / data["ATRr_14"]
    else:
         data["atr_ratio"] = 1.0

    data.ta.bbands(length=20, std=2, append=True)

    # --- Volume ---
    data.ta.obv(append=True)
    vol_sma = data["Volume"].rolling(window=20).mean()
    data["RVOL_20"] = data["Volume"] / vol_sma
    data["volume_delta"] = data["Volume"].pct_change()

    # --- Structure ---
    if "SMA_20" in data.columns:
        data["price_vs_sma20"] = data["Close"] / data["SMA_20"]
    if "SMA_50" in data.columns:
        data["price_vs_sma50"] = data["Close"] / data["SMA_50"]
    if "SMA_200" in data.columns:
        data["price_vs_sma200"] = data["Close"] / data["SMA_200"]
    
    data["dist_from_high"] = (data["High"] - data["Close"]) / data["Close"]
    data["dist_from_low"]  = (data["Close"] - data["Low"]) / data["Close"]
    
    denom = (data["High"] - data["Low"])
    data["close_loc"] = (data["Close"] - data["Low"]) / denom
    data.loc[denom == 0, "close_loc"] = 0.5

    # --- Select Last Row ---
    last_row = data.iloc[-1]
    features = {}
    
    target_cols = [
        "SMA_10", "SMA_20", "SMA_50", "SMA_200",
        "EMA_10", "EMA_50",
        "ADX_14", 
        "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
        "RSI_14", "RSI_5",
        "STOCHk_14_3_3", "STOCHd_14_3_3",
        "ROC_1", "ROC_3", "ROC_5",
        "ATRr_14", "atr_ratio",
        "BBL_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0",
        "OBV", "RVOL_20", "volume_delta",
        "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
        "dist_from_high", "dist_from_low", "close_loc"
    ]

    for col in target_cols:
        val = last_row.get(col)
        if pd.isna(val):
            features[col.lower()] = None
        else:
            features[col.lower()] = float(val)

    features["close_price"] = float(last_row["Close"])
    features["volume"] = float(last_row["Volume"])

    return features

# ─────────────────── feature orchestrator ──────────────────
def create_features(message: dict) -> dict | None:
    required = {"ticker", "date"}
    if not required.issubset(message) or any(message[k] is None for k in required):
        return None

    ticker = message["ticker"]
    target_date = message["date"]

    df = get_price_history(ticker, target_date)
    if df.empty:
        return None

    feats = generate_technical_features(df)
    if not feats:
        return None

    feats["ticker"] = ticker
    feats["date"] = target_date

    return feats