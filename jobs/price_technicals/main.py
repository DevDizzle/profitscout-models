#!/usr/bin/env python3
"""
Job: Price Techncials Feature Engineering

Part of a scheduled workflow. Fetches price data from BigQuery, calculates
technical indicators, and upserts the results to a staging table.
"""
import logging
import os
import sys
from datetime import timedelta

import pandas as pd
import pandas_ta as ta
from dateutil.parser import parse
from google.cloud import bigquery

from bq import bq_client, table_ref, upsert_json

# Configure logging for Cloud Run Jobs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ──────────────────── CONFIG ──────────────────────────
PROJECT_ID    = os.getenv("PROJECT_ID", "profitscout-lx6bb")
DATASET       = os.getenv("BQ_DATASET", "profit_scout")
STAGING_TABLE = os.getenv("STAGING_TABLE", "call_feats_staging")
TABLE         = table_ref(DATASET, STAGING_TABLE, PROJECT_ID)
PRICE_TABLE   = table_ref(DATASET, "price_data", PROJECT_ID)
WINDOW_DAYS   = int(os.getenv("WINDOW_DAYS", 90))

# ──────────────────── GLOBALS ─────────────────────────
bq = bq_client(PROJECT_ID)

# ──────────────────── CORE LOGIC ──────────────────────
def process(msg: dict):
    """Fetches price data, calculates features, and upserts to BigQuery."""
    ticker = msg["ticker"]
    qend = msg["quarter_end"]
    call_date = parse(msg["earnings_call_date"]).date()

    logging.info(f"Processing {ticker} for call date {call_date}...")

    # 1. Fetch historical price data from our BigQuery table
    query = f"""
    SELECT
        DATE(date) AS date,
        adj_close, high, low, close, volume
    FROM `{PRICE_TABLE}`
    WHERE ticker = @ticker
      AND DATE(date) BETWEEN @start_date AND @end_date  
    ORDER BY date
    """
    params = [
        bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
        bigquery.ScalarQueryParameter("start_date", "DATE", call_date - timedelta(days=150)),
        bigquery.ScalarQueryParameter("end_date", "DATE", call_date + timedelta(days=31)),
    ]
    job_cfg = bigquery.QueryJobConfig(query_parameters=params)
    df = bq.query(query, job_cfg).to_dataframe()

    if df.empty:
        logging.warning(f"No price data found for {ticker} around {call_date}. Skipping.")
        return

    df['date'] = pd.to_datetime(df['date']).dt.date

    # 2. Calculate all technical indicators
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.adx(length=14, append=True)

    # 3. Find values on the anchor dates safely
    row_now_df = df[df['date'] <= call_date]
    row_prev_df = df[df['date'] <= call_date - timedelta(days=WINDOW_DAYS)]

    if row_now_df.empty or row_prev_df.empty:
        logging.warning(f"Not enough historical data to calculate indicators for {ticker} on {call_date}. Skipping.")
        return

    row_now_series = row_now_df.iloc[-1]
    row_prev_series = row_prev_df.iloc[-1]

    def pct_delta(now, prev):
        if pd.isna(now) or pd.isna(prev) or prev == 0:
            return None
        return (now - prev) / prev

    # 4. Calculate future price features
    window_30d = df[(df['date'] > call_date) & (df['date'] <= call_date + timedelta(days=30))]
    adj_close_30d = None
    max_close_30d = None
    if not window_30d.empty:
        future_date_target = call_date + timedelta(days=30)
        future_row = window_30d[window_30d['date'] == future_date_target]
        if not future_row.empty:
            adj_close_30d = future_row['adj_close'].iloc[0]
        max_close_30d = window_30d['adj_close'].max()

    # 5. Build the row for BigQuery
    final_row = {
        "ticker": ticker, "quarter_end_date": qend, "earnings_call_date": call_date.isoformat(),
        "adj_close_30d": adj_close_30d, "max_close_30d": max_close_30d,
        "sma_20": row_now_series.get("SMA_20"), "ema_50": row_now_series.get("EMA_50"),
        "rsi_14": row_now_series.get("RSI_14"), "adx_14": row_now_series.get("ADX_14"),
        "sma_20_delta": pct_delta(row_now_series.get("SMA_20"), row_prev_series.get("SMA_20")),
        "ema_50_delta": pct_delta(row_now_series.get("EMA_50"), row_prev_series.get("EMA_50")),
        "rsi_14_delta": pct_delta(row_now_series.get("RSI_14"), row_prev_series.get("RSI_14")),
        "adx_14_delta": pct_delta(row_now_series.get("ADX_14"), row_prev_series.get("ADX_14")),
    }

    # 6. Upsert data to the staging table
    upsert_json(TABLE, final_row, bq)
    logging.info(f"Successfully upserted technicals to staging for {ticker} {qend}")

# ──────────────────── JOB ENTRYPOINT ───────────────────
if __name__ == "__main__":
    # This block is the entrypoint for the Cloud Run Job.
    # It reads command-line arguments to get the job's parameters.
    if len(sys.argv) < 4:
        logging.error("FATAL: Invalid arguments. Usage: <script> TICKER QUARTER_END_DATE EARNINGS_CALL_DATE")
        sys.exit(1)

    payload = {
        "ticker": sys.argv[1],
        "quarter_end": sys.argv[2],
        "earnings_call_date": sys.argv[3],
    }

    logging.info(f"Starting job with payload: {payload}")
    try:
        process(payload)
        logging.info("Job completed successfully.")
    except Exception as e:
        logging.error(f"Job failed with error: {e}", exc_info=True)
        # Exit with a non-zero status code to mark the job execution as FAILED
        sys.exit(1)