# FILE: merger/main.py

import os
import logging
from google.cloud import bigquery

# --- Environment Variables ---
PROJECT_ID = os.environ.get('PROJECT_ID')
STAGING_TABLE = os.environ.get('STAGING_TABLE')
FINAL_TABLE = os.environ.get('FINAL_TABLE')

# --- Clients ---
bq_client = bigquery.Client()
logging.basicConfig(level=logging.INFO)

# --- CONFIGURATION FOR DATA VALIDATION ---
NOT_NULL_COLUMNS = [
    "ticker", "quarter_end_date", "sentiment_score", "earnings_call_date",
    "key_financial_metrics_embedding", "key_discussion_points_embedding",
    "sentiment_tone_embedding", "short_term_outlook_embedding", "forward_looking_signals_embedding",
    "sector", "industry", "eps_surprise", "eps_missing", "adj_close_on_call_date",
    "sma_20", "ema_50", "rsi_14", "adx_14", "sma_20_delta", "ema_50_delta",
    "rsi_14_delta", "adx_14_delta"
]

def merge_staging_to_final(event, context):
    """
    Merges de-duplicated and validated data from a staging table into a final
    production table, then truncates the staging table.
    """
    logging.info(f"Starting merge from {STAGING_TABLE} to {FINAL_TABLE}")

    validation_filter = " AND ".join(f"{col} IS NOT NULL" for col in NOT_NULL_COLUMNS)

    merge_query = f"""
    MERGE `{PROJECT_ID}.{FINAL_TABLE}` T
    USING (
        SELECT *
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER(PARTITION BY ticker, quarter_end_date ORDER BY sentiment_score DESC) as rn
            FROM `{PROJECT_ID}.{STAGING_TABLE}`
        )
        WHERE rn = 1
    ) S
    ON T.ticker = S.ticker AND T.quarter_end_date = S.quarter_end_date
    WHEN MATCHED AND (
        -- This condition prevents unnecessary updates if the row data is identical
        T.sentiment_score != S.sentiment_score OR T.earnings_call_date != S.earnings_call_date
    ) THEN
      UPDATE SET
        sentiment_score = S.sentiment_score, earnings_call_date = S.earnings_call_date,
        key_financial_metrics_embedding = S.key_financial_metrics_embedding,
        key_discussion_points_embedding = S.key_discussion_points_embedding,
        sentiment_tone_embedding = S.sentiment_tone_embedding,
        short_term_outlook_embedding = S.short_term_outlook_embedding,
        forward_looking_signals_embedding = S.forward_looking_signals_embedding,
        sector = S.sector, industry = S.industry, eps_surprise = S.eps_surprise,
        eps_missing = S.eps_missing, adj_close_on_call_date = S.adj_close_on_call_date,
        sma_20 = S.sma_20, ema_50 = S.ema_50, rsi_14 = S.rsi_14, adx_14 = S.adx_14,
        sma_20_delta = S.sma_20_delta, ema_50_delta = S.ema_50_delta,
        rsi_14_delta = S.rsi_14_delta, adx_14_delta = S.adx_14_delta, max_close_30d = S.max_close_30d
    WHEN NOT MATCHED BY TARGET AND ({validation_filter}) THEN
      INSERT (
        ticker, quarter_end_date, sentiment_score, earnings_call_date,
        key_financial_metrics_embedding, key_discussion_points_embedding,
        sentiment_tone_embedding, short_term_outlook_embedding, forward_looking_signals_embedding,
        sector, industry, eps_surprise, eps_missing, adj_close_on_call_date,
        sma_20, ema_50, rsi_14, adx_14, sma_20_delta, ema_50_delta, rsi_14_delta,
        adx_14_delta, max_close_30d
      )
      VALUES (
        S.ticker, S.quarter_end_date, S.sentiment_score, S.earnings_call_date,
        S.key_financial_metrics_embedding, S.key_discussion_points_embedding,
        S.sentiment_tone_embedding, S.short_term_outlook_embedding, S.forward_looking_signals_embedding,
        S.sector, S.industry, S.eps_surprise, S.eps_missing, S.adj_close_on_call_date,
        S.sma_20, S.ema_50, S.rsi_14, S.adx_14, S.sma_20_delta, S.ema_50_delta, S.rsi_14_delta,
        S.adx_14_delta, S.max_close_30d
      )
    """
    
    truncate_query = f"TRUNCATE TABLE `{PROJECT_ID}.{STAGING_TABLE}`"

    try:
        merge_job = bq_client.query(merge_query)
        merge_job.result()
        logging.info(f"Merge successful. {merge_job.num_dml_affected_rows} rows were inserted or updated.")

        truncate_job = bq_client.query(truncate_query)
        truncate_job.result()
        logging.info(f"Staging table {STAGING_TABLE} truncated.")

        return "Merge completed successfully", 200
    except Exception as e:
        logging.error(f"Merge process failed: {e}", exc_info=True)
        return "Merge process failed", 500