"""Scheduled function to merge staging data into the final BigQuery table."""

import os
import logging
from google.cloud import bigquery

# --- Environment Variables ---
PROJECT_ID = os.environ.get('PROJECT_ID', 'profitscout-lx6bb')  # Fallback for testing
STAGING_TABLE = os.environ.get('STAGING_TABLE', 'profit_scout.staging_breakout_features')  # Assume dataset.table format
FINAL_TABLE = os.environ.get('FINAL_TABLE', 'profit_scout.breakout_features')

# --- Clients ---
bq_client = bigquery.Client()
logging.basicConfig(level=logging.INFO)

# --- CONFIGURATION FOR DATA VALIDATION ---
# Strict: Abort insert if null (core identifiers/embeddings)
STRICT_NOT_NULL_COLUMNS = [
    "ticker", "quarter_end_date", "earnings_call_date",
    "key_financial_metrics_embedding", "key_discussion_points_embedding",
    "sentiment_tone_embedding", "short_term_outlook_embedding", "forward_looking_signals_embedding",
    "qa_summary_embedding",
    "sector", "industry", "adj_close_on_call_date", "industry_sector"
]

# Soft: Warn if nulls > threshold (e.g., TA indicators, eps_surprise - acceptable sparsity)
SOFT_NOT_NULL_COLUMNS = [
    "sentiment_score", "eps_surprise", "eps_missing",
    "sma_20", "ema_50", "rsi_14", "adx_14", "sma_20_delta", "ema_50_delta",
    "rsi_14_delta", "adx_14_delta",
    # Engineered columns
    "eps_surprise_isnull", "price_sma20_ratio", "ema50_sma20_ratio", "rsi_centered",
    "sent_rsi", "adx_log1p", "key_financial_metrics_norm", "key_financial_metrics_mean",
    "key_financial_metrics_std", "key_discussion_points_norm", "key_discussion_points_mean",
    "key_discussion_points_std", "sentiment_tone_norm", "sentiment_tone_mean",
    "sentiment_tone_std", "short_term_outlook_norm", "short_term_outlook_mean",
    "short_term_outlook_std", "forward_looking_signals_norm", "forward_looking_signals_mean",
    "forward_looking_signals_std", "qa_summary_norm", "qa_summary_mean", "qa_summary_std",
    "cos_fin_disc", "cos_fin_tone", "cos_disc_short", "cos_short_fwd",
    # LDA topics
    "lda_topic_0", "lda_topic_1", "lda_topic_2", "lda_topic_3", "lda_topic_4",
    "lda_topic_5", "lda_topic_6", "lda_topic_7", "lda_topic_8", "lda_topic_9",
    # FinBERT sentiments
    "key_financial_metrics_pos_prob", "key_financial_metrics_neg_prob", "key_financial_metrics_neu_prob",
    "key_discussion_points_pos_prob", "key_discussion_points_neg_prob", "key_discussion_points_neu_prob",
    "sentiment_tone_pos_prob", "sentiment_tone_neg_prob", "sentiment_tone_neu_prob",
    "short_term_outlook_pos_prob", "short_term_outlook_neg_prob", "short_term_outlook_neu_prob",
    "forward_looking_signals_pos_prob", "forward_looking_signals_neg_prob", "forward_looking_signals_neu_prob",
    "qa_summary_pos_prob", "qa_summary_neg_prob", "qa_summary_neu_prob"
]

# Threshold for soft warnings (e.g., warn if >10% nulls)
NULL_THRESHOLD_PCT = 0.10  # Adjust as needed

def merge_staging_to_final(event, context):
    """
    Merges de-duplicated and validated data from a staging table into a final
    production table, then truncates the staging table.
    """
    staging_full = f"{PROJECT_ID}.{STAGING_TABLE}"
    final_full = f"{PROJECT_ID}.{FINAL_TABLE}"
    logging.info(f"Starting merge from {staging_full} to {final_full}")

    # Pre-merge: Log staging stats (row count, null counts)
    try:
        all_cols = STRICT_NOT_NULL_COLUMNS + SOFT_NOT_NULL_COLUMNS
        stats_query = f"""
        SELECT 
          COUNT(*) AS row_count,
          {', '.join(f'COUNTIF({col} IS NULL) AS null_{col}' for col in all_cols)}
        FROM `{staging_full}`
        """
        stats_job = bq_client.query(stats_query)
        stats = stats_job.to_dataframe().iloc[0]
        row_count = stats['row_count']
        logging.info(f"Staging stats: {row_count} rows")

        for col in STRICT_NOT_NULL_COLUMNS:
            null_count = stats[f'null_{col}']
            if null_count > 0:
                logging.error(f"Staging has {null_count} nulls in strict column {col}; aborting merge.")
                raise ValueError(f"Nulls in strict column {col}")

        for col in SOFT_NOT_NULL_COLUMNS:
            null_count = stats[f'null_{col}']
            null_pct = null_count / row_count if row_count > 0 else 0
            if null_pct > NULL_THRESHOLD_PCT:
                logging.warning(f"Staging has high nulls in {col}: {null_count} ({null_pct:.2%}) - proceeding but check data quality.")
            elif null_count > 0:
                logging.info(f"Staging has minor nulls in {col}: {null_count} ({null_pct:.2%})")

        if row_count == 0:
            logging.info("Staging is empty; skipping merge.")
            return "No data to merge", 200
    except Exception as e:
        logging.error(f"Failed to get staging stats: {e}")
        return "Pre-merge stats failed", 500

    # Quick schema check (columns match)
    staging_schema = bq_client.get_table(staging_full).schema
    final_schema = bq_client.get_table(final_full).schema
    staging_cols = {field.name for field in staging_schema}
    final_cols = {field.name for field in final_schema}
    if staging_cols != final_cols:
        logging.error(f"Schema mismatch: Staging extra: {staging_cols - final_cols}, Final extra: {final_cols - staging_cols}")
        return "Schema mismatch", 500

    validation_filter = " AND ".join(f"S.{col} IS NOT NULL" for col in STRICT_NOT_NULL_COLUMNS)

    merge_query = f"""
    MERGE `{final_full}` T
    USING (
        SELECT *
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER(PARTITION BY ticker, quarter_end_date ORDER BY earnings_call_date DESC) as rn
            FROM `{staging_full}`
        )
        WHERE rn = 1
    ) S
    ON T.ticker = S.ticker AND T.quarter_end_date = S.quarter_end_date
    WHEN MATCHED THEN
      UPDATE SET
        earnings_call_date = S.earnings_call_date,
        sector = S.sector,
        industry = S.industry,
        industry_sector = S.industry_sector,
        adj_close_on_call_date = S.adj_close_on_call_date,
        sentiment_score = S.sentiment_score,
        eps_surprise = S.eps_surprise,
        eps_missing = S.eps_missing,
        max_close_30d = S.max_close_30d,
        key_financial_metrics_embedding = S.key_financial_metrics_embedding,
        key_discussion_points_embedding = S.key_discussion_points_embedding,
        sentiment_tone_embedding = S.sentiment_tone_embedding,
        short_term_outlook_embedding = S.short_term_outlook_embedding,
        forward_looking_signals_embedding = S.forward_looking_signals_embedding,
        qa_summary_embedding = S.qa_summary_embedding,
        sma_20 = S.sma_20,
        ema_50 = S.ema_50,
        rsi_14 = S.rsi_14,
        adx_14 = S.adx_14,
        sma_20_delta = S.sma_20_delta,
        ema_50_delta = S.ema_50_delta,
        rsi_14_delta = S.rsi_14_delta,
        adx_14_delta = S.adx_14_delta,
        eps_surprise_isnull = S.eps_surprise_isnull,
        price_sma20_ratio = S.price_sma20_ratio,
        ema50_sma20_ratio = S.ema50_sma20_ratio,
        rsi_centered = S.rsi_centered,
        sent_rsi = S.sent_rsi,
        adx_log1p = S.adx_log1p,
        key_financial_metrics_norm = S.key_financial_metrics_norm,
        key_financial_metrics_mean = S.key_financial_metrics_mean,
        key_financial_metrics_std = S.key_financial_metrics_std,
        key_discussion_points_norm = S.key_discussion_points_norm,
        key_discussion_points_mean = S.key_discussion_points_mean,
        key_discussion_points_std = S.key_discussion_points_std,
        sentiment_tone_norm = S.sentiment_tone_norm,
        sentiment_tone_mean = S.sentiment_tone_mean,
        sentiment_tone_std = S.sentiment_tone_std,
        short_term_outlook_norm = S.short_term_outlook_norm,
        short_term_outlook_mean = S.short_term_outlook_mean,
        short_term_outlook_std = S.short_term_outlook_std,
        forward_looking_signals_norm = S.forward_looking_signals_norm,
        forward_looking_signals_mean = S.forward_looking_signals_mean,
        forward_looking_signals_std = S.forward_looking_signals_std,
        qa_summary_norm = S.qa_summary_norm,
        qa_summary_mean = S.qa_summary_mean,
        qa_summary_std = S.qa_summary_std,
        cos_fin_disc = S.cos_fin_disc,
        cos_fin_tone = S.cos_fin_tone,
        cos_disc_short = S.cos_disc_short,
        cos_short_fwd = S.cos_short_fwd,
        lda_topic_0 = S.lda_topic_0,
        lda_topic_1 = S.lda_topic_1,
        lda_topic_2 = S.lda_topic_2,
        lda_topic_3 = S.lda_topic_3,
        lda_topic_4 = S.lda_topic_4,
        lda_topic_5 = S.lda_topic_5,
        lda_topic_6 = S.lda_topic_6,
        lda_topic_7 = S.lda_topic_7,
        lda_topic_8 = S.lda_topic_8,
        lda_topic_9 = S.lda_topic_9,
        key_financial_metrics_pos_prob = S.key_financial_metrics_pos_prob,
        key_financial_metrics_neg_prob = S.key_financial_metrics_neg_prob,
        key_financial_metrics_neu_prob = S.key_financial_metrics_neu_prob,
        key_discussion_points_pos_prob = S.key_discussion_points_pos_prob,
        key_discussion_points_neg_prob = S.key_discussion_points_neg_prob,
        key_discussion_points_neu_prob = S.key_discussion_points_neu_prob,
        sentiment_tone_pos_prob = S.sentiment_tone_pos_prob,
        sentiment_tone_neg_prob = S.sentiment_tone_neg_prob,
        sentiment_tone_neu_prob = S.sentiment_tone_neu_prob,
        short_term_outlook_pos_prob = S.short_term_outlook_pos_prob,
        short_term_outlook_neg_prob = S.short_term_outlook_neg_prob,
        short_term_outlook_neu_prob = S.short_term_outlook_neu_prob,
        forward_looking_signals_pos_prob = S.forward_looking_signals_pos_prob,
        forward_looking_signals_neg_prob = S.forward_looking_signals_neg_prob,
        forward_looking_signals_neu_prob = S.forward_looking_signals_neu_prob,
        qa_summary_pos_prob = S.qa_summary_pos_prob,
        qa_summary_neg_prob = S.qa_summary_neg_prob,
        qa_summary_neu_prob = S.qa_summary_neu_prob
    WHEN NOT MATCHED BY TARGET AND ({validation_filter}) THEN
      INSERT (
        ticker, quarter_end_date, earnings_call_date, sector, industry, 
        industry_sector, adj_close_on_call_date, sentiment_score, eps_surprise, 
        eps_missing, max_close_30d, key_financial_metrics_embedding, 
        key_discussion_points_embedding, sentiment_tone_embedding, 
        short_term_outlook_embedding, forward_looking_signals_embedding, 
        qa_summary_embedding, sma_20, ema_50, rsi_14, adx_14, sma_20_delta, 
        ema_50_delta, rsi_14_delta, adx_14_delta, eps_surprise_isnull, 
        price_sma20_ratio, ema50_sma20_ratio, rsi_centered, sent_rsi, adx_log1p, 
        key_financial_metrics_norm, key_financial_metrics_mean, 
        key_financial_metrics_std, key_discussion_points_norm, 
        key_discussion_points_mean, key_discussion_points_std, 
        sentiment_tone_norm, sentiment_tone_mean, sentiment_tone_std, 
        short_term_outlook_norm, short_term_outlook_mean, short_term_outlook_std, 
        forward_looking_signals_norm, forward_looking_signals_mean, 
        forward_looking_signals_std, qa_summary_norm, qa_summary_mean, 
        qa_summary_std, cos_fin_disc, cos_fin_tone, cos_disc_short, cos_short_fwd, 
        lda_topic_0, lda_topic_1, lda_topic_2, lda_topic_3, lda_topic_4, 
        lda_topic_5, lda_topic_6, lda_topic_7, lda_topic_8, lda_topic_9, 
        key_financial_metrics_pos_prob, key_financial_metrics_neg_prob, 
        key_financial_metrics_neu_prob, key_discussion_points_pos_prob, 
        key_discussion_points_neg_prob, key_discussion_points_neu_prob, 
        sentiment_tone_pos_prob, sentiment_tone_neg_prob, sentiment_tone_neu_prob, 
        short_term_outlook_pos_prob, short_term_outlook_neg_prob, 
        short_term_outlook_neu_prob, forward_looking_signals_pos_prob, 
        forward_looking_signals_neg_prob, forward_looking_signals_neu_prob, 
        qa_summary_pos_prob, qa_summary_neg_prob, qa_summary_neu_prob
      ) VALUES (
        S.ticker, S.quarter_end_date, S.earnings_call_date, S.sector, S.industry, 
        S.industry_sector, S.adj_close_on_call_date, S.sentiment_score, S.eps_surprise, 
        S.eps_missing, S.max_close_30d, S.key_financial_metrics_embedding, 
        S.key_discussion_points_embedding, S.sentiment_tone_embedding, 
        S.short_term_outlook_embedding, S.forward_looking_signals_embedding, 
        S.qa_summary_embedding, S.sma_20, S.ema_50, S.rsi_14, S.adx_14, S.sma_20_delta, 
        S.ema_50_delta, S.rsi_14_delta, S.adx_14_delta, S.eps_surprise_isnull, 
        S.price_sma20_ratio, S.ema50_sma20_ratio, S.rsi_centered, S.sent_rsi, S.adx_log1p, 
        S.key_financial_metrics_norm, S.key_financial_metrics_mean, 
        S.key_financial_metrics_std, S.key_discussion_points_norm, 
        S.key_discussion_points_mean, S.key_discussion_points_std, 
        S.sentiment_tone_norm, S.sentiment_tone_mean, S.sentiment_tone_std, 
        S.short_term_outlook_norm, S.short_term_outlook_mean, S.short_term_outlook_std, 
        S.forward_looking_signals_norm, S.forward_looking_signals_mean, 
        S.forward_looking_signals_std, S.qa_summary_norm, S.qa_summary_mean, 
        S.qa_summary_std, S.cos_fin_disc, S.cos_fin_tone, S.cos_disc_short, S.cos_short_fwd, 
        S.lda_topic_0, S.lda_topic_1, S.lda_topic_2, S.lda_topic_3, S.lda_topic_4, 
        S.lda_topic_5, S.lda_topic_6, S.lda_topic_7, S.lda_topic_8, S.lda_topic_9, 
        S.key_financial_metrics_pos_prob, S.key_financial_metrics_neg_prob, 
        S.key_financial_metrics_neu_prob, S.key_discussion_points_pos_prob, 
        S.key_discussion_points_neg_prob, S.key_discussion_points_neu_prob, 
        S.sentiment_tone_pos_prob, S.sentiment_tone_neg_prob, S.sentiment_tone_neu_prob, 
        S.short_term_outlook_pos_prob, S.short_term_outlook_neg_prob, 
        S.short_term_outlook_neu_prob, S.forward_looking_signals_pos_prob, 
        S.forward_looking_signals_neg_prob, S.forward_looking_signals_neu_prob, 
        S.qa_summary_pos_prob, S.qa_summary_neg_prob, S.qa_summary_neu_prob
      )
    """
    
    truncate_query = f"TRUNCATE TABLE `{staging_full}`"

    try:
        logging.info("Executing MERGE statement...")
        merge_job = bq_client.query(merge_query)
        merge_job.result()
        logging.info(f"Merge successful. {merge_job.num_dml_affected_rows} rows affected.")

        # Post-merge: Sample verification (null check on final for engineered)
        post_stats_query = f"""
        SELECT 
          {', '.join(f'COUNTIF({col} IS NULL) AS null_{col}' for col in SOFT_NOT_NULL_COLUMNS if '_norm' in col or 'cos_' in col or 'lda_' in col or '_prob' in col)}  -- Focus on engineered, LDA, FinBERT
        FROM `{final_full}`
        """
        post_stats = bq_client.query(post_stats_query).to_dataframe().iloc[0]
        logging.info(f"Post-merge sample nulls in final (engineered/LDA/FinBERT): {post_stats.to_dict()}")

        logging.info("Executing TRUNCATE statement...")
        truncate_job = bq_client.query(truncate_query)
        truncate_job.result()
        logging.info(f"Staging table {STAGING_TABLE} truncated.")

        return "Merge completed successfully", 200
    except Exception as e:
        logging.error(f"Merge process failed: {str(e)}", exc_info=True)
        return "Merge process failed", 500