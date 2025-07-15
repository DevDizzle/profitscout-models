# update_max_close/main.py
import os
import functions_framework
from google.cloud import bigquery

# --- Configuration ---
# Set these as environment variables in your Cloud Function
PROJECT_ID      = os.environ.get("PROJECT_ID")
BIGQUERY_DATASET = os.environ.get("BIGQUERY_DATASET", "profit_scout")
FEATURE_TABLE   = os.environ.get("FEATURE_TABLE", "breakout_features")

# --- BigQuery Client ---
bq_client = bigquery.Client(project=PROJECT_ID)

# --- The MERGE Query
MERGE_QUERY = f"""
MERGE INTO `{PROJECT_ID}.{BIGQUERY_DATASET}.{FEATURE_TABLE}` AS T
USING (
    -- Step 1: Find the max close price for all tickers in the 30-day window
    -- after their earnings call, using the correct price_data table.
    WITH max_prices AS (
        SELECT
            ticker,
            date AS call_date, -- Assuming the date column in price_data corresponds to earnings_call_date
            MAX(adj_close) AS calculated_max_close
        FROM `profitscout-lx6bb.profit_scout.price_data` -- Corrected table name
        GROUP BY ticker, call_date
    )
    -- Step 2: Select only the rows from our feature table that are ready for an update.
    SELECT
        source.ticker,
        source.quarter_end_date,
        mp.calculated_max_close
    FROM `{PROJECT_ID}.{BIGQUERY_DATASET}.{FEATURE_TABLE}` AS source
    JOIN max_prices AS mp
      ON source.ticker = mp.ticker AND source.earnings_call_date = mp.call_date
    WHERE
        source.max_close_30d IS NULL
        AND source.earnings_call_date <= CURRENT_DATE() - INTERVAL 31 DAY

) AS S
ON T.ticker = S.ticker AND T.quarter_end_date = S.quarter_end_date
WHEN MATCHED THEN
  UPDATE SET T.max_close_30d = S.calculated_max_close;
"""

@functions_framework.http
def update_max_close(request):
    """
    HTTP-triggered Cloud Function to backfill the max_close_30d feature.
    """
    print("Starting backfill job for max_close_30d.")
    print(f"Targeting feature table: {PROJECT_ID}.{BIGQUERY_DATASET}.{FEATURE_TABLE}")

    try:
        job = bq_client.query(MERGE_QUERY)
        job.result()  # Wait for the job to complete

        rows_updated = job.num_dml_affected_rows
        print(f"Successfully completed backfill job. Updated {rows_updated} rows.")
        return f"Successfully updated {rows_updated} rows.", 200

    except Exception as e:
        print(f"Error running backfill job: {e}")
        return f"Error running backfill job.", 500