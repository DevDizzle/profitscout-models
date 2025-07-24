import logging
import json
import os
import time

from google.cloud import storage
from google.cloud import pubsub_v1
from google.cloud import bigquery

# --- Configuration ---
PROJECT_ID = "profitscout-lx6bb"
BUCKET_NAME = "profit-scout-data"
GCS_PREFIX = "earnings-call-summaries/"
PUB_SUB_TOPIC_ID = "process-earnings-call"

# --- BigQuery Configuration for checking existing features ---
BQ_FEATURE_TABLE_ID = "profitscout-lx6bb.profit_scout.staging_breakout_features"

# --- Control how many messages are published per second ---
PUBLISH_RATE_PER_SECOND = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_existing_records():
    """
    Queries the feature store table in BigQuery to get all existing
    ticker and quarter_end_date combinations.
    Returns:
        A set of tuples, e.g., {('AAPL', '2023-09-30'), ('GOOG', '2023-09-30')}
    """
    logging.info(f"Querying BigQuery table {BQ_FEATURE_TABLE_ID} for existing records...")
    bq_client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        SELECT DISTINCT ticker, quarter_end_date
        FROM `{BQ_FEATURE_TABLE_ID}`
    """

    try:
        query_job = bq_client.query(query)
        rows = query_job.result()
        existing_records = {(row.ticker, row.quarter_end_date.isoformat()) for row in rows}
        logging.info(f"Found {len(existing_records)} existing records in the feature store.")
        return existing_records
    except Exception as e:
        logging.error(f"Failed to query BigQuery: {e}")
        raise

def backfill_with_rate_limiting():
    """
    Lists GCS files, filters out those already processed, and publishes
    messages for the remaining files at a controlled, throttled rate.
    TEST MODE: Only processes 1 files to confirm updates work.
    """
    existing_records = get_existing_records()

    storage_client = storage.Client(project=PROJECT_ID)
    logging.info(f"Scanning GCS bucket gs://{BUCKET_NAME}/{GCS_PREFIX} for files...")
    all_blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix=GCS_PREFIX))

    if not all_blobs:
        logging.error("No files found in GCS bucket. Aborting.")
        return

    files_to_process = []
    for blob in all_blobs:
        if not blob.name.endswith('.txt'):
            continue
        try:
            file_name = os.path.basename(blob.name)
            ticker, q_end_str = file_name.removesuffix('.txt').split('_')
            if (ticker, q_end_str) not in existing_records:
                files_to_process.append(blob)
        except ValueError:
            logging.warning(f"Could not parse ticker/date from filename: {blob.name}. Skipping.")
            continue

    logging.info(f"Total files in GCS: {len(all_blobs)}. Already processed: {len(existing_records)}. New files to process: {len(files_to_process)}")

    if not files_to_process:
        logging.info("No new files to backfill. All records are up-to-date.")
        return

    # ✅ Limit to 2 files only for testing
    files_to_process = files_to_process[:1]
    logging.warning("TEST MODE: Only publishing 1 messages to confirm updates work.")

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, PUB_SUB_TOPIC_ID)
    delay_between_messages = 1.0 / PUBLISH_RATE_PER_SECOND

    published_count = 0
    for blob in files_to_process:
        try:
            file_name = os.path.basename(blob.name)
            ticker, q_end_str = file_name.removesuffix('.txt').split('_')
            message_data = {
                "ticker": ticker,
                "quarter_end_date": q_end_str,
                "summary_gcs_path": f"gs://{BUCKET_NAME}/{blob.name}",
                "transcript_gcs_path": f"gs://{BUCKET_NAME}/earnings-call-transcripts/{ticker}_{q_end_str}.json"
            }
            message_bytes = json.dumps(message_data).encode('utf-8')

            future = publisher.publish(topic_path, data=message_bytes)
            future.result()
            published_count += 1

            logging.info(f"Published message {published_count}: {ticker}, {q_end_str}")
            time.sleep(delay_between_messages)

        except Exception as e:
            logging.error(f"Failed to publish message for {blob.name}: {e}")

    logging.info(f"Successfully published {published_count} messages in TEST MODE.")

if __name__ == "__main__":
    backfill_with_rate_limiting()
