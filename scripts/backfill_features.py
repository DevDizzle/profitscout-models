import logging
import json
import os
import time
from google.cloud import storage, pubsub_v1, bigquery

# --- Configuration ---
PROJECT_ID = "profitscout-lx6bb"
BUCKET_NAME = "profit-scout-data"
GCS_SUMMARIES_PREFIX = "earnings-call-summaries/"
GCS_TRANSCRIPTS_PREFIX = "earnings-call-transcripts/"
PUB_SUB_TOPIC_ID = "process-earnings-call"
BQ_FEATURE_TABLE_ID = "profitscout-lx6bb.profit_scout.breakout_features"

# --- Control how many messages are published per second ---
# Adjust this value to control the backfill speed.
# e.g., 10 = 10 messages/sec, 1 = 1 message/sec
PUBLISH_RATE_PER_SECOND = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_existing_records():
    """Queries BigQuery to get existing ticker and quarter_end_date combinations."""
    logging.info(f"Querying BigQuery table {BQ_FEATURE_TABLE_ID} for existing records...")
    bq_client = bigquery.Client(project=PROJECT_ID)
    query = f"SELECT DISTINCT ticker, quarter_end_date FROM `{BQ_FEATURE_TABLE_ID}`"
    try:
        rows = bq_client.query(query).result()
        existing_records = {f"{row.ticker}_{row.quarter_end_date.isoformat()}" for row in rows}
        logging.info(f"Found {len(existing_records)} existing records.")
        return existing_records
    except Exception as e:
        logging.error(f"Failed to query BigQuery: {e}")
        raise

def backfill_with_rate_limiting():
    """
    Lists GCS files, filters out processed ones, and publishes messages
    for the remaining files at a controlled, throttled rate.
    """
    existing_records = get_existing_records()
    storage_client = storage.Client(project=PROJECT_ID)
    logging.info(f"Scanning GCS bucket gs://{BUCKET_NAME}/{GCS_SUMMARIES_PREFIX} for files...")
    all_blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix=GCS_SUMMARIES_PREFIX))

    files_to_process = []
    for blob in all_blobs:
        if not blob.name.endswith('.txt'):
            continue
        try:
            file_key = os.path.basename(blob.name).removesuffix('.txt')
            if file_key not in existing_records:
                files_to_process.append((file_key, blob.name))
        except ValueError:
            logging.warning(f"Could not parse filename: {blob.name}. Skipping.")

    if not files_to_process:
        logging.info("No new files to backfill. All records are up-to-date.")
        return

    logging.info(f"Found {len(files_to_process)} missing features. Publishing messages...")

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, PUB_SUB_TOPIC_ID)
    
    # Calculate the delay needed to maintain the desired publishing rate
    delay_between_messages = 1.0 / PUBLISH_RATE_PER_SECOND
    
    published_count = 0
    for file_key, gcs_path in files_to_process:
        try:
            ticker, q_end_str = file_key.split('_')
            message_data = {
                "ticker": ticker,
                "quarter_end_date": q_end_str,
                "summary_gcs_path": f"gs://{BUCKET_NAME}/{gcs_path}",
                "transcript_gcs_path": f"gs://{BUCKET_NAME}/{GCS_TRANSCRIPTS_PREFIX}{file_key}.json"
            }
            message_bytes = json.dumps(message_data).encode('utf-8')
            
            future = publisher.publish(topic_path, data=message_bytes)
            future.result()  # Confirm message is received by Pub/Sub
            published_count += 1
            
            if published_count % 100 == 0 and published_count > 0:
                logging.info(f"Published {published_count}/{len(files_to_process)} messages...")

            # The key to controlling the flow: wait before sending the next message
            time.sleep(delay_between_messages)
            
        except Exception as e:
            logging.error(f"Failed to publish message for {gcs_path}: {e}")

    logging.info(f"Successfully published all {published_count} new messages at a controlled rate.")
    logging.warning("The feature-engineering service will now process this batch gradually.")

if __name__ == "__main__":
    backfill_with_rate_limiting()