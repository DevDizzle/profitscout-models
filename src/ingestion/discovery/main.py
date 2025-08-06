# src/ingestion/discovery/main.py
import os
import json
import logging
from google.cloud import pubsub_v1, storage, bigquery
import functions_framework

# --- Environment Variables ---
PROJECT_ID = os.environ.get("PROJECT_ID", "profitscout-lx6bb")
PUB_SUB_TOPIC = os.environ.get("PUB_SUB_TOPIC", "process-earnings-call")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "profit-scout-data")
BQ_TABLE_ID = os.environ.get("BQ_TABLE_ID", "profitscout-lx6bb.profit_scout.breakout_features")
GCS_SUMMARIES_PREFIX = "earnings-call-summaries/"
GCS_TRANSCRIPTS_PREFIX = "earnings-call-transcripts/" # Define transcript prefix for consistency

# --- Clients ---
publisher = pubsub_v1.PublisherClient()
storage_client = storage.Client()
bq_client = bigquery.Client()

logging.basicConfig(level=logging.INFO)

def get_existing_features():
    """Queries BigQuery to get a set of existing 'ticker_date' combinations."""
    query = f"SELECT DISTINCT ticker, quarter_end_date FROM `{BQ_TABLE_ID}`"
    try:
        rows = bq_client.query(query).result()
        return {f"{row.ticker}_{row.quarter_end_date.isoformat()}" for row in rows}
    except Exception as e:
        logging.error(f"Failed to query BigQuery for existing features: {e}")
        return set()

@functions_framework.http
def discover_missing_features(request):
    """
    Scans a GCS bucket for summaries, compares against features in BigQuery,
    and publishes messages for any missing features.
    """
    logging.info("Starting discovery of missing features...")

    existing_features = get_existing_features()
    logging.info(f"Found {len(existing_features)} existing features in BigQuery.")

    all_summary_blobs = storage_client.list_blobs(BUCKET_NAME, prefix=GCS_SUMMARIES_PREFIX)
    
    files_to_process = []
    for blob in all_summary_blobs:
        if not blob.name.endswith('.txt'):
            continue
        
        file_key = os.path.basename(blob.name).removesuffix('.txt')
        
        if file_key not in existing_features:
            files_to_process.append((file_key, blob.name))

    if not files_to_process:
        logging.info("No new summaries to process. Feature store is up to date.")
        return "No new summaries to process.", 200

    topic_path = publisher.topic_path(PROJECT_ID, PUB_SUB_TOPIC)
    published_count = 0
    logging.info(f"Found {len(files_to_process)} missing features. Publishing messages...")
    
    for file_key, gcs_path in files_to_process:
        try:
            ticker, q_end_date = file_key.split('_')
            
            # --- FIX: Added the required 'transcript_gcs_path' to the message ---
            message_data = {
                "ticker": ticker,
                "quarter_end_date": q_end_date,
                "summary_gcs_path": f"gs://{BUCKET_NAME}/{gcs_path}",
                "transcript_gcs_path": f"gs://{BUCKET_NAME}/{GCS_TRANSCRIPTS_PREFIX}{file_key}.json"
            }
            
            future = publisher.publish(topic_path, data=json.dumps(message_data).encode('utf-8'))
            future.result()
            published_count += 1
            
        except Exception as e:
            logging.error(f"Failed to publish message for {gcs_path}: {e}")

    final_message = f"Discovery complete. Published {published_count} messages for processing."
    logging.info(final_message)
    return final_message, 200