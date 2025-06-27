import os
import logging
from google.cloud import storage, pubsub_v1, bigquery

# Environment Variables
PROJECT_ID = os.environ.get('PROJECT_ID')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
DESTINATION_TABLE = os.environ.get('DESTINATION_TABLE') # e.g., project.dataset.breakout_features
PUB_SUB_TOPIC = os.environ.get('PUB_SUB_TOPIC') # e.g., projects/your-project/topics/process-earnings-call

# Clients
storage_client = storage.Client()
publisher = pubsub_v1.PublisherClient()
bq_client = bigquery.Client()

logging.basicConfig(level=logging.INFO)

def discover_new_transcripts(event, context):
    """
    Cloud Function to discover new earnings call transcripts in GCS
    and publish a message to Pub/Sub for processing.
    """
    logging.info(f"Listing blobs in bucket: {BUCKET_NAME} with prefix 'earnings-call-summaries/'")
    blobs = storage_client.list_blobs(BUCKET_NAME, prefix='earnings-call-summaries/')
    
    # Get a set of already processed tickers and dates for quick lookup
    try:
        query = f"""
        SELECT DISTINCT FORMAT("%s_%t", ticker, quarter_end_date)
        FROM `{DESTINATION_TABLE}`
        """
        processed_items = {row[0] for row in bq_client.query(query).result()}
        logging.info(f"Found {len(processed_items)} already processed items in BigQuery.")
    except Exception as e:
        logging.error(f"Error querying BigQuery for processed items: {e}")
        processed_items = set()

    for blob in blobs:
        if not blob.name.endswith('.txt'):
            continue

        try:
            # Extract Ticker and Quarter End Date
            filename = os.path.basename(blob.name)
            ticker, q_end_str = os.path.splitext(filename)[0].rsplit('_', 1)
            
            item_key = f"{ticker}_{q_end_str}"
            if item_key in processed_items:
                continue

            logging.info(f"Found new item: {item_key}. Publishing to {PUB_SUB_TOPIC}")

            # CORRECTED PATHS in the message payload
            message_data = {
                "ticker": ticker,
                "quarter_end_date": q_end_str,
                "summary_gcs_path": f"gs://{BUCKET_NAME}/{blob.name}",
                "transcript_gcs_path": f"gs://{BUCKET_NAME}/earnings-call-transcripts/{ticker}_{q_end_str}.json"
            }
            
            # Publish message to Pub/Sub
            future = publisher.publish(PUB_SUB_TOPIC, data=json.dumps(message_data).encode('utf-8'))
            future.result()

        except Exception as e:
            logging.error(f"Failed to process or publish for blob {blob.name}: {e}")

    logging.info("Discovery function finished.")
    return 'OK', 200
    