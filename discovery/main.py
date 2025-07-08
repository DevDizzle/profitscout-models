"""Cloud Function to publish Pub/Sub messages for new transcripts."""

import os
import logging
import json
from google.cloud import pubsub_v1

PUB_SUB_TOPIC = os.environ.get('PUB_SUB_TOPIC') 

# Initialize a single PublisherClient
publisher = pubsub_v1.PublisherClient()

logging.basicConfig(level=logging.INFO)

def discover_new_summary(event, context):
    """
    A Cloud Function triggered by a new file in GCS. 
    
    It parses the file's name to extract metadata and publishes a message 
    to a Pub/Sub topic for the feature-engineering service to process.
    """
    # The event payload contains metadata for the file that triggered the function
    bucket_name = event['bucket']
    file_name = event['name']

    logging.info(f"Function triggered by file: {file_name} in bucket: {bucket_name}.")

    # For safety and clarity, ensure the file is what we expect.
    # The GCS trigger itself should be filtered to this path for best performance.
    if not file_name.startswith('earnings-call-summaries/') or not file_name.endswith('.txt'):
        logging.info(f"Skipping file '{file_name}' because it's not a new summary.")
        return 'OK', 200

    try:
        base_name = os.path.basename(file_name)
        ticker, q_end_str = os.path.splitext(base_name)[0].rsplit('_', 1)

        logging.info(f"Identified Ticker: {ticker}, Quarter: {q_end_str}. Publishing message.")

        # Construct the message payload for the downstream service
        message_data = {
            "ticker": ticker,
            "quarter_end_date": q_end_str,
            "summary_gcs_path": f"gs://{bucket_name}/{file_name}",
            # The path to the full transcript is inferred from a consistent naming convention
            "transcript_gcs_path": f"gs://{bucket_name}/earnings-call-transcripts/{ticker}_{q_end_str}.json"
        }

        # Publish the message to Pub/Sub, letting the feature-engineering service take over
        future = publisher.publish(PUB_SUB_TOPIC, data=json.dumps(message_data).encode('utf-8'))
        future.result()  # Ensures the message is sent before the function exits

        logging.info(f"Successfully published message for {ticker}_{q_end_str}")

    except (ValueError, IndexError) as e:
        # This error handles cases where the filename is not in the expected format
        logging.error(f"Could not parse ticker and date from filename: '{file_name}'. Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred processing file {file_name}: {e}")
    return 'OK', 200
