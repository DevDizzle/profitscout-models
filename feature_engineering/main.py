"""Cloud Run service that generates features from transcript events."""

import os
import json
import base64
import logging
from flask import Flask, request
from utils import processing
from google.cloud import pubsub_v1

# --- Environment Variables ---
PROJECT_ID = os.environ.get('PROJECT_ID')
LOADER_PUB_SUB_TOPIC = os.environ.get('LOADER_PUB_SUB_TOPIC') # e.g. 'features-for-loading'

# --- Clients ---
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, LOADER_PUB_SUB_TOPIC)
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


@app.route('/', methods=['POST'])
def process_pubsub_message():
    """
    Receives a message to process features for a transcript, then
    publishes the completed feature row to the loader topic.
    """
    envelope = request.get_json()
    if not envelope or 'message' not in envelope:
        logging.error('Bad Request: invalid Pub/Sub message format')
        return 'Bad Request: invalid Pub/Sub message format', 400

    message = envelope['message']
    
    try:
        data_str = base64.b64decode(message['data']).decode('utf-8')
        data = json.loads(data_str)
        
        logging.info(f"Processing features for ticker: {data.get('ticker')}")

        final_row = processing.create_features(message=data)

        if not final_row:
            logging.warning(f"Feature creation returned None. Acking message to prevent retries.")
            return 'Processing failed, message acknowledged', 200

        future = publisher.publish(topic_path, data=json.dumps(final_row).encode('utf-8'))
        future.result() # Wait for publish to complete

        logging.info(f"Successfully published features for {final_row.get('ticker')} to loader topic.")
        return 'Success', 204

    except Exception as e:
        logging.error(f"FATAL: Unhandled exception: {e}", exc_info=True)
        # Return a server error to trigger a retry from Pub/Sub
        return 'Internal Server Error', 500

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 8080))    app.run(host='0.0.0.0', port=PORT, debug=True)
