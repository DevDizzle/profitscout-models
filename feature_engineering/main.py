import os
import json
import base64
import logging
from flask import Flask, request
from utils import processing, bq

# We no longer need the FMP_API_KEY environment variable
PROJECT_ID = os.environ.get('PROJECT_ID')
DESTINATION_TABLE = os.environ.get('DESTINATION_TABLE')

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/', methods=['POST'])
def process_pubsub_message():
    """
    Entry point for the Cloud Run service.
    Receives and processes a Pub/Sub message.
    """
    envelope = request.get_json()
    if not envelope or 'message' not in envelope:
        logging.error('Bad Request: invalid Pub/Sub message format')
        return 'Bad Request: invalid Pub/Sub message format', 400

    message = envelope['message']
    logging.info(f"Received message: {message}")

    try:
        data_str = base64.b64decode(message['data']).decode('utf-8').replace("'", "\"")
        data = json.loads(data_str)
        
        logging.info(f"Processing data: {data}")

        # --- Main Orchestration ---
        # ** THE FIX IS HERE: **
        # We now call create_features with only the message data.
        final_row = processing.create_features(message=data)

        if not final_row:
            logging.error(f"Feature creation failed for {data.get('ticker')}")
            return 'Processing failed', 200 

        bq.upsert_row(
            row=final_row,
            table_id=DESTINATION_TABLE,
            primary_keys=['ticker', 'quarter_end_date']
        )
        
        logging.info(f"Successfully processed and loaded data for {data.get('ticker')}")
        return 'Success', 204

    except Exception as e:
        logging.error(f"FATAL: Unhandled exception: {e}")
        return 'Internal Server Error', 500

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=PORT, debug=True)