"""Cloud Run service to stream processed features into BigQuery."""

import os
import json
import base64
import logging
from flask import Flask, request
from google.cloud import bigquery
from google.cloud.bigquery.exceptions import BigQueryError

# --- Environment Variables ---
DESTINATION_TABLE = os.environ.get('DESTINATION_TABLE')

# --- Clients ---
bq_client = bigquery.Client()
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/', methods=['POST'])
def load_batch_to_bq():
    """
    Receives messages from a Pub/Sub push subscription and streams
    the data into BigQuery. Handles both single and batch messages.
    """
    envelope = request.get_json()
    if not envelope:
        logging.error('Bad Request: no Pub/Sub message received')
        return 'Bad Request: no Pub/Sub message received', 400

    rows_to_insert = []
    
    try:
        # Check for a single message
        if 'message' in envelope:
            message_data = base64.b64decode(envelope['message']['data']).decode('utf-8')
            row = json.loads(message_data)
            rows_to_insert.append(row)
        # Check for a batch of messages
        elif 'messages' in envelope:
            for message in envelope['messages']:
                message_data = base64.b64decode(message['data']).decode('utf-8')
                row = json.loads(message_data)
                rows_to_insert.append(row)
        # If neither format is found, it's an invalid request
        else:
            logging.error('Bad Request: invalid Pub/Sub message format')
            return 'Bad Request: invalid Pub/Sub message format', 400
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
        logging.error(f"Could not decode message data. Error: {e}")
        # Acknowledge message to prevent retries of malformed data
        return "Bad Request: Could not decode message data", 400


    if not rows_to_insert:
        logging.info("No valid rows to insert.")
        return "Success: No valid rows processed", 200

    logging.info(f"Attempting to insert {len(rows_to_insert)} rows into {DESTINATION_TABLE}")
    
    try:
        errors = bq_client.insert_rows_json(DESTINATION_TABLE, rows_to_insert)
        if not errors:
            logging.info("Successfully inserted all rows.")
            return "Success: Batch loaded", 204
        else:
            logging.error(f"Encountered errors while inserting rows: {errors}")
            return "Error: Partial or total batch failure", 500
    except BigQueryError as e:
        logging.error(f"FATAL: BigQuery API error: {e}", exc_info=True)
        return "Internal Server Error", 500

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 8080))    
    app.run(host='0.0.0.0', port=PORT, debug=True)
