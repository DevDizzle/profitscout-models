"""
Job 2: Calculate sentiment of the full earnings call transcript.

This service is triggered by a Pub/Sub message on the 'sentiment-todo' topic.

The incoming message contains:
- ticker: The stock ticker symbol.
- quarter_end: The quarter-end date for the earnings call.

Workflow:
1. Receives the trigger message.
2. Constructs the GCS path to the full JSON transcript for the given ticker and date.
3. Fetches the transcript from GCS, extracts the text content AND the earnings_call_date.
4. Uses the Natural Language API to analyze the sentiment of the text.
5. Upserts the sentiment score into the 'breakout_features' BigQuery table.
6. Publishes an enriched message containing ticker, quarter_end, and the newly
   extracted earnings_call_date to the 'pricing-todo' topic.
"""
from flask import Flask, request
import base64, json, os, logging
from google.cloud import language_v1, storage, pubsub_v1
from utils import bq_client, table_ref, upsert_json

PROJECT   = os.getenv("PROJECT_ID", "profitscout-lx6bb")
DATASET   = os.getenv("BQ_DATASET", "profit_scout")
TABLE     = table_ref(DATASET, "breakout_features", PROJECT)
TOPIC_OUT = os.getenv("TOPIC_PRICE", "pricing-todo")
BUCKET    = os.getenv("GCS_BUCKET", "profit-scout-data")

nl_client = language_v1.LanguageServiceClient()
publisher = pubsub_v1.PublisherClient()
storage_client = storage.Client(project=PROJECT)
bq        = bq_client(PROJECT)
app = Flask(__name__)

def process(msg):
    ticker = msg["ticker"]
    qend = msg["quarter_end"]

    blob_uri = f"earnings-call/json/{ticker}_{qend}.json"
    logging.info(f"Fetching transcript from GCS: gs://{BUCKET}/{blob_uri}")

    try:
        blob_content = storage_client.bucket(BUCKET).blob(blob_uri).download_as_text()
        transcript_data = json.loads(blob_content)
        text = transcript_data.get("content", "")
        if not text:
            logging.error(f"'content' key not found or empty in {blob_uri}")
            return
    except Exception as e:
        logging.error(f"Failed to fetch or parse transcript from {blob_uri}: {e}")
        raise

    doc = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sent = nl_client.analyze_sentiment(request={"document": doc}).document_sentiment

    row  = {"ticker": ticker, "quarter_end_date": qend,
            "sentiment_score": sent.score}
    upsert_json(TABLE, row, bq)
    logging.info(f"Upserted sentiment for {ticker} {qend}")

    # --- CHANGE IS HERE ---
    # Enrich the message for the next service by adding the earnings call date.
    msg["earnings_call_date"] = transcript_data.get("date")

    publisher.publish(f"projects/{PROJECT}/topics/{TOPIC_OUT}",
                      json.dumps(msg).encode())
    logging.info(f"Published enriched message to {TOPIC_OUT}: {msg}")

@app.route("/", methods=["POST"])
def h():
    logging.basicConfig(level=logging.INFO)
    envelope=request.get_json()
    if not envelope or "message" not in envelope:
        logging.error("Invalid Pub/Sub message format")
        return "Bad Request", 400
    try:
        payload = json.loads(base64.b64decode(envelope["message"]["data"]))
        process(payload)
        return ("", 204)
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        return "Internal Server Error", 500