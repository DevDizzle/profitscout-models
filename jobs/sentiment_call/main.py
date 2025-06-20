from flask import Flask, request
import base64, json, os, logging
# Add the storage client
from google.cloud import language_v1, storage
from utils import bq_client, table_ref, upsert_json

PROJECT   = os.getenv("PROJECT_ID", "profitscout-lx6bb")
DATASET   = os.getenv("BQ_DATASET", "profit_scout")
TABLE     = table_ref(DATASET, "breakout_features", PROJECT)
TOPIC_OUT = os.getenv("TOPIC_PRICE", "pricing-todo")
# Add the GCS Bucket name
BUCKET    = os.getenv("GCS_BUCKET", "profit-scout-data")

nl_client = language_v1.LanguageServiceClient()
publisher = pubsub_v1.PublisherClient()
# Add the GCS client
storage_client = storage.Client(project=PROJECT)
bq        = bq_client(PROJECT)
app = Flask(__name__)

def process(msg):
    ticker = msg["ticker"]
    qend = msg["quarter_end"]

    # 1. Construct the path to the JSON transcript file in GCS
    blob_uri = f"earnings-call/json/{ticker}_{qend}.json"
    logging.info(f"Fetching transcript from GCS: gs://{BUCKET}/{blob_uri}")

    try:
        # 2. Download and parse the JSON transcript
        blob_content = storage_client.bucket(BUCKET).blob(blob_uri).download_as_text()
        transcript_data = json.loads(blob_content)
        # Assuming the text is in a 'content' field from your earlier examples
        text = transcript_data.get("content", "")
        if not text:
            logging.error(f"'content' key not found or empty in {blob_uri}")
            return # Stop processing if no text
    except Exception as e:
        logging.error(f"Failed to fetch or parse transcript from {blob_uri}: {e}")
        raise # Let Cloud Run handle the error and retry

    # 3. Perform sentiment analysis
    doc = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sent = nl_client.analyze_sentiment(request={"document": doc}).document_sentiment

    # 4. Upsert to BigQuery
    row  = {"ticker": ticker, "quarter_end_date": qend,
            "sent_score": sent.score, "sent_magnitude": sent.magnitude}
    upsert_json(TABLE, [row], bq)
    logging.info(f"Upserted sentiment for {ticker} {qend}")

    # 5. Pass original trigger message to the next service
    publisher.publish(f"projects/{PROJECT}/topics/{TOPIC_OUT}",
                      json.dumps(msg).encode())
    logging.info(f"Published message to {TOPIC_OUT}")

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
