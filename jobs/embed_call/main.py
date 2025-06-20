#!/usr/bin/env python3
"""
Job 1: build Gemini embeddings for each markdown section
Publishes the *same* message to the next topic when done.
"""
import base64, json, os, logging, re
from flask import Flask, request

from google.cloud import storage, pubsub_v1
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

from utils import parse_sections, bq_client, table_ref, upsert_json

PROJECT      = os.getenv("PROJECT_ID", "profitscout-lx6bb")
DATASET      = os.getenv("BQ_DATASET", "profit_scout")
TABLE        = table_ref(DATASET, "breakout_features", PROJECT)
TOPIC_OUT    = os.getenv("TOPIC_SENTIMENT", "sentiment-todo")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "gemini-embedding-001")

storage_client = storage.Client(project=PROJECT)
publisher      = pubsub_v1.PublisherClient()
bq             = bq_client(PROJECT)
model          = TextEmbeddingModel.from_pretrained(EMBED_MODEL)

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# ──────────────────────────────────────────────────────────────
def process(msg: dict):
    logging.info(f"[process] Received msg: {msg}")
    ticker    = msg.get("ticker")
    qend      = msg.get("quarter_end")
    blob_uri  = msg.get("txt_blob")
    BUCKET    = os.getenv("GCS_BUCKET", "profit-scout-data")

    logging.info(f"[process] GCS bucket: {BUCKET}")
    logging.info(f"[process] blob_uri: {blob_uri}")

    try:
        md = storage_client.bucket(BUCKET).blob(blob_uri).download_as_text()
        logging.info(f"[process] Successfully fetched {blob_uri}")
    except Exception as e:
        logging.error(f"[process] ERROR fetching {blob_uri} from bucket {BUCKET}: {e}")
        raise

    sects = parse_sections(md)

    # Section-to-BQ column mapping
    SECTION_TO_COL = {
        "Key Financial Metrics": "key_financial_metrics_embedding",
        "Key Discussion Points": "key_discussion_points_embedding",
        "Sentiment Tone": "sentiment_tone_embedding",
        "Short-Term Outlook": "short_term_outlook_embedding",
        "Forward-Looking Signals": "forward_looking_signals_embedding",
    }

    row = {"ticker": ticker, "quarter_end_date": qend}

    for sec, col in SECTION_TO_COL.items():
        txt = sects.get(sec)
        if txt:
            logging.info(f"[process] Embedding section: {sec} -> {col}")
            try:
                # Gemini embedding: only pass a string, not a list!
                emb = model.get_embeddings([txt])[0].values
                row[col] = emb
                logging.info(f"[process] Embedding shape for {col}: {len(emb)}")
            except Exception as e:
                logging.error(f"[process] Error embedding {sec}: {e}")
                row[col] = None
        else:
            logging.warning(f"[process] Section '{sec}' not found in document. Skipping {col}.")
            row[col] = None

    logging.info(f"[process] Upserting row: {row}")

    upsert_json(TABLE, row, bq)
    logging.info(f"[process] Row upserted: {list(row.keys())}")

    # pass work downstream
    publisher.publish(
        f"projects/{PROJECT}/topics/{TOPIC_OUT}",
        json.dumps(msg).encode()
    )
    logging.info(f"[process] Published message to {TOPIC_OUT}")


@app.route("/", methods=["POST"])
def handler():
    envelope = request.get_json()
    logging.info(f"[handler] Received envelope: {envelope}")
    try:
        payload = json.loads(base64.b64decode(envelope["message"]["data"]))
        logging.info(f"[handler] Decoded payload: {payload}")
        process(payload)
    except Exception as e:
        logging.error(f"[handler] Error in handler: {e}")
        return (str(e), 500)
    return ("", 204)

# local CLI test: python main.py samples/call.txt AAPL 2025-03-31
if __name__ == "__main__":
    import sys, pathlib
    txt, ticker, q = sys.argv[1:]
    data = {
        "ticker": ticker,
        "quarter_end": q,
        "txt_blob": f"{pathlib.Path(txt).absolute()}"
    }
    logging.info(f"[main] Test mode: data={data}")
    process(data)