from flask import Flask, request
import base64, json, os, logging
from google.cloud import language_v1
from utils import bq_client, table_ref, upsert_json

PROJECT   = os.getenv("PROJECT_ID", "profitscout-lx6bb")
DATASET   = os.getenv("BQ_DATASET", "profit_scout")
TABLE     = table_ref(DATASET, "breakout_features", PROJECT)
TOPIC_OUT = os.getenv("TOPIC_PRICE", "pricing-todo")

nl_client = language_v1.LanguageServiceClient()
publisher = pubsub_v1.PublisherClient()
bq        = bq_client(PROJECT)
app = Flask(__name__)

def process(msg):
    text = msg["sent_text"]           # include this in original Pub/Sub if you want
    doc = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sent = nl_client.analyze_sentiment(request={"document": doc}).document_sentiment
    row  = {"ticker": msg["ticker"], "quarter_end_date": msg["quarter_end"],
            "sent_score": sent.score, "sent_magnitude": sent.magnitude}
    upsert_json(TABLE, row, bq)
    publisher.publish(f"projects/{PROJECT}/topics/{TOPIC_OUT}",
                      json.dumps(msg).encode())

@app.route("/", methods=["POST"])
def h(): envelope=request.get_json(); process(json.loads(base64.b64decode(envelope["message"]["data"]))); return ("",204)
