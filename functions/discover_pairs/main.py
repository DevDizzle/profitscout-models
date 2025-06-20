import os, json, datetime, re
from google.cloud import storage, pubsub_v1, bigquery
import logging
logging.info(f"Publishing msg: {msg}")

# ── ENV ──────────────────────────────────────────────────────────
PROJECT   = os.getenv("PROJECT_ID",  "profitscout-lx6bb")
BUCKET    = os.getenv("GCS_BUCKET",  "profit-scout-data")
SUMMARY_P = os.getenv("SUMMARY_PREFIX", "earnings-call-summary/")
DST_TABLE = f"{PROJECT}.profit_scout.breakout_features"
TOPIC     = os.getenv("PUBSUB_TOPIC", "ingest-todo")

# ── clients ──────────────────────────────────────────────────────
bq   = bigquery.Client(project=PROJECT)
gcs  = storage.Client(project=PROJECT).bucket(BUCKET)
pub  = pubsub_v1.PublisherClient()
topic_path = pub.topic_path(PROJECT, TOPIC)

# ── helper: set of (TICKER, QUARTER_END_DATE) already ingested ──
def already_done() -> set[tuple[str, datetime.date]]:
    sql = f"""
      SELECT DISTINCT UPPER(ticker) AS tk,
                      DATE(quarter_end_date) AS qd
      FROM `{DST_TABLE}`
    """
    df = bq.query(sql).to_dataframe()
    return {(r.tk, r.qd) for r in df.itertuples()}

# ── Cloud Function entry-point ───────────────────────────────────
def discover_pairs(request):
    done = already_done()
    published = 0

    for blob in gcs.list_blobs(prefix=SUMMARY_P):
        if not blob.name.endswith(".txt"):
            continue

        base = os.path.splitext(blob.name.split("/")[-1])[0]   # TICKER_YYYY-MM-DD
        try:
            tk, qd_s = base.split("_", 1)
            qd = datetime.date.fromisoformat(qd_s)
        except ValueError:
            # Skip filenames that don’t match the expected pattern
            continue

        if (tk.upper(), qd) in done:
            continue        # already ingested → skip

        # Publish work item
        msg = {"ticker": tk, "quarter_end": qd_s, "txt_blob": blob.name}
        pub.publish(topic_path, json.dumps(msg).encode())
        published += 1

    return f"Published {published} new work items", 200
