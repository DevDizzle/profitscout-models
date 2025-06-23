import os, json, datetime, re
from google.cloud import storage, pubsub_v1, bigquery
import logging

# ── ENV ──────────────────────────────────────────────────────────
PROJECT   = os.getenv("PROJECT_ID",  "profitscout-lx6bb")
BUCKET    = os.getenv("GCS_BUCKET",  "profit-scout-data")
SUMMARY_P = os.getenv("SUMMARY_PREFIX", "earnings-call-summary/")
SRC_TABLE = f"{PROJECT}.profit_scout.breakout_features"
# Publishes to a topic meant for the price_technicals job
TOPIC     = os.getenv("PUBSUB_TOPIC", "price-technicals-todo") 

# ── clients ──────────────────────────────────────────────────────
bq   = bigquery.Client(project=PROJECT)
gcs  = storage.Client(project=PROJECT).bucket(BUCKET)
pub  = pubsub_v1.PublisherClient()
topic_path = pub.topic_path(PROJECT, TOPIC)

# ── helper: set of (TICKER, QUARTER_END_DATE) already processed by price_technicals ──
def already_done() -> set[tuple[str, datetime.date]]:
    """
    Returns the set of (ticker, quarter_end_date) pairs where all
    technical indicator columns are NOT NULL.
    """
    # These are the 8 columns that the price_technicals job adds
    technicals_cols = [
        "sma_20", "sma_20_delta",
        "ema_50", "ema_50_delta",
        "rsi_14", "rsi_14_delta",
        "adx_14", "adx_14_delta"
    ]
    
    # Build a WHERE clause to check if all columns are filled
    where_clause = " AND ".join(f"{col} IS NOT NULL" for col in technicals_cols)

    sql = f"""
      SELECT DISTINCT UPPER(ticker) AS tk,
                      DATE(quarter_end_date) AS qd
      FROM `{SRC_TABLE}`
      WHERE {where_clause}
    """
    df = bq.query(sql).to_dataframe()
    return {(r.tk, r.qd) for r in df.itertuples()}

# ── Cloud Function entry-point ───────────────────────────────────
def discover_price_technicals_work(request):
    done = already_done()
    published = 0

    # This logic is the same as discover_pairs
    # It finds all potential work items...
    for blob in gcs.list_blobs(prefix=SUMMARY_P):
        if not blob.name.endswith(".txt"):
            continue

        base = os.path.splitext(blob.name.split("/")[-1])[0]
        try:
            tk, qd_s = base.split("_", 1)
            qd = datetime.date.fromisoformat(qd_s)
        except ValueError:
            continue

        # ...and then skips the items that are already "done"
        if (tk.upper(), qd) in done:
            continue

        # Publish work item for price_technicals job
        # The message can be simple, as price_technicals builds its own payload
        msg = {"ticker": tk, "quarter_end": qd_s}
        pub.publish(topic_path, json.dumps(msg).encode())
        logging.info(f"Publishing msg: {msg}")
        published += 1

    return f"Published {published} new items for price_technicals", 200