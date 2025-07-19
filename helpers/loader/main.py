"""
Cloud Run Pub/Sub loader: decodes message(s) and loads rows directly into BigQuery
via load jobs (NOT legacy streaming). Deterministic & simple.
"""

import os
import json
import base64
import logging
from typing import List, Dict, Any

from flask import Flask, request, jsonify
from google.cloud import bigquery

# ───── Environment ─────────────────────────────────────────
PROJECT_ID = os.environ.get("PROJECT_ID", "profitscout-lx6bb")
DESTINATION_TABLE = os.environ.get("DESTINATION_TABLE")  # e.g. profit_scout.staging_breakout_features
REQUIRED_FIELDS = os.environ.get("REQUIRED_FIELDS", "ticker,quarter_end_date,earnings_call_date") \
    .split(",")  # minimal sanity; remove or edit as needed

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

if not DESTINATION_TABLE:
    raise RuntimeError("DESTINATION_TABLE environment variable is required.")

bq_client = bigquery.Client(project=PROJECT_ID)
app = Flask(__name__)


# ───── Helpers ─────────────────────────────────────────────
def decode_pubsub_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Decode a single Pub/Sub message dict (with base64 'data')."""
    data_b64 = message["data"]
    raw = base64.b64decode(data_b64).decode("utf-8")
    return json.loads(raw)


def validate_row(row: Dict[str, Any]) -> List[str]:
    """Return list of missing required fields (simple presence check)."""
    missing = [f.strip() for f in REQUIRED_FIELDS if f.strip() and row.get(f.strip()) in (None, "", [])]
    return missing


def load_rows(rows: List[Dict[str, Any]]) -> int:
    """
    Load rows into BigQuery using a load job (WRITE_APPEND).
    Returns number of rows successfully written (BigQuery reports output_rows).
    """
    # Fully qualified table ID is fine; table must already exist (recommended) or it will error.
    job = bq_client.load_table_from_json(
        rows,
        DESTINATION_TABLE,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    )
    result = job.result()  # Wait for completion
    return result.output_rows


# ───── Routes ──────────────────────────────────────────────
@app.route("/healthz", methods=["GET"])
def health():
    return jsonify(status="ok", table=DESTINATION_TABLE), 200


@app.route("/", methods=["POST"])
def ingest():
    """
    Accepts either:
      { "message": { "data": base64(json) } }
    or:
      { "messages": [ { "data": base64(json) }, ... ] }
    """
    envelope = request.get_json(silent=True)
    if not envelope:
        return "Bad Request: empty body", 400

    # Normalize to list of message dicts
    if "message" in envelope:
        messages = [envelope["message"]]
    elif "messages" in envelope:
        messages = envelope["messages"]
    else:
        return "Bad Request: invalid Pub/Sub envelope", 400

    valid_rows = []
    invalid = 0
    for msg in messages:
        try:
            row = decode_pubsub_message(msg)
            missing = validate_row(row)
            if missing:
                invalid += 1
                log.warning(f"Skipping row missing {missing}: {row}")
                continue
            valid_rows.append(row)
        except Exception as e:
            invalid += 1
            log.warning(f"Skipping malformed message: {e}")

    if not valid_rows:
        # Acknowledge anyway so Pub/Sub doesn't retry forever on bad data
        return jsonify(status="no_valid_rows", invalid=invalid), 200

    try:
        inserted = load_rows(valid_rows)
        log.info(f"Loaded {inserted} row(s). Skipped {invalid}.")
        return jsonify(status="ok", inserted=inserted, skipped=invalid), 200
    except Exception as e:
        log.error(f"Load job failed: {e}", exc_info=True)
        # Return 500 so Pub/Sub can retry (if you prefer ACK anyway, change to 200)
        return jsonify(status="error", error=str(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
