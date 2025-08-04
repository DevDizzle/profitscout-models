import os, json, base64, logging
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from google.cloud import bigquery

PROJECT_ID = os.environ.get("PROJECT_ID", "profitscout-lx6bb")
DESTINATION_TABLE = os.environ.get("DESTINATION_TABLE")  # project.dataset.table
REQUIRED_FIELDS = os.environ.get("REQUIRED_FIELDS", "ticker,quarter_end_date,earnings_call_date").split(",")

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

if not DESTINATION_TABLE:
    raise RuntimeError("DESTINATION_TABLE env var is required.")

bq = bigquery.Client(project=PROJECT_ID)
app = Flask(__name__)


def decode_pubsub_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    raw = base64.b64decode(msg["data"]).decode("utf-8")
    return json.loads(raw)

def validate_row(row: Dict[str, Any]) -> List[str]:
    return [f.strip() for f in REQUIRED_FIELDS if f.strip() and row.get(f.strip()) in (None, "", [])]

def load_rows(rows: List[Dict[str, Any]]) -> int:
    """Stream rows with insert_rows_json."""
    # rows must be list[dict]; BigQuery will auto-map to schema by key
    errors = bq.insert_rows_json(DESTINATION_TABLE, rows, row_ids=[None]*len(rows))
    if errors:
        raise RuntimeError(f"Row insert errors: {errors}")
    return len(rows)


@app.route("/healthz", methods=["GET"])
def health():
    return jsonify(status="ok", table=DESTINATION_TABLE), 200


@app.route("/", methods=["POST"])
def ingest():
    envelope = request.get_json(silent=True)
    if not envelope:
        return "Bad Request: empty body", 400

    if "message" in envelope:
        messages = [envelope["message"]]
    elif "messages" in envelope:
        messages = envelope["messages"]
    else:
        return "Bad Request: invalid Pub/Sub envelope", 400

    valid_rows, invalid = [], 0
    for msg in messages:
        try:
            row = decode_pubsub_message(msg)
            missing = validate_row(row)
            if missing:
                invalid += 1
                log.warning("Skipping row missing %s: %s", missing, row)
                continue
            valid_rows.append(row)
        except Exception as e:
            invalid += 1
            log.warning("Skipping malformed message: %s", e)

    if not valid_rows:
        return jsonify(status="no_valid_rows", invalid=invalid), 200

    try:
        inserted = load_rows(valid_rows)
        log.info("Loaded %d row(s). Skipped %d.", inserted, invalid)
        return jsonify(status="ok", inserted=inserted, skipped=invalid), 200
    except Exception as e:
        log.error("Streaming insert failed: %s", e, exc_info=True)
        return jsonify(status="error", error=str(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)