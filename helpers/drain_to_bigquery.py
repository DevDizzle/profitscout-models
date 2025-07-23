#!/usr/bin/env python3
"""
Drain all messages from Pub/Sub pull subscription 'bootstrap-sub'
and append them to profit_scout.breakout_features in ONE load job,
ignoring any JSON keys that don't exist in the table schema.
"""

import json
import pathlib
import sys
from time import sleep
from google.cloud import pubsub_v1, bigquery
from google.api_core.exceptions import DeadlineExceeded

# ─── CONFIG ─────────────────────────────────────────────────────────
PROJECT   = "profitscout-lx6bb"
SUB_NAME  = "bootstrap-sub"
TABLE     = "profit_scout.breakout_features"
NDJSON    = pathlib.Path("/tmp/bootstrap.json")

PULL_BATCH      = 1000
PULL_DEADLINE   = 30
MAX_EMPTY_POLLS = 30
MAX_RETRIES     = 5

# ─── INIT CLIENTS ───────────────────────────────────────────────────
subscriber = pubsub_v1.SubscriberClient()
bq_client  = bigquery.Client(project=PROJECT)
SUB_PATH   = f"projects/{PROJECT}/subscriptions/{SUB_NAME}"

# ─── PULL ALL MESSAGES ──────────────────────────────────────────────
rows, ack_ids = [], []
empty_polls   = 0
retries_left  = MAX_RETRIES

print(f"Draining messages from '{SUB_NAME}' …")
while True:
    try:
        response = subscriber.pull(
            subscription=SUB_PATH,
            max_messages=PULL_BATCH,
            timeout=PULL_DEADLINE,
        )
    except DeadlineExceeded:
        print(" • pull timed out, retrying …")
        retries_left -= 1
        if retries_left == 0:
            sys.exit("Too many timeouts – aborting.")
        continue

    if not response.received_messages:
        empty_polls += 1
        if empty_polls >= MAX_EMPTY_POLLS:
            break
        sleep(1)
        continue
    empty_polls  = 0
    retries_left = MAX_RETRIES

    for m in response.received_messages:
        rows.append(json.loads(m.message.data.decode("utf-8")))
        ack_ids.append(m.ack_id)

    subscriber.acknowledge(subscription=SUB_PATH, ack_ids=ack_ids)
    ack_ids.clear()
    print(f" • Pulled {len(rows)} total so far …")

print(f"Finished pulling – total messages: {len(rows)}")
if not rows:
    sys.exit("No messages found – exiting.")

# ─── WRITE NDJSON FILE ──────────────────────────────────────────────
NDJSON.write_text("\n".join(json.dumps(r) for r in rows))
print(f"Wrote NDJSON to {NDJSON}")

# ─── ONE BIGQUERY LOAD JOB (ignore unknown fields) ─────────────────
job = bq_client.load_table_from_file(
    NDJSON.open("rb"),
    TABLE,
    job_config=bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition="WRITE_APPEND",
        ignore_unknown_values=True,          # ← key line
    ),
)
job.result()
print(f"Loaded {job.output_rows} rows into {TABLE} ✔︎")
