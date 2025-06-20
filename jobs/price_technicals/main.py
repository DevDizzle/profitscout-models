#!/usr/bin/env python3
"""
Job 3 – price_technicals
────────────────────────
• Fetch SMA-20, EMA-50, RSI-14, ADX-14 from FMP
• Compute 90-day deltas
• UPSERT cols into profit_scout.breakout_features
• Publish the untouched message to `eps-todo`
"""

import base64, json, logging, os, time
from datetime import timedelta, date
from typing import Any, Dict

import pandas as pd
import requests
from dateutil.parser import parse
from flask import Flask, request
from tenacity import retry, stop_after_attempt, wait_exponential

from utils import bq_client, table_ref, upsert_json

# ──────────────────── CONFIG ──────────────────────────
PROJECT_ID   = os.getenv("PROJECT_ID", "profitscout-lx6bb")
DATASET      = os.getenv("BQ_DATASET",  "profit_scout")
TABLE        = table_ref(DATASET, "breakout_features", PROJECT_ID)

FMP_KEY      = os.environ["FMP_API_KEY"]          # must be set!
WINDOW_DAYS  = int(os.getenv("WINDOW_DAYS", 90))  # pct-∆ window
TOPIC_OUT    = os.getenv("TOPIC_OUT", "eps-todo") # next hop

INDICATORS = {                                   # label → spec
    "sma_20": {"type": "sma", "field": "sma", "period": 20},
    "ema_50": {"type": "ema", "field": "ema", "period": 50},
    "rsi_14": {"type": "rsi", "field": "rsi", "period": 14},
    "adx_14": {"type": "adx", "field": "adx", "period": 14},
}

# ──────────────────── GLOBALS ─────────────────────────
bq        = bq_client(PROJECT_ID)
publisher = None  # will be lazy-imported to avoid heavy client at cold-start
app       = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ──────────────────── HELPERS ─────────────────────────
class Limiter:
    """≤ 45 requests / 60 s (FMP free-tier limit)."""
    def __init__(self, r=45, p=60):
        self.r, self.p, self.ts = r, p, []

    def hit(self):
        now = time.time()
        self.ts = [t for t in self.ts if now - t < self.p]
        if len(self.ts) >= self.r:
            time.sleep(self.p - (now - self.ts[0]))
        self.ts.append(now)

lim = Limiter()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _get(url: str) -> Any:
    lim.hit()
    r = requests.get(f"{url}&apikey={FMP_KEY}", timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_indicator(sym: str, typ: str, period: int | None = None):
    url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{sym}?type={typ}"
    if period:
        url += f"&period={period}"
    return _get(url)

def nearest(series, target_dt: date, field: str):
    """Latest ≤ target_dt where field is not null."""
    vals = [(parse(r["date"]).date(), r.get(field)) for r in series if r.get(field) is not None]
    vals = [v for v in vals if v[0] <= target_dt]
    return max(vals, default=(None, None))[1]

def pct_delta(series, target_dt: date, field: str, days: int):
    v_now = nearest(series, target_dt, field)
    v_prev = nearest(series, target_dt - timedelta(days=days), field)
    if v_now is None or v_prev is None or abs(v_prev) < 1e-12:
        return None
    return (v_now - v_prev) / abs(v_prev)

# ──────────────────── CORE ────────────────────────────
def process(msg: Dict[str, Any]):
    """msg keys expected: ticker, quarter_end, earnings_call_date (optional)."""
    ticker = msg["ticker"].upper()
    call_d = parse(msg.get("earnings_call_date") or msg["quarter_end"]).date()

    # 1) pull all four indicators in parallel
    ind_json = {
        k: fetch_indicator(ticker, spec["type"], spec.get("period"))
        for k, spec in INDICATORS.items()
    }

    # 2) build row
    row: Dict[str, Any] = {"ticker": ticker, "quarter_end_date": msg["quarter_end"]}
    for lbl, spec in INDICATORS.items():
        fld = spec["field"]
        row[lbl]             = nearest(ind_json[lbl], call_d, fld)
        row[f"{lbl}_delta"]  = pct_delta(ind_json[lbl], call_d, fld, WINDOW_DAYS)

    # 3) UPSERT
    upsert_json(TABLE, row, bq)
    logging.info("Updated technicals for %s %s", ticker, msg["quarter_end"])

    # 4) hand off to EPS job
    global publisher
    if publisher is None:
        from google.cloud import pubsub_v1
        publisher = pubsub_v1.PublisherClient()
    publisher.publish(f"projects/{PROJECT_ID}/topics/{TOPIC_OUT}",
                      json.dumps(msg).encode())

# ──────────────────── FLASK HANDLER ───────────────────
@app.route("/", methods=["POST"])
def handle():
    envelope = request.get_json()
    if not envelope or "message" not in envelope:
        return ("Bad Request", 400)

    payload = json.loads(base64.b64decode(envelope["message"]["data"]))
    process(payload)
    return ("", 204)

# local dev:  python main.py AAPL 2025-03-31 2025-04-25
if __name__ == "__main__":
    import sys
    process({
        "ticker":        sys.argv[1],
        "quarter_end":   sys.argv[2],
        "earnings_call_date": sys.argv[3] if len(sys.argv) > 3 else sys.argv[2],
    })
