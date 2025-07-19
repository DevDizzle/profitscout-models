# ───────────────────────── imports ─────────────────────────
import os, json, logging, re, requests, textwrap
import pandas as pd, pandas_ta as ta
import numpy as np
from numpy.linalg import norm
from typing import Optional
from google.cloud import storage, aiplatform, bigquery, secretmanager
from vertexai.language_models import TextEmbeddingModel
from genai_client import generate 
from . import bq

# ───────────────────────── init ────────────────────────────
PROJECT_ID = os.environ.get("PROJECT_ID")
aiplatform.init(project=PROJECT_ID, location="us-central1")

storage_client  = storage.Client()
secret_client   = secretmanager.SecretManagerServiceClient()
embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

logging.basicConfig(level=logging.INFO)

# ─────────────────────── helpers ───────────────────────────
def get_fmp_api_key() -> str | None:
    try:
        name = f"projects/{PROJECT_ID}/secrets/FMP_API_KEY/versions/latest"
        return secret_client.access_secret_version(request={"name": name}).payload.data.decode()
    except Exception as e:
        logging.error("get_fmp_api_key: %s", e)
        return None

def safe_cosine(u, v):
    """Calculate cosine similarity, handling potential zero vectors."""
    u, v = np.array(u), np.array(v)
    if u.ndim == 1: u = u.reshape(1, -1)
    if v.ndim == 1: v = v.reshape(1, -1)
    
    den = norm(u, axis=1) * norm(v, axis=1)
    if den == 0:
        return 0.0
    
    num = (u * v).sum(axis=1)
    return (num / den)[0]


SECTION_PATTERNS = {
    'key_financial_metrics'  : re.compile(r'^\s*(?:#{1,6}\s*|\*\*)\s*Key Financial Metrics\b.*',  re.I),
    'key_discussion_points'  : re.compile(r'^\s*(?:#{1,6}\s*|\*\*)\s*Key Discussion Points\b.*',  re.I),
    'sentiment_tone'         : re.compile(r'^\s*(?:#{1,6}\s*|\*\*)\s*Sentiment Tone\b.*',         re.I),
    'short_term_outlook'     : re.compile(r'^\s*(?:#{1,6}\s*|\*\*)\s*Short\-?Term Outlook\b.*',   re.I),
    'forward_looking_signals': re.compile(r'^\s*(?:#{1,6}\s*|\*\*)\s*Forward\-?Looking Signals\b.*', re.I),
}

def _parse_sections(md: str) -> dict:
    """Return a dict of section_name → text."""
    sections = {k: "" for k in SECTION_PATTERNS}
    current  = None
    for line in md.splitlines():
        hit = next((k for k, rx in SECTION_PATTERNS.items() if rx.match(line.strip())), None)
        current = hit or current
        if current and hit is None:
            sections[current] += line + "\n"
    
    logging.info("Parsed sections from summary markdown:")
    for k, v in sections.items():
        logging.info("  %-25s -> %4d chars", k, len(v.strip()))
        sample = textwrap.shorten(v.strip().replace("\n", " "), width=120) or "(blank)"
        logging.debug("    Sample: %s", sample)
    return sections


def get_embeddings(summary_gcs_path: str) -> dict:
    """Download summary markdown, embed each section, INFO-log vector length & sample."""
    logging.info("Starting embedding generation for: %s", summary_gcs_path)
    try:
        bucket, blob = summary_gcs_path.removeprefix("gs://").split("/", 1)
        text = storage_client.bucket(bucket).blob(blob).download_as_text()
        logging.info("Successfully downloaded summary file.")
        
        sections, results = _parse_sections(text), {}

        for sec, txt in sections.items():
            txt = (txt or "").strip()
            key = f"{sec}_embedding"

            if txt:
                logging.info("Generating embedding for section: %s", sec)
                emb = embedding_model.get_embeddings(
                    [txt],
                    output_dimensionality=768
                )[0]

                vec = list(emb.values)
                vec_np = np.array(vec, dtype=float)

                # L2 normalize
                norm = np.linalg.norm(vec_np)
                if norm > 0:
                    vec_np = vec_np / norm

                results[key] = vec_np.tolist()
                logging.info(
                    "Embedded %s | len=%d | first5=%s",
                    sec, len(vec_np), vec_np[:5]
                )
            else:
                logging.warning("Section '%s' is empty. Skipping embedding.", sec)
                results[key] = None

        return results
    except Exception as e:
        logging.error("get_embeddings function failed: %s", e, exc_info=True)
        return {}

def _flash_score(summary_text: str) -> float:
    prompt = (
    "You are a sell-side equity analyst.  Evaluate the MANAGEMENT sentiment in the "
    "earnings-call summary below, taking into account:\n"
    "  • Narrative language and confident / cautious phrasing\n"
    "  • Direction and magnitude of the numeric results (revenue, EPS, margins, guidance)\n"
    "  • Surprises vs. expectations (beats / misses) and any forward-looking statements\n\n"
    "Convert your judgment into **one floating-point number** between −1.000 and 1.000, "
    "inclusive, with **exactly three digits after the decimal point**. "
    "Interpretation: −1.000 = strongly negative, 0.000 = neutral, 1.000 = strongly positive.\n"
    "Valid formats: -0.732   0.000   0.415\n\n"
    "⚠️  Output STRICTLY the number only – no words, symbols, or explanations.\n\n"
    "### Earnings-Call Summary\n"
    f"{summary_text}"
)
    return float(generate(prompt))

def get_sentiment_score(summary_gcs_path: str) -> dict:
    try:
        bucket, blob = summary_gcs_path.removeprefix("gs://").split("/", 1)
        text = storage_client.bucket(bucket).blob(blob).download_as_text()
        if not text:
            logging.warning("Empty summary at %s", summary_gcs_path)
            return {"sentiment_score": None}

        return {"sentiment_score": _flash_score(text)}

    except Exception as e:
        logging.error("get_sentiment_score: %s", e, exc_info=True)
        return {"sentiment_score": None}

def get_earnings_call_date(transcript_gcs_path: str) -> dict:
    """Reads the original transcript JSON *only* to pull the 'date' field."""
    try:
        bucket, blob = transcript_gcs_path.removeprefix("gs://").split("/", 1)
        raw = storage_client.bucket(bucket).blob(blob).download_as_string()
        date_str = json.loads(raw).get("date")
        if date_str:
            date_str = date_str.split(" ")[0]
        return {"earnings_call_date": date_str}

    except Exception as e:
        logging.error("get_earnings_call_date: %s", e, exc_info=True)
        return {"earnings_call_date": None}

def get_stock_metadata(ticker: str) -> dict:
    try:
        table = os.getenv("METADATA_TABLE", "profitscout-lx6bb.profit_scout.stock_metadata")
        query = f"SELECT sector, industry FROM `{table}` WHERE ticker = @ticker LIMIT 1"
        df = bq.bq_client.query(
            query,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
            ),
        ).to_dataframe()
        return df.iloc[0].to_dict() if not df.empty else {"sector": None, "industry": None}
    except Exception as e:
        logging.error("get_stock_metadata: %s", e)
        return {"sector": None, "industry": None}

def get_eps_surprise(api_key: Optional[str], ticker: str) -> dict:
    if not api_key:
        return {"eps_surprise": None, "eps_missing": True}

    url = (f"https://financialmodelingprep.com/api/v3/"
           f"earnings-surprises/{ticker}?apikey={api_key}")
    try:
        data = requests.get(url, timeout=10).json()
        if not data:
            raise ValueError("empty response")
        actual, estimate = data[0]["actualEarningResult"], data[0]["estimatedEarning"]
        surprise = actual - estimate
        surprise_pct = surprise / abs(estimate) if estimate else None
        return {"eps_surprise": surprise_pct, "eps_missing": False}
    except Exception as e:
        logging.error("get_eps_surprise: %s", e)
        return {"eps_surprise": None, "eps_missing": True}

def get_price_technicals(ticker: str, call_date: str) -> dict:
    try:
        price_table = os.environ["PRICE_TABLE"]
        call_date = call_date.split(" ")[0]
        query = f"SELECT date, open, high, low, adj_close FROM `{price_table}` WHERE ticker = @ticker AND date <= @call_date ORDER BY date DESC LIMIT 100"
        df = (
            bq.bq_client.query(
                query,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                        bigquery.ScalarQueryParameter("call_date", "DATE", call_date),
                    ]
                ),
            ).to_dataframe().set_index("date").sort_index()
            .rename(columns={"open": "Open", "high": "High", "low": "Low", "adj_close": "Close"})
        )
        if df.empty: return {}

        df["Adj Close"] = df["Close"]
        df.ta.sma(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.adx(length=14, append=True, high=df["High"], low=df["Low"], close=df["Close"])

        latest, prev = df.iloc[-1], df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        delta = lambda cur, pre: cur - pre if pd.notna(cur) and pd.notna(pre) else None

        return {
            "adj_close_on_call_date": latest["Adj Close"], "sma_20": latest.get("SMA_20"),
            "ema_50": latest.get("EMA_50"), "rsi_14": latest.get("RSI_14"), "adx_14": latest.get("ADX_14"),
            "sma_20_delta": delta(latest.get("SMA_20"), prev.get("SMA_20")),
            "ema_50_delta": delta(latest.get("EMA_50"), prev.get("EMA_50")),
            "rsi_14_delta": delta(latest.get("RSI_14"), prev.get("RSI_14")),
            "adx_14_delta": delta(latest.get("ADX_14"), prev.get("ADX_14")),
        }
    except Exception as e:
        logging.error("get_price_technicals: %s", e)
        return {}

def get_max_close_30d(ticker: str, call_date: str) -> dict:
    try:
        start_date_ts = pd.Timestamp(call_date.split(" ")[0])
        end_date_window_ts = start_date_ts + pd.Timedelta(days=30)
        today_ts = pd.Timestamp.now().floor('D')
        if today_ts < end_date_window_ts:
            return {"max_close_30d": None}

        price_table = os.environ["PRICE_TABLE"]
        max_q = f"SELECT MAX(adj_close) AS max_price FROM `{price_table}` WHERE ticker=@ticker AND date BETWEEN @start_date AND @end_date"
        max_df = bq.bq_client.query(
            max_q,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                    bigquery.ScalarQueryParameter("start_date", "DATE", start_date_ts.strftime('%Y-%m-%d')),
                    bigquery.ScalarQueryParameter("end_date", "DATE", end_date_window_ts.strftime('%Y-%m-%d')),
                ]
            )
        ).to_dataframe()
        return {"max_close_30d": max_df.at[0, "max_price"]} if not max_df.empty else {"max_close_30d": None}
    except Exception as e:
        logging.error("get_max_close_30d failed for ticker %s: %s", ticker, e)
        return {"max_close_30d": None}

def create_features(message: dict) -> dict | None:
    """Orchestrates feature creation and returns a complete row for BigQuery."""
    required_keys = ["ticker", "quarter_end_date", "summary_gcs_path", "transcript_gcs_path"]
    if not all(key in message and message[key] is not None for key in required_keys):
        logging.error(f"FATAL: Message missing required keys: {message}")
        return None

    ticker = message.get("ticker")
    row = {"ticker": ticker, "quarter_end_date": message.get("quarter_end_date")}

    row.update(get_stock_metadata(ticker))
    row.update(get_embeddings(message.get("summary_gcs_path")))
    row.update(get_sentiment_score(message["summary_gcs_path"]))
    row.update(get_earnings_call_date(message["transcript_gcs_path"]))
    row.update(get_eps_surprise(get_fmp_api_key(), ticker))

    if row.get("earnings_call_date"):
        row["earnings_call_date"] = row["earnings_call_date"].split(" ")[0]
        row.update(get_price_technicals(ticker, row["earnings_call_date"]))
        row.update(get_max_close_30d(ticker, row["earnings_call_date"]))
    else:
        logging.warning("No earnings_call_date; skipped price features for ticker %s", ticker)
    
    # --- [NEW] Engineered Features (Single Row) ---
    # This is the only new block of code. Everything else is from your original file.
    logging.info(f"Engineering new single-row features for {ticker}...")
    
    # Combined categorical feature
    sector = row.get("sector", "UNK_SEC") or "UNK_SEC"
    industry = row.get("industry", "UNK_IND") or "UNK_IND"
    row["industry_sector"] = f"{sector}_{industry}"

    # Embedding stats and similarities
    EMB_COLS_NAMES = [
        "key_financial_metrics_embedding", "key_discussion_points_embedding",
        "sentiment_tone_embedding", "short_term_outlook_embedding",
        "forward_looking_signals_embedding",
    ]
    vecs = {c: np.array(row[c]) for c in EMB_COLS_NAMES if row.get(c) is not None}
    
    for col_name, vec in vecs.items():
        base_name = col_name.replace("_embedding", "")
        row[f"{base_name}_norm"] = norm(vec) if vec.size > 0 else 0.0
        row[f"{base_name}_mean"] = vec.mean() if vec.size > 0 else 0.0
        row[f"{base_name}_std"] = vec.std() if vec.size > 0 else 0.0

    sim_pairs = [
        ("cos_fin_disc", "key_financial_metrics_embedding", "key_discussion_points_embedding"),
        ("cos_fin_tone", "key_financial_metrics_embedding", "sentiment_tone_embedding"),
        ("cos_disc_short", "key_discussion_points_embedding", "short_term_outlook_embedding"),
        ("cos_short_fwd", "short_term_outlook_embedding", "forward_looking_signals_embedding"),
    ]
    for key, col1, col2 in sim_pairs:
        row[key] = safe_cosine(vecs.get(col1, []), vecs.get(col2, []))

    # Null indicator for EPS
    row["eps_surprise_isnull"] = 1 if row.get("eps_surprise") is None else 0

    # Price and technical ratios
    if row.get("adj_close_on_call_date") and row.get("sma_20") and row.get("sma_20") != 0:
        row["price_sma20_ratio"] = row["adj_close_on_call_date"] / row["sma_20"]
    if row.get("ema_50") and row.get("sma_20") and row.get("sma_20") != 0:
        row["ema50_sma20_ratio"] = row["ema_50"] / row["sma_20"]
    if row.get("rsi_14") is not None:
        row["rsi_centered"] = row["rsi_14"] - 50
        if row.get("sentiment_score") is not None:
            row["sent_rsi"] = row["sentiment_score"] * row["rsi_centered"]
    if row.get("adx_14") is not None:
        row["adx_log1p"] = np.log1p(row["adx_14"])
    # --- [END OF NEW BLOCK] ---

    if not {"earnings_call_date", "sentiment_score"}.issubset(row):
        logging.error(f"Failed to generate critical features for {ticker}. Aborting row.")
        return None
    return row
