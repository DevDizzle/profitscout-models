# ───────────────────────── imports ─────────────────────────
import os, json, logging, re, requests, textwrap
import pandas as pd, pandas_ta as ta
from typing import Optional
from google.cloud import storage, aiplatform, language_v1, bigquery, secretmanager
from vertexai.language_models import TextEmbeddingModel
from . import bq

# ───────────────────────── init ────────────────────────────
PROJECT_ID = os.environ.get("PROJECT_ID")
aiplatform.init(project=PROJECT_ID, location="us-central1")

storage_client  = storage.Client()
language_client = language_v1.LanguageServiceClient()
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


SECTION_PATTERNS = {
    # matches:  optional spaces  (# ### etc.) OR (**)  then heading text
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
    
    # ADDED LOGGING
    logging.info("Parsed sections from summary markdown:")
    for k, v in sections.items():
        logging.info("  %-25s -> %4d chars", k, len(v.strip()))
        sample = textwrap.shorten(v.strip().replace("\n", " "), width=120) or "(blank)"
        logging.debug("    Sample: %s", sample) # Use logging.debug for more verbose output
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
            txt = txt.strip()
            if txt:
                logging.info("Generating embedding for section: %s", sec)
                # Note: Ensure the Gemini model you're using doesn't require the `task_type` parameter.
                # The "gemini-embedding-001" model does not.
                emb = embedding_model.get_embeddings([txt])[0]
                vec = list(emb.values)
                results[f"{sec}_embedding"] = vec
                
                # This is the detailed log you wanted to see
                logging.info("Embedded %s | len=%d | first5=%s", sec, len(vec), vec[:5])
            else:
                # This will clearly log if a section is being skipped
                logging.warning("Section '%s' is empty. Skipping embedding.", sec)
                results[f"{sec}_embedding"] = None

        return results
    except Exception as e:
        # Log the full error if something goes wrong during the process
        logging.error("get_embeddings function failed: %s", e, exc_info=True)
        return {}


def get_sentiment_and_date(transcript_gcs_path: str) -> dict:
    try:
        bucket, blob = transcript_gcs_path.removeprefix("gs://").split("/", 1)
        data = json.loads(storage_client.bucket(bucket).blob(blob).download_as_string())
        doc = language_v1.Document(content=data.get("content", ""),
                                   type_=language_v1.Document.Type.PLAIN_TEXT)
        score = language_client.analyze_sentiment(document=doc).document_sentiment.score
        return {"sentiment_score": score, "earnings_call_date": data.get("date")}
    except Exception as e:
        logging.error("get_sentiment_and_date: %s", e)
        return {}


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
    """
    Returns a dict shaped for BigQuery table
        eps_surprise  – FLOAT64  (relative surprise, e.g. 0.12 for +12 %)
        eps_missing   – BOOL

    Assumes table already has these two columns.
    """
    if not api_key:
        # no key ⇒ treat as missing
        return {"eps_surprise": None, "eps_missing": True}

    url = (f"https://financialmodelingprep.com/api/v3/"
           f"earnings-surprises/{ticker}?apikey={api_key}")
    try:
        data = requests.get(url, timeout=10).json()
        if not data:
            raise ValueError("empty response")

        actual   = data[0]["actualEarningResult"]
        estimate = data[0]["estimatedEarning"]

        # Absolute surprise
        surprise = actual - estimate

        # Normalised (% of estimate).  Handle zero-estimate edge case.
        surprise_pct = surprise / abs(estimate) if estimate else None

        return {"eps_surprise": surprise_pct, "eps_missing": False}

    except Exception as e:
        logging.error("get_eps_surprise: %s", e)
        return {"eps_surprise": None, "eps_missing": True}


# ───────────────── price features (unchanged) ─────────────────
def get_price_technicals(ticker: str, call_date: str) -> dict:
    try:
        price_table = os.environ["PRICE_TABLE"]
        call_date = call_date.split(" ")[0]
        query = f"""
        SELECT date, open, high, low, adj_close
        FROM `{price_table}`
        WHERE ticker = @ticker AND date <= @call_date
        ORDER BY date DESC
        LIMIT 100
        """
        df = (
            bq.bq_client.query(
                query,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                        bigquery.ScalarQueryParameter("call_date", "DATE", call_date),
                    ]
                ),
            )
            .to_dataframe()
            .set_index("date")
            .sort_index()
            .rename(columns={"open": "Open", "high": "High", "low": "Low", "adj_close": "Close"})
        )

        if df.empty:
            logging.warning("No price data for %s up to %s", ticker, call_date)
            return {}

        df["Adj Close"] = df["Close"]
        df.ta.sma(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.adx(length=14, append=True, high=df["High"], low=df["Low"], close=df["Close"])

        latest, prev = df.iloc[-1], df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        delta = lambda cur, pre: cur - pre if pd.notna(cur) and pd.notna(pre) else None

        return {
            "adj_close_on_call_date": latest["Adj Close"],
            "sma_20":      latest.get("SMA_20"),
            "ema_50":      latest.get("EMA_50"),
            "rsi_14":      latest.get("RSI_14"),
            "adx_14":      latest.get("ADX_14"),
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
        price_table = os.environ["PRICE_TABLE"]
        call_date = call_date.split(" ")[0]
        end_q = f"""
        SELECT MAX(date) AS effective_date
        FROM `{price_table}` WHERE ticker = @ticker AND date <= @call_date
        """
        end_df = bq.bq_client.query(
            end_q,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                    bigquery.ScalarQueryParameter("call_date", "DATE", call_date),
                ]
            ),
        ).to_dataframe()

        if end_df.empty or pd.isna(end_df.at[0, "effective_date"]):
            logging.warning("No trading day ≤ %s for %s", call_date, ticker)
            return {"max_close_30d": None}

        end_date = end_df.at[0, "effective_date"]
        start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

        max_q = f"""
        SELECT MAX(adj_close) AS max_price
        FROM `{price_table}`
        WHERE ticker=@ticker AND date BETWEEN @start_date AND @end_date
        """
        max_df = bq.bq_client.query(
            max_q,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                    bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
                    bigquery.ScalarQueryParameter("end_date", "DATE", str(end_date)),
                ]
            ),
        ).to_dataframe()

        return {"max_close_30d": max_df.at[0, "max_price"]} if not max_df.empty else {"max_close_30d": None}
    except Exception as e:
        logging.error("get_max_close_30d: %s", e)
        return {"max_close_30d": None}

# ─────────────────── orchestrator ─────────────────────────
def create_features(message: dict) -> dict | None:
    """
    Validates message, orchestrates feature creation, and returns a
    complete row for BigQuery.
    """
    # --- 1. Validate Input Message ---
    # Ensure the core GCS paths required for processing are present.
    required_keys = ["ticker", "quarter_end_date", "summary_gcs_path", "transcript_gcs_path"]
    if not all(key in message and message[key] is not None for key in required_keys):
        logging.error(f"FATAL: Message is missing one or more required keys or values. Message data: {message}")
        return None # Stop processing immediately

    # --- 2. Orchestrate Feature Generation ---
    ticker = message.get("ticker")
    row = {"ticker": ticker, "quarter_end_date": message.get("quarter_end_date")}

    row.update(get_stock_metadata(ticker))
    row.update(get_embeddings(message.get("summary_gcs_path")))
    row.update(get_sentiment_and_date(message.get("transcript_gcs_path")))
    row.update(get_eps_surprise(get_fmp_api_key(), ticker))

    if row.get("earnings_call_date"):
        row.update(get_price_technicals(ticker, row["earnings_call_date"]))
        row.update(get_max_close_30d(ticker, row["earnings_call_date"]))
    else:
        logging.warning("No earnings_call_date; skipped price features for ticker %s", ticker)

    # Final check to ensure critical data was generated
    if not {"earnings_call_date", "sentiment_score"}.issubset(row):
        logging.error(f"Failed to generate critical features (date/sentiment) for {ticker}. Aborting row.")
        return None

    return row
