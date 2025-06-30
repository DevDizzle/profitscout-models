# ───────────────────────── imports ─────────────────────────
import os, json, logging, re, requests
import pandas as pd, pandas_ta as ta
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
    "key_financial_metrics"  : re.compile(r"^#{1,6}\s*Key Financial Metrics\s*:?",  re.I),
    "key_discussion_points"  : re.compile(r"^#{1,6}\s*Key Discussion Points\s*:?",  re.I),
    "sentiment_tone"         : re.compile(r"^#{1,6}\s*Sentiment Tone\s*:?",         re.I),
    "short_term_outlook"     : re.compile(r"^#{1,6}\s*Short\-?Term Outlook\s*:?",   re.I),
    "forward_looking_signals": re.compile(r"^#{1,6}\s*Forward\-?Looking Signals\s*:?", re.I),
}

def _parse_sections(md: str) -> dict:
    sections = {k: "" for k in SECTION_PATTERNS}
    current  = None
    for line in md.splitlines():
        hit = next((k for k, rx in SECTION_PATTERNS.items() if rx.match(line.strip())), None)
        current = hit or current
        if current and hit is None:
            sections[current] += line + "\n"
    return sections


def get_embeddings(summary_gcs_path: str) -> dict:
    """Download summary markdown, embed each section, INFO-log vector length & sample."""
    try:
        bucket, blob = summary_gcs_path.removeprefix("gs://").split("/", 1)
        text = storage_client.bucket(bucket).blob(blob).download_as_text()
        sections, results = _parse_sections(text), {}

        for sec, txt in sections.items():
            if txt.strip():
                emb = embedding_model.get_embeddings([txt])[0]     # ← no task_type
                vec = list(emb.values)
                results[f"{sec}_embedding"] = vec

                # sanity log
                logging.info("Embedded %s | len=%d | first5=%s", sec, len(vec), vec[:5])
            else:
                results[f"{sec}_embedding"] = None

        return results
    except Exception as e:
        logging.error("get_embeddings: %s", e)
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


def get_eps_surprise(api_key: str | None, ticker: str) -> dict:
    if not api_key:
        return {"eps_surprise": None, "eps_missing": True}
    try:
        url = f"https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}?apikey={api_key}"
        data = requests.get(url, timeout=10).json()
        return {"eps_surprise": data[0]["actualEarningResult"], "eps_missing": False} if data else {"eps_surprise": None, "eps_missing": True}
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
        logging.warning("No earnings_call_date; skipped price features")

    return row if {"earnings_call_date", "sentiment_score"}.issubset(row) else None
