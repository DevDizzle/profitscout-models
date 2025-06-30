import os
import json
import logging
import requests
import pandas as pd
import pandas_ta as ta
from google.cloud import storage, aiplatform, language_v1, bigquery, secretmanager
from vertexai.language_models import TextEmbeddingModel
from . import bq  # Correctly import the bq module

# Initialize clients
storage_client = storage.Client()
aiplatform.init(project=os.environ.get('PROJECT_ID'), location='us-central1')
embedding_model = TextEmbeddingModel("textembedding-gecko@001")
language_client = language_v1.LanguageServiceClient()
secret_client = secretmanager.SecretManagerServiceClient()

logging.basicConfig(level=logging.INFO)

def get_fmp_api_key():
    """Fetches the FMP API key from Secret Manager."""
    try:
        name = "projects/profitscout-lx6bb/secrets/FMP_API_KEY/versions/latest"
        response = secret_client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logging.error(f"Failed to fetch FMP_API_KEY from Secret Manager: {e}")
        return None

def _parse_sections(markdown_text):
    """Splits markdown text into predefined sections."""
    sections = {
        'key_financial_metrics': '',
        'key_discussion_points': '',
        'sentiment_tone': '',
        'short_term_outlook': '',
        'forward_looking_signals': ''
    }
    current_section = None
    for line in markdown_text.split('\n'):
        if line.startswith('###'):
            section_key = line.replace('###', '').strip().lower().replace(' ', '_')
            if section_key in sections:
                current_section = section_key
        elif current_section:
            sections[current_section] += line + '\n'
    return sections

def get_embeddings(summary_gcs_path: str) -> dict:
    """Downloads summary text and generates embeddings for each section."""
    try:
        bucket_name, blob_name = summary_gcs_path.replace("gs://", "").split("/", 1)
        blob = storage_client.bucket(bucket_name).blob(blob_name)
        summary_text = blob.download_as_string().decode('utf-8')
        
        sections = _parse_sections(summary_text)
        
        embedding_results = {}
        for section_name, text in sections.items():
            if text.strip():
                embeddings = embedding_model.get_embeddings([text])
                embedding_results[f"{section_name}_embedding"] = [v for v in embeddings[0].values]
            else:
                embedding_results[f"{section_name}_embedding"] = None
        
        logging.info("Successfully generated embeddings.")
        return embedding_results
    except Exception as e:
        logging.error(f"Failed to get embeddings: {e}")
        return {}


def get_sentiment_and_date(transcript_gcs_path: str) -> dict:
    """Downloads full transcript, gets sentiment score and earnings call date."""
    try:
        bucket_name, blob_name = transcript_gcs_path.replace("gs://", "").split("/", 1)
        blob = storage_client.bucket(bucket_name).blob(blob_name)
        content = blob.download_as_string().decode('utf-8')
        data = json.loads(content)
        
        document = language_v1.Document(content=data.get('content', ''), type_=language_v1.Document.Type.PLAIN_TEXT)
        sentiment = language_client.analyze_sentiment(document=document).document_sentiment
        
        logging.info("Successfully generated sentiment.")
        return {
            "sentiment_score": sentiment.score,
            "earnings_call_date": data.get('date')
        }
    except Exception as e:
        logging.error(f"Failed to get sentiment: {e}")
        return {}


def get_eps_surprise(fmp_api_key: str, ticker: str) -> dict:
    """Fetches EPS surprise from FinancialModelingPrep API."""
    if not fmp_api_key:
        logging.error("FMP API Key is missing, cannot fetch EPS surprise.")
        return {"eps_surprise": None, "eps_missing": True}
    try:
        url = f"https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}?apikey={fmp_api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data:
            return {"eps_surprise": data[0].get('actualEarningResult'), "eps_missing": False}
        
        logging.warning(f"No EPS data found for {ticker}")
        return {"eps_surprise": None, "eps_missing": True}
    except Exception as e:
        logging.error(f"Failed to get EPS surprise for {ticker}: {e}")
        return {"eps_surprise": None, "eps_missing": True}

def get_price_technicals(ticker: str, earnings_call_date: str) -> dict:
    """Queries BigQuery for price data and calculates technical indicators."""
    try:
        price_table = os.environ.get('PRICE_TABLE')
        call_date_only = earnings_call_date.split(' ')[0]

        query = f"""
        SELECT date, adj_close FROM `{price_table}`
        WHERE ticker = @ticker AND date <= @call_date
        ORDER BY date DESC
        LIMIT 100 
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                bigquery.ScalarQueryParameter("call_date", "DATE", call_date_only),
            ]
        )
        df = bq.bq_client.query(query, job_config=job_config).to_dataframe().set_index('date').sort_index()

        if df.empty:
            logging.warning(f"No price data found for {ticker} on or before {call_date_only}")
            return {}

        df.ta.sma(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.adx(length=14, append=True)

        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest

        # FIX #2: Safely calculate deltas only if both values are not None
        def safe_delta(latest_val, prev_val):
            return latest_val - prev_val if latest_val is not None and prev_val is not None else None

        return {
            "adj_close_on_call_date": latest['adj_close'],
            "sma_20": latest.get('SMA_20'),
            "ema_50": latest.get('EMA_50'),
            "rsi_14": latest.get('RSI_14'),
            "adx_14": latest.get('ADX_14'),
            "sma_20_delta": safe_delta(latest.get('SMA_20'), previous.get('SMA_20')),
            "ema_50_delta": safe_delta(latest.get('EMA_50'), previous.get('EMA_50')),
            "rsi_14_delta": safe_delta(latest.get('RSI_14'), previous.get('RSI_14')),
            "adx_14_delta": safe_delta(latest.get('ADX_14'), previous.get('ADX_14')),
        }
    except Exception as e:
        logging.error(f"Failed to get price technicals for {ticker}: {e}")
        return {}
def create_features(message: dict) -> dict:
    """Main orchestration function to generate all features for a given earnings call."""
    fmp_api_key = get_fmp_api_key()
    ticker = message.get("ticker")
    quarter_end_date = message.get("quarter_end_date")
    summary_gcs_path = message.get("summary_gcs_path")
    transcript_gcs_path = message.get("transcript_gcs_path")

    # Start building the final row
    final_row = {"ticker": ticker, "quarter_end_date": quarter_end_date}

    # --- Execute feature generation steps ---
    embedding_data = get_embeddings(summary_gcs_path)
    sentiment_data = get_sentiment_and_date(transcript_gcs_path)
    eps_data = get_eps_surprise(fmp_api_key, ticker)

    final_row.update(embedding_data)
    final_row.update(sentiment_data)
    final_row.update(eps_data)

    # Price technicals depend on the earnings_call_date from sentiment_data
    if final_row.get("earnings_call_date"):
        technicals_data = get_price_technicals(ticker, final_row["earnings_call_date"])
        final_row.update(technicals_data)
    else:
        logging.warning("Skipping price technicals due to missing earnings_call_date.")

    # You can add logic here to check for essential fields before returning
    if not all(k in final_row for k in ["earnings_call_date", "sentiment_score"]):
        logging.error("Missing critical data, returning None.")
        return None

    return final_row