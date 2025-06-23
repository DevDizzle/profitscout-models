import os
import time
import io  # Added missing import
import requests
import pandas as pd
from google.cloud import bigquery, storage
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Config ──────────────────────────────────────────────────────────────
PROJECT_ID = os.getenv("PROJECT_ID", "profitscout-lx6bb")
DATASET = os.getenv("BQ_DATASET", "profit_scout")
TABLE = os.getenv("MASTER_TABLE", "stock_metadata")
TICKER_XLSX_GCS = os.getenv("TICKER_XLSX_GCS", "gs://profit-scout-data/tickerlist.xlsx")
FMP_KEY = os.environ["FMP_API_KEY"]
MAX_RETRIES = 3

bq = bigquery.Client(project=PROJECT_ID)
gcs = storage.Client(project=PROJECT_ID)

def load_ticker_list(path: str) -> pd.DataFrame:
    try:
        bucket_name, blob_path = path.replace("gs://", "").split("/", 1)
        buf = io.BytesIO()
        gcs.bucket(bucket_name).blob(blob_path).download_to_file(buf)
        buf.seek(0)
        df = pd.read_excel(buf, header=0)
        logging.info(f"Loaded {len(df)} rows from {path} with columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logging.error(f"Failed to load ticker list from {path}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_json(url: str) -> list|dict:
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def fetch_quarterly_ends(sym: str) -> pd.DataFrame:
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{sym}?period=quarter&limit=12&apikey={FMP_KEY}"
    try:
        data = get_json(url)
        logging.debug(f"Income statement response for {sym}: {data[:1] if data else 'Empty'}")
        if not data:
            logging.warning(f"No income statement data for {sym}. API returned empty response.")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if df.empty or 'date' not in df.columns:
            logging.warning(f"No 'date' column or empty data for {sym}. Columns: {df.columns if not df.empty else 'empty'}")
            return pd.DataFrame()
        return df[["date"]].rename(columns={"date": "quarter_end_date"})
    except Exception as e:
        logging.warning(f"Failed to fetch quarterly ends for {sym}: {e}")
        return pd.DataFrame()

def fetch_earnings_calls(sym: str) -> pd.DataFrame:
    url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{sym}?limit=12&apikey={FMP_KEY}"
    try:
        data = get_json(url)
        logging.debug(f"Earnings calendar response for {sym}: {data[:1] if data else 'Empty'}")
        if not data:
            logging.warning(f"No earnings data for {sym}. API returned empty response.")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if df.empty or 'date' not in df.columns:
            logging.warning(f"No 'date' column or empty data for {sym} earnings. Columns: {df.columns if not df.empty else 'empty'}")
            return pd.DataFrame()
        return df[["date"]].rename(columns={"date": "earnings_call_date"})
    except Exception as e:
        logging.warning(f"Failed to fetch earnings calls for {sym}: {e}")
        return pd.DataFrame()

def refresh_stock_metadata(request):
    # Load the full ticker list with all columns
    ticker_df = load_ticker_list(TICKER_XLSX_GCS)
    if ticker_df.empty:
        logging.error("Ticker list DataFrame is empty, aborting.")
        return "Failed to load ticker list", 500
    tickers = ticker_df["Ticker"].dropna().str.upper().str.strip().tolist()
    logging.info(f"🔍 Processing {len(tickers)} tickers from tickerlist.xlsx")

    # Fetch quarterly ends and earnings calls
    quarterly_rows = []
    for tkr in tickers:
        try:
            logging.info(f"Processing ticker: {tkr}")
            qs = fetch_quarterly_ends(tkr)
            if not qs.empty:
                ecs = fetch_earnings_calls(tkr)
                if ecs.empty:
                    logging.warning(f"No earnings data for {tkr}, using income data only.")
                    ecs = pd.DataFrame({"earnings_call_date": [None] * len(qs)})
                # Align by minimum length
                min_len = min(len(qs), len(ecs))
                qs = qs.head(min_len)
                ecs = ecs.head(min_len)
                merged = pd.concat([qs, ecs], axis=1)
                merged["ticker"] = tkr
                # Merge with ticker_df to include company_name, sector, industry
                ticker_info = ticker_df[ticker_df["Ticker"] == tkr].iloc[0]
                merged["company_name"] = ticker_info["Company Name"]
                merged["industry"] = ticker_info["Industry"]
                merged["sector"] = ticker_info["Sector"]
                quarterly_rows.append(merged)
            else:
                logging.warning(f"No quarterly data for {tkr}, skipping.")
        except Exception as e:
            logging.error(f"Error processing {tkr}: {e}")
            continue
        time.sleep(0.25)  # Respect API rate limits

    if not quarterly_rows:
        logging.error("No data collected for any ticker.")
        return "No data to process", 400

    quarters = pd.concat(quarterly_rows, ignore_index=True)
    logging.info(f"Collected data for {len(quarters)} quarters across {len(set(quarters['ticker']))} tickers.")

    # Convert dates to datetime, handle None values
    quarters['quarter_end_date'] = pd.to_datetime(quarters['quarter_end_date'], errors='coerce').dt.date
    quarters['earnings_call_date'] = pd.to_datetime(quarters['earnings_call_date'], errors='coerce').dt.date

    # Prepare final DataFrame with explicit schema
    df_fresh = quarters[["ticker", "company_name", "industry", "sector", "quarter_end_date", "earnings_call_date"]]
    schema = [
        bigquery.SchemaField("ticker", "STRING"),
        bigquery.SchemaField("company_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("industry", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("sector", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("quarter_end_date", "DATE"),
        bigquery.SchemaField("earnings_call_date", "DATE", mode="NULLABLE")
    ]

    # Load into temporary table with explicit schema
    temp_table_id = f"{PROJECT_ID}.{DATASET}.{TABLE}_temp"
    job_cfg = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", schema=schema)
    job = bq.load_table_from_dataframe(df_fresh, temp_table_id, job_config=job_cfg)
    job.result()
    logging.info(f"Loaded {len(df_fresh)} rows to temporary table {temp_table_id}")

    # Create or verify target table
    target_table_id = f"{PROJECT_ID}.{DATASET}.{TABLE}"
    try:
        table = bq.get_table(target_table_id)
        logging.info(f"Target table {target_table_id} exists.")
    except google.api_core.exceptions.NotFound:
        logging.info(f"Creating target table {target_table_id}")
        table = bigquery.Table(target_table_id, schema=schema)
        table = bq.create_table(table)
        logging.info(f"Created target table {target_table_id}")

    # Merge into master table
    merge_sql = f"""
    MERGE `{target_table_id}` T
    USING `{temp_table_id}` S
    ON T.ticker = S.ticker AND T.quarter_end_date = S.quarter_end_date
    WHEN MATCHED THEN
      UPDATE SET
        T.company_name = S.company_name,
        T.industry = S.industry,
        T.sector = S.sector,
        T.earnings_call_date = S.earnings_call_date
    WHEN NOT MATCHED THEN
      INSERT (ticker, company_name, industry, sector, quarter_end_date, earnings_call_date)
      VALUES (S.ticker, S.company_name, S.industry, S.sector, S.quarter_end_date, S.earnings_call_date)
    """
    try:
        query_job = bq.query(merge_sql)
        query_job.result()  # Wait for the query to complete
        logging.info(f"✅ Merge complete into {target_table_id}")
    except Exception as e:
        logging.error(f"Failed to merge into {target_table_id}: {e}")
        return f"Merge failed: {str(e)}", 500

    # Clean up
    bq.delete_table(temp_table_id, not_found_ok=True)
    logging.info(f"Deleted temporary table {temp_table_id}")
    
    return f"Successfully refreshed {target_table_id}", 200