import os
import logging
import requests
from google.cloud import bigquery

# Environment Variables
PROJECT_ID = os.environ.get('PROJECT_ID')
FMP_API_KEY = os.environ.get('FMP_API_KEY')
METADATA_TABLE = os.environ.get('METADATA_TABLE') # e.g., your_project.your_dataset.stock_metadata

# Clients
bq_client = bigquery.Client()
logging.basicConfig(level=logging.INFO)

def refresh_metadata(event, context):
    """
    Cloud Function to fetch all stock metadata from FMP and load it into BigQuery.
    """
    logging.info("Starting stock metadata refresh job.")
    
    try:
        # 1. Fetch data from FinancialModelingPrep API
        logging.info("Fetching stock list from FMP...")
        url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={FMP_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        logging.info(f"Successfully fetched {len(data)} records from FMP.")

        if not data:
            logging.warning("API returned no data. Exiting.")
            return "No data from API", 200

        # 2. Load data into BigQuery
        job_config = bigquery.LoadJobConfig(
            # Overwrite the table with the fresh data
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            # Specify the schema (adjust types as needed)
            schema=[
                bigquery.SchemaField("symbol", "STRING"),
                bigquery.SchemaField("name", "STRING"),
                bigquery.SchemaField("exchange", "STRING"),
                bigquery.SchemaField("sector", "STRING"), # Custom field
                bigquery.SchemaField("industry", "STRING"), # Custom field
            ],
            # FMP provides more fields, ignore any we don't need
            ignore_unknown_values=True,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        )

        load_job = bq_client.load_table_from_json(
            data, METADATA_TABLE, job_config=job_config
        )
        load_job.result() # Wait for the job to complete

        logging.info(f"Successfully loaded {load_job.output_rows} rows to {METADATA_TABLE}")
        return "Success", 200

    except Exception as e:
        logging.error(f"Metadata refresh failed: {e}")
        return "Error", 500