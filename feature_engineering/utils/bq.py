import logging
from google.cloud import bigquery

# Initialize BigQuery Client
bq_client = bigquery.Client()

def upsert_row(row: dict, table_id: str, primary_keys: list):
    """
    Merges a single row into a BigQuery table based on primary keys.
    """
    logging.info(f"Upserting row into {table_id}")
    
    # Dynamically build the MERGE statement
    columns = ", ".join(row.keys())
    values = ", ".join(f"@{key}" for key in row.keys())
    update_set = ", ".join(f"T.{key} = S.{key}" for key in row.keys() if key not in primary_keys)
    match_condition = " AND ".join(f"T.{key} = S.{key}" for key in primary_keys)

    query = f"""
    MERGE `{table_id}` AS T
    USING (SELECT * FROM UNNEST([@row_param])) AS S
    ON {match_condition}
    WHEN MATCHED THEN
      UPDATE SET {update_set}
    WHEN NOT MATCHED THEN
      INSERT ({columns}) VALUES ({columns})
    """

    # Set up job configuration
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.StructQueryParameter("row_param", bigquery.SqlParameter(name="row_param", type_=row))
        ]
    )

    try:
        query_job = bq_client.query(query, job_config=job_config)
        query_job.result()  # Wait for the job to complete
        logging.info("Upsert completed successfully.")
    except Exception as e:
        logging.error(f"BigQuery upsert failed: {e}")
        logging.error(f"Failed query: {query}")
        logging.error(f"Row data: {row}")
        raise # Re-raise the exception to be caught by the main handler