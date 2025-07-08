import logging
from google.cloud import bigquery
import json
import re

# Initialize BigQuery Client
bq_client = bigquery.Client()

def _format_value(value):
    """Formats Python values for use in a SQL string."""
    if value is None:
        return "NULL"
    if isinstance(value, (int, float, bool)):
        return str(value).upper()
    if isinstance(value, list):
        return f"[{','.join(map(str, value))}]"
    
    # Handle both date and datetime strings by casting the date portion.
    if isinstance(value, str) and re.match(r'^\d{4}-\d{2}-\d{2}', value):
        # Take only the date part of the string (e.g., '2025-05-01')
        date_part = value.split(' ')[0]
        escaped_date_part = date_part.replace("'", "''")
        return f"CAST('{escaped_date_part}' AS DATE)"

    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"

def upsert_row(row: dict, table_id: str, primary_keys: list):
    """
    Merges a single row into a BigQuery table based on primary keys
    by constructing a safe SQL query string.
    """
    logging.info(f"Upserting row into {table_id}")
    query = ""
    try:
        valid_row_items = {k: v for k, v in row.items() if v is not None}
        
        using_clause_parts = []
        for k, v in valid_row_items.items():
            using_clause_parts.append(f"{_format_value(v)} AS `{k}`")
        
        using_clause = ", ".join(using_clause_parts)
        
        columns = ", ".join(f"`{k}`" for k in valid_row_items.keys())
        update_set = ", ".join(f"T.`{k}` = S.`{k}`" for k in valid_row_items.keys() if k not in primary_keys)
        match_condition = " AND ".join(f"T.`{k}` = S.`{k}`" for k in primary_keys)
        
        query = f"""
        MERGE `{table_id}` AS T
        USING (SELECT {using_clause}) AS S
        ON {match_condition}
        WHEN MATCHED THEN
            UPDATE SET {update_set}
        WHEN NOT MATCHED THEN
            INSERT ({columns}) VALUES ({columns})
        """

        query_job = bq_client.query(query)
        query_job.result()
        logging.info("Upsert completed successfully.")

    except Exception as e:
        logging.error(f"BigQuery upsert failed: {e}")
        logging.error(f"Failed query: {query}")
        safe_row = {k: (type(v).__name__ if isinstance(v, list) else v) for k, v in row.items()}
        logging.error(f"Row data (types simplified): {safe_row}")        raise
