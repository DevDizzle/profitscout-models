# /jobs/price_technicals/bq.py

import typing as t
from google.cloud import bigquery

def bq_client(project: str | None = None) -> bigquery.Client:
    return bigquery.Client(project=project)

def table_ref(dataset: str, table: str, project: str) -> str:
    return f"{project}.{dataset}.{table}"

MERGE_TMPL = """
MERGE `{tbl}` T
USING (SELECT @ticker AS ticker,
              DATE(@qend) AS quarter_end_date) S
ON  T.ticker = S.ticker
AND T.quarter_end_date = S.quarter_end_date
WHEN MATCHED THEN UPDATE SET {set_cols}
WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ({vals})
"""

def upsert_json(tbl: str, row: dict, client: bigquery.Client):
    params = [
        bigquery.ScalarQueryParameter("ticker", "STRING", row.get("ticker")),
        bigquery.ScalarQueryParameter("qend", "DATE", row.get("quarter_end_date")),
    ]
    
    set_clause_parts = []
    cols_parts = []
    vals_parts = []

    for k, v in row.items():
        if k not in ("ticker", "quarter_end_date"):
            set_clause_parts.append(f"{k}=@{k}")
            cols_parts.append(k)
            vals_parts.append(f"@{k}")

            param_type = "STRING"
            if isinstance(v, bool):
                param_type = "BOOL"
            elif isinstance(v, int):
                param_type = "INT64"
            elif isinstance(v, float):
                param_type = "FLOAT64"
            elif v is None:
                # Let BigQuery infer NULL type
                pass
            
            # For non-array types
            if not isinstance(v, list):
                 params.append(bigquery.ScalarQueryParameter(k, param_type, v))

    set_clause = ",\n    ".join(set_clause_parts)
    cols = "ticker, quarter_end_date, " + ", ".join(cols_parts)
    vals = "@ticker, DATE(@qend), " + ", ".join(vals_parts)

    query = MERGE_TMPL.format(tbl=tbl, set_cols=set_clause, cols=cols, vals=vals)
    job_cfg = bigquery.QueryJobConfig(query_parameters=params)
    client.query(query, job_cfg).result()