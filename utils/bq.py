import typing as t, os, json
from google.cloud import bigquery

def bq_client(project: str | None = None) -> bigquery.Client:
    return bigquery.Client(project=project)

def table_ref(dataset: str, table: str, project: str) -> str:
    return f"{project}.{dataset}.{table}"

# single-row upsert → MERGE
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
    cols = [k for k in row if k not in ("ticker", "quarter_end_date")]
    set_clause = ",\n    ".join(f"{c}=S2.{c}" for c in cols)
    placeholders = ", ".join(f"@{c}" for c in cols)
    query = MERGE_TMPL.format(
        tbl=tbl,
        set_cols=set_clause,
        cols="ticker, quarter_end_date, " + ", ".join(cols),
        vals="@ticker, DATE(@qend), " + placeholders,
    )
    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", row["ticker"]),
            bigquery.ScalarQueryParameter("qend", "DATE",  row["quarter_end_date"]),
        ] + [
            bigquery.ScalarQueryParameter(k, "FLOAT64" if isinstance(v, float) else "STRING", v)
            for k, v in row.items() if k not in ("ticker", "quarter_end_date")
        ]
    )
    client.query(query, job_cfg).result()
