import typing as t, os, json
from google.cloud import bigquery

def bq_client(project: str | None = None) -> bigquery.Client:
    return bigquery.Client(project=project)

def table_ref(dataset: str, table: str, project: str) -> str:
    return f"{project}.{dataset}.{table}"

MERGE_TMPL_BASE = """
MERGE `{tbl}` T
USING (
    SELECT
        @ticker AS ticker,
        DATE(@qend) AS quarter_end_date{select_cols}
) S
ON  T.ticker = S.ticker
AND T.quarter_end_date = S.quarter_end_date
{when_matched}
WHEN NOT MATCHED THEN INSERT ({all_cols})
VALUES ({all_vals})
"""

def upsert_json(tbl: str, row: dict, client: bigquery.Client):
    cols = [k for k in row if k not in ("ticker", "quarter_end_date")]
    select_cols = ""
    if cols:
        select_cols = ",\n        " + ",\n        ".join(f"@{c} AS {c}" for c in cols)
    set_cols = ",\n    ".join(f"{c}=S.{c}" for c in cols)
    all_cols = "ticker, quarter_end_date" + (", " + ", ".join(cols) if cols else "")
    all_vals = "@ticker, DATE(@qend)" + (", " + ", ".join(f"@{c}" for c in cols) if cols else "")
    MERGE_TMPL = """
    MERGE `{tbl}` T
    USING (
        SELECT
            @ticker AS ticker,
            DATE(@qend) AS quarter_end_date
            {select_cols}
    ) S
    ON  T.ticker = S.ticker
    AND T.quarter_end_date = S.quarter_end_date
    WHEN MATCHED THEN UPDATE SET
        {set_cols}
    WHEN NOT MATCHED THEN INSERT ({all_cols})
    VALUES ({all_vals})
    """
    query = MERGE_TMPL.format(
        tbl=tbl,
        select_cols=select_cols,
        set_cols=set_cols,
        all_cols=all_cols,
        all_vals=all_vals,
    )

    params = [
        bigquery.ScalarQueryParameter("ticker", "STRING", row["ticker"]),
        bigquery.ScalarQueryParameter("qend", "DATE", row["quarter_end_date"]),
    ]
    for k, v in row.items():
        if k not in ("ticker", "quarter_end_date"):
            if isinstance(v, list):
                params.append(
                    bigquery.ArrayQueryParameter(k, "FLOAT64", v)
                )
            else:
                # If not list, treat as scalar (string/float)
                dtype = "FLOAT64" if isinstance(v, float) else "STRING"
                params.append(
                    bigquery.ScalarQueryParameter(k, dtype, v)
                )

    job_cfg = bigquery.QueryJobConfig(query_parameters=params)
    client.query(query, job_cfg).result()