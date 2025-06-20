import os
from google.cloud import bigquery

# ────────────── Config via env vars (set when deploying/running) ──────────────
PROJECT_ID   = os.getenv("PROJECT_ID", "profitscout-lx6bb")
BQ_DATASET   = os.getenv("BQ_DATASET", "profit_scout")
STAGING_NAME = os.getenv("STAGING_TABLE", "call_feats_staging")
DEST_NAME    = os.getenv("DEST_TABLE", "breakout_features")

STAGING = f"{PROJECT_ID}.{BQ_DATASET}.{STAGING_NAME}"
DEST    = f"{PROJECT_ID}.{BQ_DATASET}.{DEST_NAME}"

bq = bigquery.Client(project=PROJECT_ID)

def merge_features(request):
    """HTTP handler: Merge new/updated features from staging to destination table."""

    # Build SET clause dynamically from staging schema
    stg_schema = bq.get_table(STAGING).schema
    set_cols = ", ".join(
        f"{f.name}=COALESCE(S.{f.name}, T.{f.name})"
        for f in stg_schema
        if f.name not in ("ticker", "quarter_end_date")
    )
    sql = f"""
    MERGE `{DEST}` T
    USING `{STAGING}` S
      ON T.ticker = S.ticker
     AND DATE(T.quarter_end_date) = S.quarter_end_date
    WHEN MATCHED THEN UPDATE SET {set_cols}
    WHEN NOT MATCHED THEN INSERT ROW
    """
    bq.query(sql).result()
    return "merge complete", 200
