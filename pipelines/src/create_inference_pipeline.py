#!/usr/bin/env python3
"""
Vertex AI pipeline that
  1) builds a prediction batch in BigQuery
  2) runs a custom‑container job to score it
  gcloud builds submit --tag us-central1-docker.pkg.dev/profitscout-lx6bb/profit-scout-repo/predictor:latest .
"""

import os
import subprocess
from kfp import dsl, compiler
from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

# ───────── config ─────────
PROJECT_ID  = "profitscout-lx6bb"
REGION      = "us-central1"

PIPELINE_ROOT   = f"gs://{PROJECT_ID}-pipeline-artifacts/inference"
PREDICTOR_IMAGE = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/profit-scout-repo/predictor:latest"
)

DATASET               = "profit_scout"
FEATURES_TBL          = f"{DATASET}.breakout_features"
PREDICTION_INPUT_TBL  = f"{PROJECT_ID}.{DATASET}.prediction_input"
PREDICTION_OUTPUT_TBL = f"{PROJECT_ID}.{DATASET}.predictions"

# ───────── pipeline ─────────
@dsl.pipeline(
    name="profitscout-custom-batch-inference",
    pipeline_root=PIPELINE_ROOT,
)
def inference_pipeline(
    project:  str = PROJECT_ID,
    location: str = REGION,
    model_version_dir: str = f"gs://{PROJECT_ID}-pipeline-artifacts/training/model-artifacts",
):
    stage_batch = BigqueryQueryJobOp(
        project=project,
        location=location,
        query=f"""
            CREATE OR REPLACE TABLE `{PREDICTION_INPUT_TBL}` AS
            SELECT *
            FROM   `{project}.{FEATURES_TBL}`
            WHERE  max_close_30d IS NULL
        """,
    )

    CustomTrainingJobOp(
        display_name="profitscout-prediction-job",
        project=project,
        location=location,
        worker_pool_specs=[{
            "machine_spec":  {"machine_type": "n1-standard-4"},
            "replica_count": 1,
            "container_spec": {
                "image_uri": PREDICTOR_IMAGE,
                "args": [
                    "--project-id",         project,
                    "--source-table",       f"{DATASET}.prediction_input",
                    "--destination-table",  PREDICTION_OUTPUT_TBL,
                    "--model-dir",          model_version_dir,
                ],
            },
        }],
    ).after(stage_batch)

# ───────── compile ─────────
if __name__ == "__main__":
    local_path = "pipelines/compiled/inference_pipeline.json"
    gcs_path   = f"{PIPELINE_ROOT}/inference_pipeline.json"

    # Ensure local folder exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=inference_pipeline,
        package_path=local_path,
    )
    print(f"✓ Compiled to {local_path}")

    # Upload to GCS
    try:
        subprocess.run(["gsutil", "cp", local_path, gcs_path], check=True)
        print(f"✓ Uploaded to {gcs_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to upload to GCS: {e}")