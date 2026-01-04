#!/usr/bin/env python3
# FILE: inference_pipeline.py
"""
Compiles the ProfitScout Batch Prediction Pipeline.
"""

from kfp import dsl, compiler
from google_cloud_pipeline_components.v1.custom_job import (
    create_custom_training_job_from_component,
)
import subprocess
import os

# ─────────────────── Config ───────────────────
PROJECT_ID = "profitscout-lx6bb"
REGION = "us-central1"
PIPELINE_ROOT = f"gs://{PROJECT_ID}-pipeline-artifacts/inference"
PREDICTOR_IMAGE_URI = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/profit-scout-repo/profitscout-predictor:latest"
)
# Note: Inference doesn't typically output a model artifact, but CustomJob reqs a base_output_dir
BASE_OUTPUT_DIR = f"{PIPELINE_ROOT}/job-output"

# ───────────────── Component ─────────────────
@dsl.container_component
def prediction_task(
    project: str,
    source_table: str,
    destination_table: str,
    model_base_dir: str,
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=PREDICTOR_IMAGE_URI,
        command=["python3", "main.py"],
        args=[
            "--project-id", project,
            "--source-table", source_table,
            "--destination-table", destination_table,
            "--model-base-dir", model_base_dir,
        ],
    )

# Using create_custom_training_job_from_component allows us to run this as a "Custom Job"
# even though it's inference. This is often cheaper/simpler than BatchPredictionJob 
# if we have custom logic (like feature engineering inside the container).
prediction_op = create_custom_training_job_from_component(
    component_spec=prediction_task,
    display_name="profitscout-batch-prediction",
    machine_type="n1-standard-8",
    replica_count=1,
    base_output_directory=BASE_OUTPUT_DIR,
)

# ───────────────── Pipeline ─────────────────
@dsl.pipeline(
    name="profitscout-daily-prediction-pipeline",
    description="Generate High Gamma predictions for all tickers.",
    pipeline_root=PIPELINE_ROOT,
)
def inference_pipeline(
    project: str = PROJECT_ID,
    source_table: str = "profit_scout.price_data",
    destination_table: str = "profit_scout.daily_predictions",
    model_base_dir: str = "gs://profitscout-lx6bb-pipeline-artifacts/production/model", 
):
    prediction_op(
        project=project,
        source_table=source_table,
        destination_table=destination_table,
        model_base_dir=model_base_dir,
    )

# ───────────────── Compile and Upload ─────────────────
if __name__ == "__main__":
    local_path = "pipelines/compiled/inference_pipeline.json"
    gcs_path = f"{PIPELINE_ROOT}/inference_pipeline.json"

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    compiler.Compiler().compile(
        pipeline_func=inference_pipeline,
        package_path=local_path,
    )
    print(f"✓ Compiled to {local_path}")

    try:
        subprocess.run(["gsutil", "cp", local_path, gcs_path], check=True)
        print(f"✓ Uploaded to {gcs_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ GCS upload failed: {e}")