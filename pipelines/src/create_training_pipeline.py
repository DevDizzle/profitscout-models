#!/usr/bin/env python3
# FILE: training_pipeline.py
"""
Compiles the ProfitScout High-Gamma Training Pipeline.
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
PIPELINE_ROOT = f"gs://{PROJECT_ID}-pipeline-artifacts/training"
TRAINER_IMAGE_URI = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/profit-scout-repo/profitscout-trainer:latest"
)
MODEL_ARTIFACT_DIR = f"{PIPELINE_ROOT}/model-artifacts"

# ───────────────── Component ─────────────────
@dsl.container_component
def training_task(
    project: str,
    source_table: str,
    direction: str,
    xgb_max_depth: int,
    learning_rate: float,
    xgb_min_child_weight: int,
    xgb_subsample: float,
    colsample_bytree: float,
    gamma: float,
    alpha: float,
    reg_lambda: float,
    scale_pos_weight: float,
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE_URI,
        command=["python3", "main.py"],
        args=[
            "--project-id", project,
            "--source-table", source_table,
            "--direction", direction,
            "--xgb-max-depth", xgb_max_depth,
            "--learning-rate", learning_rate,
            "--xgb-min-child-weight", xgb_min_child_weight,
            "--xgb-subsample", xgb_subsample,
            "--colsample-bytree", colsample_bytree,
            "--gamma", gamma,
            "--alpha", alpha,
            "--reg-lambda", reg_lambda,
            "--scale-pos-weight", scale_pos_weight,
        ],
    )


def _compile_and_upload_pipeline(direction: str):
    # Dynamically define the training_op display name
    training_op_display_name = f"profitscout-high-gamma-training-job-{direction.lower()}"
    
    training_op_component = create_custom_training_job_from_component(
        component_spec=training_task,
        display_name=training_op_display_name,
        machine_type="n1-standard-16",
        replica_count=1,
        base_output_directory=MODEL_ARTIFACT_DIR,
    )

    # Dynamically define the pipeline
    @dsl.pipeline(
        name=f"profitscout-high-gamma-training-pipeline-{direction.lower()}",
        description=f"Train High Gamma Classification Model (XGBoost) for {direction} direction on Vertex AI.",
        pipeline_root=PIPELINE_ROOT,
    )
    def training_pipeline(
        project: str = PROJECT_ID,
        source_table: str = "profit_scout.price_data",
        # Default direction is now fixed per compiled pipeline
        xgb_max_depth: int = 6,
        learning_rate: float = 0.06,
        xgb_min_child_weight: int = 12,
        xgb_subsample: float = 0.7,
        colsample_bytree: float = 0.7,
        gamma: float = 2.0,
        alpha: float = 0.5,
        reg_lambda: float = 2.0,
        scale_pos_weight: float = 0.0, # 0.0 = auto
    ):
        training_op_component(
            project=project,
            source_table=source_table,
            direction=direction, # Pass the fixed direction here
            xgb_max_depth=xgb_max_depth,
            learning_rate=learning_rate,
            xgb_min_child_weight=xgb_min_child_weight,
            xgb_subsample=xgb_subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            alpha=alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
        )

    local_path = f"pipelines/compiled/training_pipeline_{direction.lower()}.json"
    gcs_path = f"{PIPELINE_ROOT}/training_pipeline_{direction.lower()}.json"

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=local_path,
    )
    print(f"✓ Compiled to {local_path}")

    try:
        subprocess.run(["gsutil", "cp", local_path, gcs_path], check=True)
        print(f"✓ Uploaded to {gcs_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ GCS upload failed: {e}")

# ───────────────── Compile and Upload ─────────────────
if __name__ == "__main__":
    _compile_and_upload_pipeline("LONG")
    _compile_and_upload_pipeline("SHORT")