#!/usr/bin/env python3
# FILE: training_pipeline.py

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
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/profit-scout-repo/trainer:latest"
)
MODEL_ARTIFACT_DIR = f"{PIPELINE_ROOT}/model-artifacts"

# ───────────────── Component ─────────────────
@dsl.container_component
def training_task(
    project: str,
    source_table: str,
    pca_n: int,
    xgb_max_depth: int,
    xgb_min_child_weight: int,
    xgb_subsample: float,
    learning_rate: float,
    gamma: float,
    colsample_bytree: float,
    alpha: float,
    reg_lambda: float,
    focal_gamma: float,
    top_k_features: int,
    auto_prune: str,
    metric_tol: float,
    prune_step: int,
    use_full_data: str,
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE_URI,
        command=["python3", "main.py"],
        args=[
            "--project-id", project,
            "--source-table", source_table,
            "--pca-n", pca_n,
            "--xgb-max-depth", xgb_max_depth,
            "--xgb-min-child-weight", xgb_min_child_weight,
            "--xgb-subsample", xgb_subsample,
            "--learning-rate", learning_rate,
            "--gamma", gamma,
            "--colsample-bytree", colsample_bytree,
            "--alpha", alpha,
            "--reg-lambda", reg_lambda,
            "--focal-gamma", focal_gamma,
            "--top-k-features", top_k_features,
            "--auto-prune", auto_prune,
            "--metric-tol", metric_tol,
            "--prune-step", prune_step,
            "--use-full-data", use_full_data,
        ],
    )

training_op = create_custom_training_job_from_component(
    component_spec=training_task,
    display_name="profitscout-training-job",
    machine_type="n1-standard-16",
    replica_count=1,
    base_output_directory=MODEL_ARTIFACT_DIR,
)

# ───────────────── Pipeline ─────────────────
@dsl.pipeline(
    name="profitscout-standard-training-pipeline",
    description="Train and save ProfitScout model artifacts with HPO‑selected hyperparameters and optional feature selection.",
    pipeline_root=PIPELINE_ROOT,
)
def training_pipeline(
    project: str = PROJECT_ID,
    location: str = REGION,
    source_table: str = "profit_scout.breakout_features",
    pca_n: int = 128,
    xgb_max_depth: int = 7,
    xgb_min_child_weight: int = 2,
    xgb_subsample: float = 0.9,
    learning_rate: float = 0.02,
    gamma: float = 0.10,
    colsample_bytree: float = 0.9,
    alpha: float = 1e-5,
    reg_lambda: float = 2e-5,
    focal_gamma: float = 2.0,
    top_k_features: int = 0,
    auto_prune: str = "false",
    metric_tol: float = 0.002,
    prune_step: int = 25,
    use_full_data: str = "true",
):
    training_op(
        project=project,
        source_table=source_table,
        pca_n=pca_n,
        xgb_max_depth=xgb_max_depth,
        xgb_min_child_weight=xgb_min_child_weight,
        xgb_subsample=xgb_subsample,
        learning_rate=learning_rate,
        gamma=gamma,
        colsample_bytree=colsample_bytree,
        alpha=alpha,
        reg_lambda=reg_lambda,
        focal_gamma=focal_gamma,
        top_k_features=top_k_features,
        auto_prune=auto_prune,
        metric_tol=metric_tol,
        prune_step=prune_step,
        use_full_data=use_full_data,
    )

# ───────────────── Compile and Upload ─────────────────
if __name__ == "__main__":
    local_path = "pipelines/compiled/training_pipeline.json"
    gcs_path = f"{PIPELINE_ROOT}/training_pipeline.json"

    # ✅ Ensure local directory exists
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