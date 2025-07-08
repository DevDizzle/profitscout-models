"""
create_training_pipeline.py
Author: ProfitScout ML Engineering
Description:
  • Trains the XGBoost classifier weekly on Vertex AI
  • Imports the unmanaged model artifact from GCS
  • Uploads / registers the model in Vertex AI Model Registry
"""

# ───────────────────────── Imports ─────────────────────────
import os
from kfp import dsl, compiler
from kfp.dsl import importer                                # universal importer
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp

# ──────────────────── Configurable constants ──────────────
PROJECT_ID        = "profitscout-lx6bb"
REGION            = "us-central1"
<<<<<<< HEAD
PIPELINE_ROOT     = "gs://profit-scout-pipeline-artifacts/pipeline-root"

# container image that runs training
TRAINER_IMAGE_URI = (
    "us-central1-docker.pkg.dev/profitscout-lx6bb/"
    "profit-scout-repo/trainer:latest"
)

MODEL_DISPLAY_NAME = "profitscout-xgboost-classifier"

# one canonical folder where every run puts its model artefacts
MODEL_DIR = f"{PIPELINE_ROOT}/models"

# ───────────────────────── Pipeline ────────────────────────
@dsl.pipeline(
    name="weekly-training-and-register-pipeline",
    description="Train and register the ProfitScout model weekly.",
    pipeline_root=PIPELINE_ROOT,
)
def weekly_training_pipeline(
    project: str = PROJECT_ID,
    location: str = REGION,
    model_display_name: str = MODEL_DISPLAY_NAME,
    source_table: str = "profit_scout.breakout_features",
    metadata_table: str = "profit_scout.stock_metadata",
):
    """Weekly training + registration workflow."""

    # 1️⃣ Custom Training Job spec (Vertex AI Training)
    worker_pool_specs = [{
        "machine_spec": {"machine_type": "n1-standard-4"},
=======
PIPELINE_ROOT     = f"gs://{PROJECT_ID}-pipeline-artifacts/training"
TRAINER_IMAGE_URI = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/"
    "profit-scout-repo/trainer:latest"
)
MODEL_ARTIFACT_DIR = f"{PIPELINE_ROOT}/model-artifacts"

# ───────────────────────── Pipeline ────────────────────────
@dsl.pipeline(
    name="profitscout-standard-training-pipeline",
    description="Trains and saves the ProfitScout model artifacts with specified hyperparameters.",
    pipeline_root=PIPELINE_ROOT,
)
def training_pipeline(
    project: str = PROJECT_ID,
    location: str = REGION,
    source_table: str = "profit_scout.breakout_features",
    # --- Pass the best hyperparameters from the HPO job here ---
    pca_n: int = 64,
    xgb_max_depth: int = 7,
    xgb_min_child_weight: int = 5,
    xgb_subsample: float = 0.8,
    logreg_c: float = 0.1,
):
    """Defines the standard training job spec."""
    worker_pool_specs = [{
        "machine_spec": {"machine_type": "n1-standard-8"},
>>>>>>> f338ebf (Local edits before rebase)
        "replica_count": 1,
        "container_spec": {
            "image_uri": TRAINER_IMAGE_URI,
            "args": [
<<<<<<< HEAD
                f"--source-table={source_table}",
                f"--metadata-table={metadata_table}",
            ],
            # Explicitly point the trainer to the folder Vertex AI will sync
            "env": [{"name": "AIP_MODEL_DIR", "value": MODEL_DIR}],
=======
                f"--project-id={project}",
                f"--source-table={source_table}",
                # Pass hyperparameters to the trainer script
                f"--pca-n={pca_n}",
                f"--xgb-max-depth={xgb_max_depth}",
                f"--xgb-min-child-weight={xgb_min_child_weight}",
                f"--xgb-subsample={xgb_subsample}",
                f"--logreg-c={logreg_c}",
            ],
            "env": [{"name": "AIP_MODEL_DIR", "value": MODEL_ARTIFACT_DIR}],
>>>>>>> f338ebf (Local edits before rebase)
        },
    }]

    train_job = CustomTrainingJobOp(
<<<<<<< HEAD
        display_name="weekly-profitscout-training-job",
        project=project,
        location=location,
        worker_pool_specs=worker_pool_specs,
        base_output_directory=MODEL_DIR,  # Vertex AI copies $AIP_MODEL_DIR → here
    )

    # 2️⃣ Import the model artefact produced by the training container
    model_artifact = importer(
        artifact_uri=MODEL_DIR,
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={"containerSpec": {"imageUri": TRAINER_IMAGE_URI}},
    )

    # 3️⃣ Upload / register the imported model
    upload_job = ModelUploadOp(
        display_name=model_display_name,
        project=project,
        location=location,
        unmanaged_container_model=model_artifact.output,
    )

    # Ensure upload runs after training
    upload_job.after(train_job)

# ───────────────────────── Compile ─────────────────────────
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=weekly_training_pipeline,
        package_path="weekly_training_pipeline.json",
    )
    print("✓ Compiled weekly_training_pipeline.json")
=======
        display_name="profitscout-training-job",
        project=project,
        location=location,
        worker_pool_specs=worker_pool_specs,
        base_output_directory=PIPELINE_ROOT,
    )

# ───────────────────────── Compile ─────────────────────────
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="training_pipeline.json",
    )
    print("✓ Compiled training_pipeline.json")
>>>>>>> f338ebf (Local edits before rebase)
