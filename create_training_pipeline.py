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
        "replica_count": 1,
        "container_spec": {
            "image_uri": TRAINER_IMAGE_URI,
            "args": [
                f"--source-table={source_table}",
                f"--metadata-table={metadata_table}",
            ],
            # Explicitly point the trainer to the folder Vertex AI will sync
            "env": [{"name": "AIP_MODEL_DIR", "value": MODEL_DIR}],
        },
    }]

    train_job = CustomTrainingJobOp(
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
