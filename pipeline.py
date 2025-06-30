import os
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import importer
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google_cloud_pipeline_components.types import artifact_types  # NEW

# ─── Constants ──────────────────────────────────────────────────────
PROJECT_ID = "profitscout-lx6bb"
REGION = "us-central1"
PIPELINE_ROOT = "gs://profit-scout-pipeline-artifacts/pipeline-root"
BASE_OUTPUT_DIR = "gs://profit-scout-pipeline-artifacts/models"
TRAINER_IMAGE_URI = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/profit-scout-repo/trainer:latest"
)

# BigQuery tables / dataset
SOURCE_TABLE = "profit_scout.breakout_features"
METADATA_TABLE = "profit_scout.stock_metadata"
PREDICTION_DESTINATION_TABLE = "profit_scout"  # dataset only → Vertex creates the table

# ─── Pipeline Definition ────────────────────────────────────────────
@dsl.pipeline(
    name="profitscout-training-pipeline-final",
    pipeline_root=PIPELINE_ROOT,
)
def profitscout_pipeline(
    model_display_name: str = "profitscout-xgboost-classifier",
    source_table: str = SOURCE_TABLE,
    metadata_table: str = METADATA_TABLE,
    prediction_destination: str = PREDICTION_DESTINATION_TABLE,
    base_output_dir: str = BASE_OUTPUT_DIR,
):
    # 1. Training
    train_op = CustomTrainingJobOp(
        project=PROJECT_ID,
        location=REGION,
        display_name="profitscout-final-training-run",
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": "n1-standard-4"},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": TRAINER_IMAGE_URI,
                    "args": [
                        f"--source-table={source_table}",
                        f"--metadata-table={metadata_table}",
                    ],
                },
            }
        ],
        base_output_directory=base_output_dir,
    )

    # 2. Import model artifacts in GCS as an unmanaged model
    import_model_op = importer(
        artifact_uri=base_output_dir,
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={
            "containerSpec": {
                "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest"
            }
        },
    ).after(train_op)

    # 3. Upload to Vertex AI Model Registry
    model_upload_op = ModelUploadOp(
        project=PROJECT_ID,
        location=REGION,
        display_name=model_display_name,
        unmanaged_container_model=import_model_op.outputs["artifact"],
    )

    # 4. Batch prediction
    batch_predict_op = ModelBatchPredictOp(
        project=PROJECT_ID,
        location=REGION,
        job_display_name="profitscout-batch-prediction",
        model=model_upload_op.outputs["model"],
        bigquery_source_input_uri=f"bq://{PROJECT_ID}.{source_table}",  # FIXED
        bigquery_destination_output_uri=f"bq://{PROJECT_ID}.{prediction_destination}",
        instances_format="bigquery",
        predictions_format="bigquery",
        machine_type="n1-standard-4",
    )
    batch_predict_op.after(model_upload_op)


# ─── Compile entry point ────────────────────────────────────────────
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=profitscout_pipeline,
        package_path="profitscout_pipeline.json",
    )
    print("Pipeline compiled successfully to profitscout_pipeline.json")
