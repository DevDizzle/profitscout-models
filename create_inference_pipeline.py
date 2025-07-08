"""Compile the daily batch inference pipeline for Vertex AI."""

from kfp import dsl, compiler
from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google_cloud_pipeline_components.types import artifact_types

# ───── constants ─────
PROJECT_ID   = "profitscout-lx6bb"
REGION       = "us-central1"
PIPE_ROOT    = "gs://profit-scout-pipeline-artifacts/pipeline-root"

DATASET      = "profit_scout"
FEATURES_TBL = f"{DATASET}.breakout_features"
INPUT_TBL_ID = "daily_prediction_input"
INPUT_TBL    = f"{PROJECT_ID}.{DATASET}.{INPUT_TBL_ID}"
OUTPUT_DATASET_URI = f"bq://{PROJECT_ID}.{DATASET}"
MODEL_ID     = "profitscout-xgboost-classifier@default"   # registry alias

# ───── pipeline ─────
@dsl.pipeline(
    name="profitscout-daily-batch-inference",
    pipeline_root=PIPE_ROOT,
    description="Queries new rows and runs Vertex AI BatchPredict.",
)
def daily_inference_pipeline(
    project: str = PROJECT_ID,
    location: str = REGION,
):
    """Queries new rows then performs batch predictions."""

    # 1️⃣ filter new rows → BigQuery
    BigqueryQueryJobOp(
        project  = project,
        location = location,
        query = f"""
          SELECT *
          FROM `{project}.{FEATURES_TBL}`
          WHERE max_close_30d IS NULL
        """,
        job_configuration_query = {
            "destinationTable": {
                "projectId": project,
                "datasetId": DATASET,
                "tableId"  : INPUT_TBL_ID,
            },
            "writeDisposition": "WRITE_TRUNCATE",
        },
    )

<<<<<<< HEAD
    # 2️⃣ importer node converts the registry model to a VertexModel artifact
    model_artifact = dsl.importer(
        artifact_uri = (
            f"https://{location}-aiplatform.googleapis.com/v1/"
            f"projects/{project}/locations/{location}/models/{MODEL_ID}"
        ),
        artifact_class = artifact_types.VertexModel,
        metadata = {"resourceName":
            f"projects/{project}/locations/{location}/models/{MODEL_ID}"}
    )

    # 3️⃣ batch prediction
    ModelBatchPredictOp(
        project                         = project,
        location                        = location,
        job_display_name                = "daily-profitscout-batch-prediction",
        model                           = model_artifact.outputs["artifact"], # use artifact
        instances_format                = "bigquery",
        predictions_format              = "bigquery",
        bigquery_source_input_uri       = INPUT_TBL,
        bigquery_destination_output_uri = OUTPUT_DATASET_URI,
        machine_type                    = "n1-standard-4",
    )

# ───── compile ─────
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func = daily_inference_pipeline,
        package_path  = "daily_inference_pipeline.json",
    )
    print("✓ Compiled daily_inference_pipeline.json")
=======
    # 2️⃣  Run your custom-container job for inference
    CustomTrainingJobOp(
        display_name = "profitscout-prediction-job",
        project      = project,
        location     = location,
        worker_pool_specs=[{
            "machine_spec": {"machine_type": "n1-standard-4"},
            "replica_count": 1,
            "container_spec": {
                "image_uri": PREDICTOR_IMAGE,
                "args": [
                    f"--project-id={project}",
                    f"--source-table={PREDICTION_INPUT_TBL}",
                    f"--destination-table={PREDICTION_OUTPUT_TBL}",
                    f"--model-dir={MODEL_ARTIFACT_DIR}",
                ],
            },
        }],
    ).after(stage_batch)

# ───────── compile ─────────
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func = inference_pipeline,
        package_path  = "inference_pipeline.json",
    )
    print("✓ Compiled inference_pipeline.json")
>>>>>>> f338ebf (Local edits before rebase)
