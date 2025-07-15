# FILE: create_inference_pipeline.py
"""
Vertex AI pipeline that
  1) pulls new feature rows into a staging table
  2) runs a custom-container job to generate predictions
"""

from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp  # ← use this
from kfp import dsl, compiler

# ───────── config ─────────
PROJECT_ID  = "profitscout-lx6bb"
REGION      = "us-central1"

PIPELINE_ROOT   = f"gs://{PROJECT_ID}-pipeline-artifacts/inference"
PREDICTOR_IMAGE = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/profit-scout-repo/predictor:latest"
)
MODEL_ARTIFACT_DIR = (
    f"gs://{PROJECT_ID}-pipeline-artifacts/training/model-artifacts"
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
):
    # 1️⃣  Stage the batch you want to score
    stage_batch = BigqueryQueryJobOp(
        project  = project,
        location = location,
        query=f"""
          SELECT *
          FROM `{project}.{FEATURES_TBL}`
          -- TODO: replace this filter
          WHERE ticker = 'AAPL'
        """,
        job_configuration_query={
            "destinationTable": {
                "projectId": project,
                "datasetId": DATASET,
                "tableId":   "prediction_input",
            },
            "writeDisposition": "WRITE_TRUNCATE",
        },
    )

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
