#!/usr/bin/env python3
"""
Vertex AI pipeline that
  1) builds a prediction batch in BigQuery
  2) runs a custom‑container job to score it
"""

from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from google_cloud_pipeline_components.v1.custom_job   import CustomTrainingJobOp
from kfp import dsl, compiler

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
    # artefacts for the model version you want to run
    model_version_dir: str = (
        f"gs://{PROJECT_ID}-pipeline-artifacts/training/model-artifacts"
    ),
    # pruning knobs (kept for parity with training)
    top_k_features: int = 0,
    auto_prune: str = "false",
    metric_tol: float = 0.002,
    prune_step: int = 25,
):
    # 1️⃣  Build the batch to predict (rows with NULL max_close_30d)
    stage_batch = BigqueryQueryJobOp(
        project  = project,
        location = location,
        query=f"""
          CREATE OR REPLACE TABLE `{PREDICTION_INPUT_TBL}` AS
          SELECT *
          FROM   `{project}.{FEATURES_TBL}`
          WHERE  max_close_30d IS NULL
        """,
    )

    # 2️⃣  Score the batch using your custom container
    CustomTrainingJobOp(
        display_name = "profitscout-prediction-job",
        project      = project,
        location     = location,
        worker_pool_specs=[{
            "machine_spec":  {"machine_type": "n1-standard-4"},
            "replica_count": 1,
            "container_spec": {
                "image_uri": PREDICTOR_IMAGE,
                "args": [
                    "--project-id",         project,
                    "--source-table", f"{DATASET}.prediction_input",
                    "--destination-table",  PREDICTION_OUTPUT_TBL,
                    "--model-dir",          model_version_dir,
                    "--top-k-features",     top_k_features,
                    "--auto-prune",         auto_prune,
                    "--metric-tol",         metric_tol,
                    "--prune-step",         prune_step,
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
