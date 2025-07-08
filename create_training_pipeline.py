# FILE: create_training_pipeline.py (Final Version)
"""
Creates a Vertex AI Pipeline to run a standard, single training job,
accepting hyperparameters as arguments.
"""
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from kfp import dsl, compiler

# ──────────────────── Configurable constants ──────────────
PROJECT_ID        = "profitscout-lx6bb"
REGION            = "us-central1"
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
        "replica_count": 1,
        "container_spec": {
            "image_uri": TRAINER_IMAGE_URI,
            "args": [
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
        },
    }]

    train_job = CustomTrainingJobOp(
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
