"""
Creates a Vertex AI Pipeline to run a Hyperparameter Tuning job.
"""
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google_cloud_pipeline_components.v1.hyperparameter_tuning_job import \
    HyperparameterTuningJobRunOp
from kfp import dsl, compiler

# ──────────────────── Configurable constants ──────────────
PROJECT_ID = "profitscout-lx6bb"
REGION = "us-central1"
PIPELINE_ROOT = "gs://profit-scout-pipeline-artifacts/pipeline-root-hpo"

# Your trainer container image URI
TRAINER_IMAGE_URI = (
    "us-central1-docker.pkg.dev/profitscout-lx6bb/"
    "profit-scout-repo/trainer:latest"
)

# ───────────────────────── Pipeline ────────────────────────
@dsl.pipeline(
    name="hpo-pipeline-for-profitscout",
    description="Runs HPO to find the best parameters for the ProfitScout model.",
    pipeline_root=PIPELINE_ROOT,
)
def hpo_pipeline(
    project: str = PROJECT_ID,
    location: str = REGION,
    source_table: str = "profit_scout.breakout_features",
    max_trial_count: int = 20,
    parallel_trial_count: int = 5,
):
    """Defines the HPO job spec and pipeline."""
    
    # Define the spec for a single trial. This is a CustomJob that
    # runs our trainer container.
    worker_pool_specs = [{
        "machine_spec": {"machine_type": "n1-standard-8"}, # More power for faster trials
        "replica_count": 1,
        "container_spec": {
            "image_uri": TRAINER_IMAGE_URI,
            "args": [
                f"--project-id={project}",
                f"--source-table={source_table}",
            ],
        },
    }]

    # Define the metric to optimize
    metric_spec = {
        "pr_auc": "maximize",
    }

    # Define the parameter search space for Vertex AI Vizier
    parameter_spec = {
        "pca_n": hpt.IntegerParameterSpec(min=32, max=128, scale="linear"),
        "xgb_max_depth": hpt.IntegerParameterSpec(min=4, max=10, scale="linear"),
        "xgb_min_child_weight": hpt.IntegerParameterSpec(min=1, max=10, scale="linear"),
        "xgb_subsample": hpt.DoubleParameterSpec(min=0.7, max=1.0, scale="linear"),
        "logreg_c": hpt.DoubleParameterSpec(min=0.01, max=1.0, scale="log"),
    }

    # The main HPO job operator
    hpo_job = HyperparameterTuningJobRunOp(
        display_name="profitscout-hpo-job",
        project=project,
        location=location,
        worker_pool_specs=worker_pool_specs,
        study_spec_metrics=[metric_spec],
        study_spec_parameters=parameter_spec,
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
        base_output_directory=PIPELINE_ROOT,
    )


# ───────────────────────── Compile ─────────────────────────
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=hpo_pipeline,
        package_path="hpo_pipeline.json",
    )    print("✓ Compiled hpo_pipeline.json")
