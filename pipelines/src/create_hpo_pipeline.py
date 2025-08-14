"""
Vertex AI pipeline that runs a Hyperparameter-Tuning job
for the ProfitScout model.
Requires: pip install -U kfp google-cloud-pipeline-components google-cloud-aiplatform
"""
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google_cloud_pipeline_components.v1.hyperparameter_tuning_job import (
    HyperparameterTuningJobRunOp,
    serialize_metrics,
    serialize_parameters,
)
from kfp import dsl, compiler

# ─────────────────── Config ───────────────────
PROJECT_ID    = "profitscout-lx6bb"
REGION        = "us-central1"
PIPELINE_ROOT = f"gs://{PROJECT_ID}-pipeline-artifacts/hpo"
TRAINER_IMAGE = f"us-central1-docker.pkg.dev/{PROJECT_ID}/profit-scout-repo/trainer:latest"

# ─────────────────── Pipeline ───────────────────
@dsl.pipeline(
    name="profitscout-hpo-pipeline",
    description="Final tight HPO sweep (focal loss, no SMOTE).",
    pipeline_root=PIPELINE_ROOT,
)
def hpo_pipeline(
    project: str = PROJECT_ID,
    location: str = REGION,
    source_table: str = "profit_scout.breakout_features",
    max_trial_count: int = 60,
    parallel_trial_count: int = 3,
):
    # -------- training container spec --------
    worker_pool_specs = [
        {
            "machine_spec": {"machine_type": "n1-highmem-8"},
            "replica_count": 1,
            "container_spec": {
                "image_uri": TRAINER_IMAGE,
                "args": [
                    "--project-id", project,
                    "--source-table", source_table,
                    "--auto-prune", "false",
                    "--top-k-features", "0",
                    "--use-full-data", "false",
                ],
            },
        }
    ]

    # -------- single optimisation metric --------
    metric_spec = serialize_metrics({"pr_auc": "maximize"})

    # -------- refined search space --------
    parameter_spec = serialize_parameters({
        "pca_n":                hpt.DiscreteParameterSpec(values=[256, 512, 768], scale="linear"),
        "xgb_max_depth":        hpt.IntegerParameterSpec(7, 12, "linear"),
        "xgb_min_child_weight": hpt.IntegerParameterSpec(8, 12, "linear"),
        "xgb_subsample":        hpt.DoubleParameterSpec(0.9, 1.0, "linear"),
        "learning_rate":        hpt.DoubleParameterSpec(0.005, 0.03, "log"),
        "gamma":                hpt.DoubleParameterSpec(0.3, 0.5, "linear"),
        "colsample_bytree":     hpt.DoubleParameterSpec(0.6, 0.9, "linear"),
        "alpha":                hpt.DoubleParameterSpec(1e-5, 5e-4, "log"),
        "reg_lambda":           hpt.DoubleParameterSpec(1e-6, 1e-4, "log"),
        "focal_gamma":          hpt.DoubleParameterSpec(1.0, 2.5, "linear"),
    })

    # -------- HPT job op --------
    HyperparameterTuningJobRunOp(
        display_name="profitscout-hpo-job",
        project=project,
        location=location,
        base_output_directory=PIPELINE_ROOT,
        worker_pool_specs=worker_pool_specs,
        study_spec_metrics=metric_spec,
        study_spec_parameters=parameter_spec,
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
        # study_spec_algorithm='BAYESIAN_OPTIMIZATION',  # Optional override
    )

# ─────────────────── Compile ───────────────────
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=hpo_pipeline,
        package_path="hpo_pipeline.json",
    )
    print("✓ Compiled hpo_pipeline.json")
