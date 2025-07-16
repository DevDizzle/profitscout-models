# FILE: hpo_pipeline.py
"""
Vertex AI pipeline that launches a Hyperparameter‑Tuning job
for the ProfitScout model.

Fast sweep (9 trials, 3 parallel) with:
  • PCA grid {256, 384, 512}
  • Narrow XGBoost / LogReg ranges
  • Correct Vertex placeholder syntax (flag & value split)

Requirements:
  pip install --upgrade google-cloud-pipeline-components google-cloud-aiplatform "kfp>=2.6"
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
TRAINER_IMAGE = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/profit-scout-repo/trainer:latest"
)

# ─────────────────── Pipeline ───────────────────
@dsl.pipeline(
    name="profitscout-hpo-pipeline",
    description="Fast HPO sweep with trimmed search space.",
    pipeline_root=PIPELINE_ROOT,
)
def hpo_pipeline(
    project: str = PROJECT_ID,
    location: str = REGION,
    source_table: str = "profit_scout.breakout_features",
    max_trial_count: int = 9,
    parallel_trial_count: int = 3,
):
    worker_pool_specs = [
        {
            "machine_spec": {"machine_type": "n1-standard-8"},
            "replica_count": 1,
            "container_spec": {
                "image_uri": TRAINER_IMAGE,
                "args": [
                    "--project-id", project,
                    "--source-table", source_table,
                    "--pca-n", "{{$.trial.parameters['pca_n']}}",
                    "--xgb-max-depth", "{{$.trial.parameters['xgb_max_depth']}}",
                    "--xgb-min-child-weight", "{{$.trial.parameters['xgb_min_child_weight']}}",
                    "--xgb-subsample", "{{$.trial.parameters['xgb_subsample']}}",
                    "--logreg-c", "{{$.trial.parameters['logreg_c']}}",
                ],
            },
        }
    ]

    metric_spec = serialize_metrics({"pr_auc": "maximize"})
    parameter_spec = serialize_parameters({
        "pca_n":                hpt.DiscreteParameterSpec([256, 384, 512], scale="linear"),
        "xgb_max_depth":        hpt.IntegerParameterSpec(5, 8,  "linear"),
        "xgb_min_child_weight": hpt.IntegerParameterSpec(1, 5,  "linear"),
        "xgb_subsample":        hpt.DoubleParameterSpec(0.8, 1.0, "linear"),
        "logreg_c":             hpt.DoubleParameterSpec(0.05, 0.5, "log"),
    })

    HyperparameterTuningJobRunOp(
        display_name          = "profitscout-hpo-job",
        project               = project,
        location              = location,
        base_output_directory = PIPELINE_ROOT,
        worker_pool_specs     = worker_pool_specs,
        study_spec_metrics    = metric_spec,
        study_spec_parameters = parameter_spec,
        max_trial_count       = max_trial_count,
        parallel_trial_count  = parallel_trial_count,
    )

# ─────────────────── Compile ───────────────────
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func = hpo_pipeline,
        package_path  = "hpo_pipeline.json",
    )
    print("✓ Compiled hpo_pipeline.json")
