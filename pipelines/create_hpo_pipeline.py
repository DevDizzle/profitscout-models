# FILE: hpo_pipeline.py
"""
Vertex AI pipeline that launches a Hyperparameter Tuning job
for the ProfitScout model.

Requirements:
```bash
python -m pip install --upgrade \
  google-cloud-pipeline-components \
  google-cloud-aiplatform \
  "kfp>=2.6"
```
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
    description="Final HPO round with 50 trials, focused on top hyps + smote.",
    pipeline_root=PIPELINE_ROOT,
)
def hpo_pipeline(
    project: str = PROJECT_ID,
    location: str = REGION,
    source_table: str = "profit_scout.breakout_features",
    max_trial_count: int = 50,
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
                ],
            },
        }
    ]
    metric_spec = serialize_metrics({"pr_auc": "maximize"})
    parameter_spec = serialize_parameters({
        "pca_n":                hpt.DiscreteParameterSpec([128, 192, 256], scale="linear"),  # Focused around 128
        "xgb_max_depth":        hpt.IntegerParameterSpec(5, 7, "linear"),  # Around 6
        "xgb_min_child_weight": hpt.IntegerParameterSpec(1, 3, "linear"),  # Around 2
        "xgb_subsample":        hpt.DoubleParameterSpec(0.85, 0.95, "linear"),  # Around 0.9
        "logreg_c":             hpt.DoubleParameterSpec(0.08, 0.12, "log"),  # Around 0.1
        "blend_weight":         hpt.DoubleParameterSpec(0.55, 0.65, "linear"),  # Around 0.6
        "learning_rate":        hpt.DoubleParameterSpec(0.015, 0.025, "log"),  # Around 0.02
        "gamma":                hpt.DoubleParameterSpec(0.05, 0.15, "linear"),  # Around 0.09
        "colsample_bytree":     hpt.DoubleParameterSpec(0.85, 0.95, "linear"),  # Around 0.9
        "scale_pos_weight":     hpt.DoubleParameterSpec(5.0, 15.0, "log"),  # Focused higher
        "alpha":                hpt.DoubleParameterSpec(1e-5, 0.05, "log"),
        "reg_lambda":           hpt.DoubleParameterSpec(1e-5, 0.05, "log"),
        "use_smote":            hpt.CategoricalParameterSpec(['true', 'false']),  # New discrete for SMOTE
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