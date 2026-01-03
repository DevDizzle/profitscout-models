#!/usr/bin/env python3
# FILE: hpo_pipeline.py
"""
Creates a Vertex AI Hyperparameter Tuning Job for the High Gamma Model.
"""

from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

PROJECT_ID = "profitscout-lx6bb"
REGION = "us-central1"
STAGING_BUCKET = f"gs://{PROJECT_ID}-pipeline-artifacts"
TRAINER_IMAGE_URI = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/profit-scout-repo/trainer:latest"
)

def create_hpo_job():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

    # Define the custom job spec (what runs in each trial)
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-16",
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": TRAINER_IMAGE_URI,
                "args": [
                    "--project-id", PROJECT_ID,
                    "--source-table", "profit_scout.price_data",
                    # HPO service will append the tuning args automatically
                ],
            },
        }
    ]

    # Create a CustomJob object for the trials
    custom_job = aiplatform.CustomJob(
        display_name="profitscout-hpo-trial", # This is the display name for each trial's CustomJob
        worker_pool_specs=worker_pool_specs,
        project=PROJECT_ID,
        location=REGION,
    )

    # Define the search space
    parameter_spec = {
        "learning-rate": hpt.DoubleParameterSpec(min=0.01, max=0.3, scale="log"),
        "xgb-max-depth": hpt.IntegerParameterSpec(min=3, max=10, scale="linear"),
        "xgb-min-child-weight": hpt.IntegerParameterSpec(min=1, max=20, scale="linear"),
        "xgb-subsample": hpt.DoubleParameterSpec(min=0.5, max=1.0, scale="linear"),
        "colsample-bytree": hpt.DoubleParameterSpec(min=0.5, max=1.0, scale="linear"),
        "gamma": hpt.DoubleParameterSpec(min=0.0, max=5.0, scale="linear"),
        "alpha": hpt.DoubleParameterSpec(min=0.0, max=1.0, scale="linear"),
        "reg-lambda": hpt.DoubleParameterSpec(min=0.0, max=5.0, scale="linear"),
    }

    # Create the Tuning Job
    hpo_job = aiplatform.HyperparameterTuningJob(
        display_name="profitscout-high-gamma-hpo",
        custom_job=custom_job,
        metric_spec={"pr_auc": "maximize"},
        parameter_spec=parameter_spec,
        max_trial_count=20,
        parallel_trial_count=4,
    )

    print("Submitting HPO Job...")
    hpo_job.run(sync=False) # Submit and return (don't block CLI)
    print(f"HPO Job submitted. Resource name: {hpo_job.resource_name}")
    print(f"View in Console: https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs?project={PROJECT_ID}")

if __name__ == "__main__":
    create_hpo_job()