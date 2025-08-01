# FILE: training_job_run_now.py
from datetime import datetime
from google.cloud import aiplatform

aiplatform.init(project="profitscout-lx6bb", location="us-central1")

PIPELINE_JSON = (
    "gs://profitscout-lx6bb-pipeline-artifacts/training/training_pipeline.json"
)

# put each ad‑hoc run in its own folder
run_stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
PIPELINE_ROOT = (
    f"gs://profitscout-lx6bb-pipeline-artifacts/training/manual/{run_stamp}"
)

PARAMS = {
    "project":        "profitscout-lx6bb",
    "location":       "us-central1",
    "source_table":   "profit_scout.breakout_features",

    # locked HPs
    "pca_n": 128,
    "xgb_max_depth": 7,
    "xgb_min_child_weight": 3,
    "xgb_subsample": 0.9,
    "logreg_c": 0.001,
    "blend_weight": 0.6,
    "learning_rate": 0.02,
    "gamma": 0.10,
    "colsample_bytree": 0.9,
    "alpha": 1e-5,
    "reg_lambda": 2e-5,
    "focal_gamma": 2.0,

    # feature‑selection OFF, full data ON
    "auto_prune": "false",
    "top_k_features": 0,
    "use_full_data": "true",
}

job = aiplatform.PipelineJob(
    display_name=f"profitscout-fullfit-{run_stamp}",
    template_path=PIPELINE_JSON,
    pipeline_root=PIPELINE_ROOT,
    parameter_values=PARAMS,
    enable_caching=False,
)

job.run(sync=True)   # <-- change False → True
print("Launched on-demand training job:", job.resource_name)
print("Logs & artefacts at:", PIPELINE_ROOT)