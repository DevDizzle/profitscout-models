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

    # Use trainer defaults for HPs (no overrides here)

    # Feature selection: Auto-prune on, top_k off, full data off for validation
    "auto_prune": "true",
    "top_k_features": 0,
    "metric_tol": 0.002,
    "prune_step": 25,
    "use_full_data": "false",
}

job = aiplatform.PipelineJob(
    display_name=f"profitscout-feature-selection-{run_stamp}",
    template_path=PIPELINE_JSON,
    pipeline_root=PIPELINE_ROOT,
    parameter_values=PARAMS,
    enable_caching=False,
)

job.run(sync=True)
print("Launched on-demand feature selection job:", job.resource_name)
print("Logs & artefacts at:", PIPELINE_ROOT)