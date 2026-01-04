# FILE: training_job.py
from datetime import datetime
from google.cloud import aiplatform
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--direction", default="LONG", choices=["LONG", "SHORT"], help="Training direction for the pipeline (LONG or SHORT)")
args = parser.parse_args()

aiplatform.init(project="profitscout-lx6bb", location="us-central1")

PIPELINE_JSON = (
    f"gs://profitscout-lx6bb-pipeline-artifacts/training/training_pipeline_{args.direction.lower()}.json"
)

# put each ad‑hoc run in its own folder
run_stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
PIPELINE_ROOT = (
    f"gs://profitscout-lx6bb-pipeline-artifacts/training/manual/{args.direction.lower()}/{run_stamp}"
)

PARAMS = {
    "project":        "profitscout-lx6bb",
    "source_table":   "profit_scout.price_data",
    
    # Overrides (Optional)
    "xgb_max_depth": 6,
    "learning_rate": 0.06,
    "xgb_min_child_weight": 12,
    "xgb_subsample": 0.7,
    "colsample_bytree": 0.7,
    "gamma": 2.0,
    "alpha": 0.5,
    "reg_lambda": 2.0,
}

job = aiplatform.PipelineJob(
    display_name=f"profitscout-training-{args.direction.lower()}-{run_stamp}",
    template_path=PIPELINE_JSON,
    pipeline_root=PIPELINE_ROOT,
    parameter_values=PARAMS,
    enable_caching=False,
)

job.run(sync=True)
print("Launched training job:", job.resource_name)
print("Logs & artefacts at:", PIPELINE_ROOT)
