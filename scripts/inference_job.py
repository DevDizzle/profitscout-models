#!/usr/bin/env python3
"""
Launch a batch‑inference Vertex AI Pipeline run for ProfitScout.
Run: python scripts/inference_job.py
"""

from datetime import datetime
from google.cloud import aiplatform

aiplatform.init(
    project="profitscout-lx6bb",
    location="us-central1",
)

aiplatform.PipelineJob(
    display_name=f"profitscout-batch-inf-{datetime.utcnow():%Y%m%d%H%M%S}",
    template_path="gs://profitscout-lx6bb-pipeline-artifacts/inference/inference_pipeline.json",
    pipeline_root="gs://profitscout-lx6bb-pipeline-artifacts/inference",
    parameter_values={
        "project": "profitscout-lx6bb",
        "source_table": "profit_scout.price_data",
        "destination_table": "profit_scout.daily_predictions",
        # Point to the base directory containing 'long' and 'short' subfolders
        "model_base_dir": "gs://profitscout-lx6bb-pipeline-artifacts/production/model", 
    },
    enable_caching=False,
).run(sync=False)