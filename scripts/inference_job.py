#!/usr/bin/env python3
"""
Launch a batch‑inference Vertex AI Pipeline run for ProfitScout.
Save this as inference_job.py and run:  python inference_job.py
"""

from datetime import datetime
from google.cloud import aiplatform

# ────────────────── Vertex AI project/region ──────────────────
aiplatform.init(
    project="profitscout-lx6bb",
    location="us-central1",
)

# ────────────────── Launch the pipeline ──────────────────
aiplatform.PipelineJob(
    display_name=f"profitscout-batch-inf-{datetime.utcnow():%Y%m%d%H%M%S}",
    template_path="gs://profitscout-lx6bb-pipeline-artifacts/inference/inference_pipeline.json",
    pipeline_root="gs://profitscout-lx6bb-pipeline-artifacts/inference",
    parameter_values={
        "location": "US",
        "model_version_dir": "gs://profitscout-lx6bb-pipeline-artifacts/training/model-artifacts/model",  # ← updated
        "top_k_features": 0,
        "auto_prune": "false",
    },
    enable_caching=False,
).run(sync=False)
