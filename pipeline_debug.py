import os
from kfp import dsl
from kfp.v2 import compiler
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.wait_gcp_resources import WaitGcpResourcesOp

# --- Configuration ---
PROJECT_ID = "profitscout-lx6bb"
REGION = "us-central1"
PIPELINE_ROOT = "gs://profit-scout-pipeline-artifacts/pipeline-root"
BASE_OUTPUT_DIR = "gs://profit-scout-pipeline-artifacts/models"
TRAINER_IMAGE_URI = f"us-central1-docker.pkg.dev/{PROJECT_ID}/profit-scout-repo/trainer:latest"


@dsl.pipeline(
    name="profitscout-debug-pipeline-final",
    pipeline_root=PIPELINE_ROOT,
)
def debug_pipeline(
    project: str = PROJECT_ID,
    region: str = REGION,
    base_output_dir: str = BASE_OUTPUT_DIR
):
    """A pipeline to diagnose the outputs of the training job."""

    train_op = CustomTrainingJobOp(
        project=project,
        location=region,
        display_name="profitscout-trainer-job-debug",
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": "n1-standard-4"},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": TRAINER_IMAGE_URI,
                    "args": [
                        f"--project-id={project}",
                        "--source-table=dummy",
                        "--metadata-table=dummy",
                        # THIS LINE IS ADDED BACK IN:
                        "--model-bucket=dummy-bucket",
                    ],
                },
            }
        ],
        base_output_directory=base_output_dir,
    )

    wait_op = WaitGcpResourcesOp(
        gcp_resources=train_op.outputs["gcp_resources"]
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=debug_pipeline,
        package_path="debug_pipeline.json",
    )
    print("Debug pipeline compiled successfully to debug_pipeline.json")