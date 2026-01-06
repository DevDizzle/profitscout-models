from google.cloud import aiplatform

PROJECT_ID = "profitscout-lx6bb"
REGION = "us-central1"
BUCKET = "gs://profitscout-lx6bb-pipeline-artifacts"

aiplatform.init(project=PROJECT_ID, location=REGION)

def create_schedule(
    display_name: str,
    cron_expression: str,
    template_path: str,
    pipeline_root: str,
    parameter_values: dict
):
    print(f"Creating schedule: {display_name}...")
    
    # Create the PipelineJob object
    pipeline_job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=template_path,
        pipeline_root=pipeline_root,
        parameter_values=parameter_values,
        enable_caching=False
    )

    # Create the PipelineJobSchedule wrapper
    schedule = aiplatform.PipelineJobSchedule(
        pipeline_job=pipeline_job,
        display_name=display_name
    )
    
    # Submit the creation request
    schedule.create(
        cron=cron_expression,
        max_concurrent_run_count=1
    )
    
    print(f"Schedule created: {schedule.resource_name}")
    return schedule

if __name__ == "__main__":
    # 1. Inference Schedule: Mon-Fri at 5:00 PM EST (22:00 UTC)
    # Cron: "0 22 * * 1-5"
    # create_schedule(
    #     display_name="profitscout-daily-inference",
    #     cron_expression="0 22 * * 1-5", 
    #     template_path=f"{BUCKET}/inference/inference_pipeline.json",
    #     pipeline_root=f"{BUCKET}/inference",
    #     parameter_values={
    #         "project": PROJECT_ID,
    #         "source_table": "profit_scout.price_data",
    #         "destination_table": "profit_scout.daily_predictions",
    #         # Point to the stable production path
    #         "model_base_dir": f"{BUCKET}/production/model"
    #     }
    # )

    # 2. Training Schedule: Every Sunday at 2:00 PM UTC (9:00 AM EST)
    # Cron: "0 14 * * 0"
    # Note: If this schedule already exists, it will duplicate unless cleaned up first.
    # For now, we allow it or user should clean it up if needed.
    create_schedule(
        display_name="profitscout-weekly-training",
        cron_expression="0 14 * * 0",
        template_path=f"{BUCKET}/training/training_pipeline.json",
        pipeline_root=f"{BUCKET}/training",
        parameter_values={
            "project": PROJECT_ID,
            "source_table": "profit_scout.price_data",
            "xgb_max_depth": 6,
            "learning_rate": 0.06,
            "xgb_min_child_weight": 12,
            "xgb_subsample": 0.7,
            "colsample_bytree": 0.7,
            "gamma": 2.0,
            "alpha": 0.5,
            "reg_lambda": 2.0,
        }
    )