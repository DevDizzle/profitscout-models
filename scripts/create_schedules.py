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
    
    # Define the pipeline job specification
    # We use .from_pipeline_func logic essentially, but here we just point to the JSON
    
    schedule = aiplatform.Schedule.create(
        display_name=display_name,
        cron=cron_expression,
        pipeline_job=aiplatform.PipelineJob(
            display_name=display_name,
            template_path=template_path,
            pipeline_root=pipeline_root,
            parameter_values=parameter_values,
            enable_caching=False
        ),
        max_concurrent_run_count=1
    )
    
    print(f"Schedule created: {schedule.resource_name}")
    return schedule

if __name__ == "__main__":
    # 1. Inference Schedule: Mon-Fri at 6:30 PM EST (23:30 UTC)
    # Cron: "30 23 * * 1-5"
    create_schedule(
        display_name="profitscout-daily-inference",
        cron_expression="30 23 * * 1-5", 
        template_path=f"{BUCKET}/inference/inference_pipeline.json",
        pipeline_root=f"{BUCKET}/inference",
        parameter_values={
            "project": PROJECT_ID,
            "source_table": "profit_scout.price_data",
            "destination_table": "profit_scout.daily_predictions",
            # Point to the stable production path
            "model_dir": f"{BUCKET}/production/model"
        }
    )

    # 2. Training Schedule: Every Sunday at 2:00 PM UTC (9:00 AM EST)
    # Cron: "0 14 * * 0"
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
