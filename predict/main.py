import os
import logging
import argparse
from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO)

def main(args):
    """Main batch prediction script."""
    aiplatform.init(project=args.project_id, location=args.region)

    logging.info(f"Fetching model: {args.model_name}")
    # Get the latest version of the model from the Model Registry
    model = aiplatform.Model(model_name=args.model_name)

    logging.info(f"Starting batch prediction job from {args.source_table} to {args.destination_table}")
    batch_predict_job = model.batch_predict(
        job_display_name=f'prediction_job_{args.model_name}',
        bigquery_source=f'bq://{args.project_id}.{args.source_table}',
        bigquery_destination_prefix=f'bq://{args.project_id}.{args.destination_table}',
        instances_format='bigquery',
        predictions_format='bigquery',
        machine_type='n1-standard-4',
    )
    
    batch_predict_job.wait()
    logging.info("Batch prediction job complete.")
    logging.info(f"State: {batch_predict_job.state}")
    logging.info(f"Output table: {batch_predict_job.output_info.bigquery_output_table}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', type=str, required=True, help='Google Cloud project ID')
    parser.add_argument('--region', type=str, default='us-central1', help='Google Cloud region')
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model in Vertex AI Model Registry')
    parser.add_argument('--source-table', type=str, required=True, help='BigQuery source table for prediction')
    parser.add_argument('--destination-table', type=str, required=True, help='BigQuery destination table for predictions')

    known_args, _ = parser.parse_known_args()
    main(known_args)