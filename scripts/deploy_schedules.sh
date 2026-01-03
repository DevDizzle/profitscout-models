#!/bin/bash
PROJECT_ID="profitscout-lx6bb"
REGION="us-central1"
ACCESS_TOKEN=$(gcloud auth print-access-token)

echo "Deploying Inference Schedule..."
curl -X POST \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d @scripts/schedule_inference.json \
    "https://$REGION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$REGION/schedules"

echo -e "\n\nDeploying Training Schedule..."
curl -X POST \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d @scripts/schedule_training.json \
    "https://$REGION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$REGION/schedules"
