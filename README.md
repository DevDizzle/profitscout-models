# ProfitScout ML Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-brightgreen.svg)
![Vertex AI](https://img.shields.io/badge/Vertex_AI-Pipelines_&_Training-ff69b4.svg)

### 1. Project Overview  
The **ProfitScout ML Pipeline** is an end-to-end, *serverless* system on Google Cloud Platform that predicts short-term stock-price moves after quarterly earnings calls. It:  

* **Ingests** earnings-call transcripts.  
* **Engineers** rich features with NLP, sentiment analysis, technical indicators, and fundamental surprises.  
* **Orchestrates** a full ML lifecycle in Vertex AI—training, model registry, and batch prediction—showcasing production-grade MLOps.  

### 2. Core Features  
- **Event-Driven Ingestion**: Cloud Storage upload → Cloud Function trigger.  
- **Scalable Feature Engineering** (Cloud Run microservice):  
  - Gemini-Embedding-001 text embeddings  
  - Cloud Natural Language sentiment on full transcripts  
  - Historical technical indicators  
  - EPS-surprise & other fundamental enrichments  
- **BigQuery Feature Store**: single source of truth for training & scoring.  
- **Declarative Orchestration**: Vertex AI Pipelines manage the graph.  
- **Containerized, Reproducible Training** (Docker).
- **Automated Model Versioning & Deployment** to Vertex AI Model Registry.
- **Reliable Staging Pattern**: a `loader` service streams rows into a staging
  table while a scheduled `merger` service deduplicates and merges them into the
  final table.
- **Dead-letter Queues & Extended Ack Deadline** make Pub/Sub processing more
  resilient under heavy load.
- **Custom Predictor Service** simplifies batch predictions using the latest
  model artifacts.

### 3. Live Architecture  

```mermaid
graph TD
  subgraph "Event-Driven Processing"
    A[GCS: New Earnings Call Transcript] -->|Event Trigger| B(Cloud Function: discovery)
    B -->|Pub/Sub| C(Cloud Run: feature-engineering)
    C -->|Pub/Sub| D(Cloud Run: loader)
    D -->|Stream| E[BigQuery: Staging Table]
    F[Cloud Scheduler] -->|Invoke| G(Cloud Function: merger)
    E -->|Merge| H[BigQuery: Feature Store]
  end

  subgraph "ML Lifecycle (Vertex AI Orchestrated)"
    I(Vertex AI Pipeline) -->|1️⃣ Triggers| J(Custom Training Job)
    H -->|Input Data| J
    J -->|Model Artifact| K[GCS Bucket]
    I -->|2️⃣ Import & Register| L(Vertex AI Model Registry)
    K --> L
    I -->|3️⃣ Batch Predict| M(Batch Prediction Job)
    L -->|Registered Model| M
    H -->|Prediction Data| M
    M -->|Writes| N[BigQuery: Prediction Table]
  end
  ```
  ### 4. Technology Stack  

| **Layer**            | **Choice**                                                                                           |
|----------------------|-------------------------------------------------------------------------------------------------------|
| Cloud                | Google Cloud Platform                                                                                 |
| ML Platform          | Vertex AI (Pipelines, Training, Model Registry, Batch Prediction)                                     |
| Compute              | Cloud Functions (Gen 2), Cloud Run                                                                    |
| Storage              | Cloud Storage, BigQuery                                                                               |
| Messaging            | Pub/Sub                                                                                                |
| Containerization     | Docker, Artifact Registry                                                                             |
| Language / ML Libs   | Python · scikit-learn · XGBoost · pandas                                                               |
| GCP SDKs             | google-cloud-aiplatform · kfp · storage · pubsub · bigquery                                            |
| AI/NLP               | Gemini-Embedding-001 · Cloud Natural Language API                                                     |

---

### 5. Key Design & Engineering Decisions  

- **Serverless, Event-Driven Ingestion** keeps costs low and scales automatically.  
- **Pub/Sub Decoupling** guarantees resilience—messages persist until processed.  
- **BigQuery Feature Store** enables reuse, decouples prep from training, and scales effortlessly.  
- **Declarative Vertex AI Pipelines** provide a visual, auditable DAG and easy re-runs.  
- **Containerized Training** eliminates “works-on-my-machine” drift via immutable Docker images.
- **Staging Table Pattern** separates streaming inserts from analytical tables, enabling safe deduplication and large-scale loads.

---

### 6. Service Breakdown  

| **Component**        | **Type**               | **Trigger / Orchestrator** | **Purpose**                                                     |
|----------------------|------------------------|----------------------------|----------------------------------------------------------------|
| `discovery`          | Cloud Function         | GCS event                  | Detect new transcript; publish Pub/Sub message                 |
| `feature-engineering`| Cloud Run              | Pub/Sub                    | Build features; publish to loader                              |
| `loader`             | Cloud Run              | Pub/Sub                    | Stream feature rows into BigQuery staging table                |
| `merger`             | Cloud Function         | Cloud Scheduler            | Deduplicate & merge staging table into feature store           |
| `training-job`       | Vertex AI Custom Job   | Vertex AI Pipeline         | Train XGBoost; save artifact to GCS                            |
| `model-upload`       | Vertex AI Pipeline Op  | Vertex AI Pipeline         | Import & register model version                                |
| `batch-prediction`   | Vertex AI Pipeline Op  | Vertex AI Pipeline         | Score feature table; write results to BigQuery                 |
| `predictor`          | Cloud Run              | Vertex AI Pipeline         | Serve custom batch predictions using the latest model artifacts |

---

### 7. Setup & Deployment  

> **Prerequisites**  
> * gcloud SDK configured  
> * GCP project with Cloud Functions, Cloud Run, Artifact Registry, BigQuery & Vertex AI APIs enabled  

1. **Clone the repo**  
   ```bash
   git clone <YOUR_REPO_URL>
   cd profitscout-ml-pipeline

2. **Build & push all container images**
   ```bash
   gcloud builds submit trainer \
     --tag us-central1-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/trainer:latest
   gcloud builds submit predictor \
     --tag us-central1-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/predictor:latest
   gcloud builds submit feature_engineering \
     --tag us-central1-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/feature-engineering:latest
   gcloud builds submit loader \
     --tag us-central1-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/loader:latest
   ```

3. **Deploy ingestion services**  
   ```bash
   # Cloud Function: discovery
   gcloud functions deploy discover_new_transcripts --gen2 --runtime python39 \
     --trigger-resource YOUR_BUCKET \
     --trigger-event google.storage.object.finalize \
     --region us-central1

   # Cloud Run: feature-engineering
   gcloud run deploy feature-engineering \
     --image us-central1-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/feature-engineering:latest \
     --region us-central1

   # Cloud Run: loader
   gcloud run deploy loader \
     --image us-central1-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/loader:latest \
     --region us-central1 \
     --set-env-vars DESTINATION_TABLE=YOUR_DATASET.staging_table

  # Cloud Run: predictor
  gcloud run deploy predictor \
    --image us-central1-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/predictor:latest \
    --region us-central1

   # Cloud Function: merger (scheduled)
  gcloud functions deploy merge_staging_to_final --gen2 --runtime python39 \
    --trigger-topic merge-features \
    --region us-central1 \
    --set-env-vars PROJECT_ID=YOUR_PROJECT,STAGING_TABLE=YOUR_DATASET.staging_table,FINAL_TABLE=YOUR_DATASET.feature_store
   ```

4. **Compile the pipelines**
   ```bash
   python create_training_pipeline.py
   python create_inference_pipeline.py
   python create_hpo_pipline.py
   ```

5. **Run a pipeline**
   ```bash
   gcloud ai pipelines run --pipeline-file training_pipeline.json \
       --region us-central1 \
       --project YOUR_PROJECT
   ```

### 8. How to Run the Pipeline  

**Data Processing (fully automated)**  
Upload a new `.txt` transcript to `gs://<BUCKET>/earnings-call-summaries/` — the event chain ingests & engineers features end-to-end.

**ML Training & Prediction (manual or scheduled)**  
```bash
gcloud ai pipelines run --pipeline-file training_pipeline.json \
    --region us-central1 \
    --project YOUR_PROJECT
# swap the pipeline file for inference_pipeline.json or hpo_pipeline.json as needed
```
