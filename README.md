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

### 3. Live Architecture  

```mermaid
graph TD
  subgraph "Data Ingestion & Feature Engineering (Event-Driven)"
    A[GCS: New Earnings Call Transcript] -->|Event Trigger| B(Cloud Function: discovery)
    B -->|Pub/Sub Message| C(Cloud Run: feature-engineering)
    C -->|Upserts Features| D[BigQuery: Feature Store]
  end

  subgraph "ML Lifecycle (Vertex AI Orchestrated)"
    E(Vertex AI Pipeline) -->|1️⃣ Triggers| F(Custom Training Job)
    D -->|Input Data| F
    F -->|Model Artifact| G[GCS Bucket]
    E -->|2️⃣ Import & Register| H(Vertex AI Model Registry)
    G --> H
    E -->|3️⃣ Batch Predict| I(Batch Prediction Job)
    H -->|Registered Model| I
    D -->|Prediction Data| I
    I -->|Writes| J[BigQuery: Prediction Table]
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

---

### 6. Service Breakdown  

| **Component**        | **Type**               | **Trigger / Orchestrator** | **Purpose**                                                     |
|----------------------|------------------------|----------------------------|----------------------------------------------------------------|
| `discovery`          | Cloud Function         | GCS event                  | Detect new transcript; publish Pub/Sub message                 |
| `feature-engineering`| Cloud Run              | Pub/Sub                    | Build features; upsert into BigQuery                           |
| `training-job`       | Vertex AI Custom Job   | Vertex AI Pipeline         | Train XGBoost; save artifact to GCS                            |
| `model-upload`       | Vertex AI Pipeline Op  | Vertex AI Pipeline         | Import & register model version                                |
| `batch-prediction`   | Vertex AI Pipeline Op  | Vertex AI Pipeline         | Score feature table; write results to BigQuery                 |

---

### 7. Setup & Deployment  

> **Prerequisites**  
> * gcloud SDK configured  
> * GCP project with Cloud Functions, Cloud Run, Artifact Registry, BigQuery & Vertex AI APIs enabled  

1. **Clone the repo**  
   ```bash
   git clone <YOUR_REPO_URL>
   cd profitscout-ml-pipeline

2. **Build & push the trainer image**  
   ```bash
   cd trainer
   gcloud builds submit \
     --tag us-central1-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/trainer:latest .

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

4. **Compile & run the pipeline**  
   ```bash
   python pipeline.py   # generates profitscout_pipeline.json

Open **Vertex AI → Pipelines** in the Google Cloud Console, upload `profitscout_pipeline.json`, and start a run.

### 8. How to Run the Pipeline  

**Data Processing (fully automated)**  
Upload a new `.txt` transcript to `gs://<BUCKET>/earnings-call-summaries/` — the event chain ingests & engineers features end-to-end.

**ML Training & Prediction (manual or scheduled)**  
```bash
gcloud ai pipelines run --pipeline-file profitscout_pipeline.json \
    --region us-central1 \
    --project YOUR_PROJECT
```
