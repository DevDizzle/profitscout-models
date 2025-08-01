# FILE: training_pipeline.py
from kfp import dsl, compiler
from google_cloud_pipeline_components.v1.custom_job import (
    create_custom_training_job_from_component,
)

# ─────────────────── Config ───────────────────
PROJECT_ID = "profitscout-lx6bb"
REGION = "us-central1"
PIPELINE_ROOT = f"gs://{PROJECT_ID}-pipeline-artifacts/training"
TRAINER_IMAGE_URI = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/profit-scout-repo/trainer:latest"
)
MODEL_ARTIFACT_DIR = f"{PIPELINE_ROOT}/model-artifacts"

# ───────────────── Component ─────────────────
@dsl.container_component
def training_task(
    # core params
    project: str,
    source_table: str,
    # hyper‑params
    pca_n: int,
    xgb_max_depth: int,
    xgb_min_child_weight: int,
    xgb_subsample: float,
    logreg_c: float,
    blend_weight: float,
    learning_rate: float,
    gamma: float,
    colsample_bytree: float,
    alpha: float,
    reg_lambda: float,
    focal_gamma: float,
    # NEW feature‑selection flags
    top_k_features: int,
    auto_prune: str,       # "true"/"false" for pipeline UI toggle
    metric_tol: float,
    prune_step: int,
    use_full_data: str,    # "true"/"false"
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE_URI,
        command=["python3", "main.py"],
        args=[
            "--project-id", project,
            "--source-table", source_table,

            # hyper‑params
            "--pca-n", pca_n,
            "--xgb-max-depth", xgb_max_depth,
            "--xgb-min-child-weight", xgb_min_child_weight,
            "--xgb-subsample", xgb_subsample,
            "--logreg-c", logreg_c,
            "--blend-weight", blend_weight,
            "--learning-rate", learning_rate,
            "--gamma", gamma,
            "--colsample-bytree", colsample_bytree,
            "--alpha", alpha,
            "--reg-lambda", reg_lambda,
            "--focal-gamma", focal_gamma,

            # feature‑selection flags (always present, value is "true"/"false")
            "--top-k-features", top_k_features,
            "--auto-prune", auto_prune,
            "--metric-tol", metric_tol,
            "--prune-step", prune_step,
            "--use-full-data", use_full_data,
        ],
    )

training_op = create_custom_training_job_from_component(
    component_spec=training_task,
    display_name="profitscout-training-job",
    machine_type="n1-standard-16",
    replica_count=1,
    base_output_directory=MODEL_ARTIFACT_DIR,
)

# ───────────────── Pipeline ─────────────────
@dsl.pipeline(
    name="profitscout-standard-training-pipeline",
    description="Train and save ProfitScout model artifacts with HPO‑selected hyperparameters and optional feature selection.",
    pipeline_root=PIPELINE_ROOT,
)
def training_pipeline(
    project: str = PROJECT_ID,
    location: str = REGION,
    source_table: str = "profit_scout.breakout_features",
    # hyper‑param defaults (unchanged)
    pca_n: int = 128,
    xgb_max_depth: int = 7,
    xgb_min_child_weight: int = 2,
    xgb_subsample: float = 0.9,
    logreg_c: float = 0.001,
    blend_weight: float = 0.6,
    learning_rate: float = 0.02,
    gamma: float = 0.10,
    colsample_bytree: float = 0.9,
    alpha: float = 1e-5,
    reg_lambda: float = 2e-5,
    focal_gamma: float = 2.0,
    # ---- NEW feature‑selection knobs ----
    top_k_features: int = 0,         # 0 = no fixed‑K pruning
    auto_prune: str = "false",       # "true" to enable MI auto‑prune
    metric_tol: float = 0.002,       # tolerated drop in PR‑AUC/Brier
    prune_step: int = 25,            # features removed per iteration
    use_full_data: str = "true",    # "true" to train on 100% of rows
):
    training_op(
        project=project,
        source_table=source_table,
        pca_n=pca_n,
        xgb_max_depth=xgb_max_depth,
        xgb_min_child_weight=xgb_min_child_weight,
        xgb_subsample=xgb_subsample,
        logreg_c=logreg_c,
        blend_weight=blend_weight,
        learning_rate=learning_rate,
        gamma=gamma,
        colsample_bytree=colsample_bytree,
        alpha=alpha,
        reg_lambda=reg_lambda,
        focal_gamma=focal_gamma,
        # pass new flags
        top_k_features=top_k_features,
        auto_prune=auto_prune,
        metric_tol=metric_tol,
        prune_step=prune_step,
        use_full_data=use_full_data,
    )

# ───────────────── Compile ─────────────────
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="training_pipeline.json",
    )
    print("✓ Compiled training_pipeline.json")
