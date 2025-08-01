# ───────────────────────── Trainer Dockerfile ─────────────────────────
FROM python:3.9-slim

# System-level prerequisite for XGBoost’s OpenMP binary
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# App code lives here
WORKDIR /app

# Python deps ─ first your list, then the pinned XGBoost build
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
                 xgboost==1.6.2        # ← exact same version as predictor

# Copy training script
COPY main.py .

# Default entry-point
ENTRYPOINT ["python3", "main.py"]
