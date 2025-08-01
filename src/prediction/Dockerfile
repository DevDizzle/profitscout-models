# ───────────────────────── Predictor Dockerfile ─────────────────────────
# 1) Start from a lightweight base
FROM python:3.9-slim

# 2) Install the one native dependency XGBoost needs on Debian/Ubuntu:
#    libgomp1  – OpenMP runtime (libgomp.so)                     ──▶ fixes the
#    “libgomp.so.1: cannot open shared object file” import error.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 3) Copy requirements and pin XGBoost so trainer & predictor agree
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
                 xgboost==1.6.2        # ← explicit pin (Option A)

# 4) Copy predictor code and set entrypoint
COPY main.py .
ENTRYPOINT ["python3", "main.py"]
