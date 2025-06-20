"""utils/config.py
Shared configuration & helper utilities for all Profit‑Scout Vertex AI batch jobs.
Keeps notebook code DRY and ensures every container step can import the
same BigQuery / GCS clients and common math helpers.
"""

from __future__ import annotations

import os, time
from functools import lru_cache
from typing import Tuple, List

import numpy as np
from numpy.linalg import norm
from google.cloud import bigquery, storage
from tenacity import retry, stop_after_attempt, wait_exponential
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────

def env(name: str, default: str | None = None, *, required: bool = False) -> str:
    """Fetch **name** from `os.environ`, fail fast if required and missing."""
    val = os.getenv(name, default)
    if required and val is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val

PROJECT_ID: str = env("PROJECT_ID", required=True)
BQ_DATASET: str = env("BQ_DATASET", "profit_scout")
GCS_BUCKET: str = env("GCS_BUCKET", required=True)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy‑singletons (Application Default Credentials are picked up automatically)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def bq_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)

@lru_cache(maxsize=None)
def storage_client() -> storage.Client:
    return storage.Client(project=PROJECT_ID)

# Convenience wrappers --------------------------------------------------------

def table(name: str) -> str:
    """Return fully‑qualified BigQuery table reference inside default dataset."""
    return name if "." in name else f"{PROJECT_ID}.{BQ_DATASET}.{name}"

def bucket() -> storage.bucket.Bucket:  # type: ignore[attr-defined]
    return storage_client().bucket(GCS_BUCKET)

# ─────────────────────────────────────────────────────────────────────────────
# Numeric helpers (re‑used by training & scoring steps)
# ─────────────────────────────────────────────────────────────────────────────

def winsorise(a: np.ndarray, q: Tuple[float, float] = (1, 99)) -> np.ndarray:
    """Clip *a* in‑place to the *q*th and *100‑q*th percentiles (default 1–99)."""
    lo, hi = np.nanpercentile(a, q)
    return np.clip(a, lo, hi)


def safe_cosine(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised cosine similarity that guards against division‑by‑zero."""
    num = (u * v).sum(axis=1)
    den = norm(u, axis=1) * norm(v, axis=1)
    den = np.where(den == 0, 1, den)
    return num / den

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight request rate limiter + retry wrapper (for FMP API etc.)
# ─────────────────────────────────────────────────────────────────────────────

class Limiter:
    """Simple in‑memory rate limiter (requests‐per‐period)."""

    def __init__(self, rate: int = 45, period: float = 60.0):
        self.rate = rate
        self.period = period
        self.ts: List[float] = []

    def hit(self):
        now = time.time()
        self.ts = [t for t in self.ts if now - t < self.period]
        if len(self.ts) >= self.rate:
            time.sleep(self.period - (now - self.ts[0]))
        self.ts.append(now)

# Global limiter instance -----------------------------------------------------
_limiter = Limiter()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def http_get(url: str, *, timeout: int = 20, rate_limit: bool = True, **kw):
    """Wrapper around *requests.get* with retry + optional rate limiting."""
    if rate_limit:
        _limiter.hit()
    resp = requests.get(url, timeout=timeout, **kw)
    resp.raise_for_status()
    return resp.json()

# Public re‑exports -----------------------------------------------------------
__all__ = [
    "env", "PROJECT_ID", "BQ_DATASET", "GCS_BUCKET",
    "bq_client", "storage_client", "table", "bucket",
    "winsorise", "safe_cosine", "Limiter", "http_get",
]
