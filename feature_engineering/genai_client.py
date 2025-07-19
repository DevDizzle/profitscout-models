"""
Tiny wrapper around the Vertex AI GenAI SDK, reused across the pipeline.

Usage:
    from feature_engineering.genai_client import generate
    text = generate("Return ONLY 'hello'")
"""

import logging, sys
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google import genai
from google.genai import types

# ── CONFIG  ────────────────────────────────
PROJECT_ID        = "profitscout-lx6bb"
LOCATION          = "global"
MODEL_NAME        = "gemini-2.0-flash"
TEMPERATURE       = 0.2
MAX_OUTPUT_TOKENS = 8
CANDIDATE_COUNT   = 1
# ------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
_log = logging.getLogger(__name__)

def _init_client() -> genai.Client | None:
    try:
        _log.info("Initialising Vertex AI GenAI client …")
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
            http_options=types.HttpOptions(api_version="v1"),
        )
        _log.info("Vertex AI client ready.")
        return client
    except Exception as e:
        _log.critical("FAILED to initialise Vertex AI client: %s", e, exc_info=True)
        print(f"CRITICAL: Vertex AI init failure: {e}", file=sys.stderr)
        return None

_client = _init_client()

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def generate(prompt: str) -> str:
    if _client is None:
        raise RuntimeError("Vertex AI client failed to initialise.")

    cfg = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        candidate_count=CANDIDATE_COUNT,
    )
    resp = _client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=cfg,
    )
    return resp.text.strip()
