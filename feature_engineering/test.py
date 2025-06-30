"""
Quick sanity-check for:
  1. Markdown parser (_parse_sections).
  2. Gemini embedding endpoint.

Run:  python test.py

Prereqs:
  • GOOGLE_CLOUD_PROJECT  or PROJECT_ID env variable set
  • Application default creds (or service-account key) active
  • google-cloud-aiplatform ≥ 1.100.0 in the venv
"""

import os, json, re, logging, pathlib, textwrap
from vertexai.language_models import TextEmbeddingModel
import vertexai

# ── config ────────────────────────────────────────────────
PROJECT_ID = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
REGION     = "us-central1"                  # change if you enabled Gemini elsewhere
TEST_FILE  = pathlib.Path("test.txt")       # same dir as this script

# ── same regex table you use in production ────────────────
SECTION_PATTERNS = {
    "key_financial_metrics"  : re.compile(r"^#{1,6}\s*Key Financial Metrics\s*:?",  re.I),
    "key_discussion_points"  : re.compile(r"^#{1,6}\s*Key Discussion Points\s*:?",  re.I),
    "sentiment_tone"         : re.compile(r"^#{1,6}\s*Sentiment Tone\s*:?",         re.I),
    "short_term_outlook"     : re.compile(r"^#{1,6}\s*Short\-?Term Outlook\s*:?",   re.I),
    "forward_looking_signals": re.compile(r"^#{1,6}\s*Forward\-?Looking Signals\s*:?", re.I),
}

def _parse_sections(md: str) -> dict:
    """Return a dict of section_name → text."""
    sections = {k: "" for k in SECTION_PATTERNS}
    current  = None
    for line in md.splitlines():
        hit = next((k for k, rx in SECTION_PATTERNS.items() if rx.match(line.strip())), None)
        current = hit or current
        if current and hit is None:
            sections[current] += line + "\n"
    return sections

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 1️⃣  Load local test file
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"{TEST_FILE.resolve()} not found")
    text = TEST_FILE.read_text()

    # 2️⃣  Parse sections
    sections = _parse_sections(text)
    logging.info("Parsed sections:")
    for k, v in sections.items():
        logging.info("  %-25s -> %4d chars", k, len(v.strip()))
        sample = textwrap.shorten(v.strip().replace("\n", " "), width=120) or "(blank)"
        logging.debug("    %s", sample)

    # 3️⃣  Initialise Vertex AI + model
    vertexai.init(project=PROJECT_ID, location=REGION)
    model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

    # 4️⃣  Call embeddings for each non-empty section
    for name, body in sections.items():
        body = body.strip()
        if not body:
            print(f"{name}: EMPTY — skipped")
            continue

        vec = model.get_embeddings([body])[0].values   # single-text call
        print(f"{name}: len={len(vec)}, first5={vec[:5]}")

if __name__ == "__main__":
    main()
