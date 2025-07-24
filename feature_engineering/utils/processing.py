# ───────────────────────── imports ─────────────────────────
import os, json, logging, re, requests, textwrap, time, difflib
import pandas as pd, pandas_ta as ta
import numpy as np
from numpy.linalg import norm
from typing import Optional
from google.cloud import storage, aiplatform, bigquery, secretmanager
from vertexai.language_models import TextEmbeddingModel
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models import LdaModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ───────────────────────── init ────────────────────────────
PROJECT_ID = os.environ.get("PROJECT_ID")
aiplatform.init(project=PROJECT_ID, location="us-central1")
bq_client = bigquery.Client()
storage_client = storage.Client()
secret_client = secretmanager.SecretManagerServiceClient()
embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

logging.basicConfig(level=logging.INFO)

# ─────────────────────── helpers ───────────────────────────
def get_fmp_api_key() -> str | None:
    try:
        name = f"projects/{PROJECT_ID}/secrets/FMP_API_KEY/versions/latest"
        return secret_client.access_secret_version(request={"name": name}).payload.data.decode()
    except Exception as e:
        logging.error("get_fmp_api_key: %s", e)
        return None

def safe_cosine(u, v):
    """Calculate cosine similarity, handling potential zero vectors."""
    u, v = np.array(u), np.array(v)
    if u.ndim == 1: u = u.reshape(1, -1)
    if v.ndim == 1: v = v.reshape(1, -1)
    
    den = norm(u, axis=1) * norm(v, axis=1)
    if den == 0:
        return 0.0
    
    num = (u * v).sum(axis=1)
    return (num / den)[0]

SECTION_PATTERNS = {
    'key_financial_metrics': re.compile(r'^\s*(?:(?:#{1,6}\s*|\*\*)\s*)?Key Financial Metrics\b.*', re.I),
    'key_discussion_points': re.compile(r'^\s*(?:(?:#{1,6}\s*|\*\*)\s*)?Key Discussion Points\b.*', re.I),
    'sentiment_tone': re.compile(r'^\s*(?:(?:#{1,6}\s*|\*\*)\s*)?Sentiment Tone\b.*', re.I),
    'short_term_outlook': re.compile(r'^\s*(?:(?:#{1,6}\s*|\*\*)\s*)?Short\-?Term Outlook\b.*', re.I),
    'forward_looking_signals': re.compile(r'^\s*(?:(?:#{1,6}\s*|\*\*)\s*)?Forward\-?Looking Signals\b.*', re.I),
    'qa_summary': re.compile(r'^\s*(?:(?:#{1,6}\s*|\*\*)\s*)?Q&A Summary\b.*', re.I),
}

# For fuzzy fallback: Standard titles without extras
STANDARD_TITLES = {
    'key_financial_metrics': 'Key Financial Metrics',
    'key_discussion_points': 'Key Discussion Points',
    'sentiment_tone': 'Sentiment Tone',
    'short_term_outlook': 'Short-Term Outlook',
    'forward_looking_signals': 'Forward-Looking Signals',
    'qa_summary': 'Q&A Summary',
}

def _parse_sections(text_content: str) -> dict:
    # Normalize input: Strip, remove BOM, collapse multiple newlines
    text_content = text_content.strip().lstrip('\ufeff')
    text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
    
    sections = {k: "" for k in SECTION_PATTERNS}
    matched_count = 0
    
    # --- Attempt Strict JSON Parse ---
    try:
        data = json.loads(text_content)
        logging.info("JSON loaded successfully")
        summary_data = next(iter(data.values()))
        logging.info("Summary data keys: %s", list(summary_data.keys()))
        
        for section_key, text_value in summary_data.items():
            clean_key = re.sub(r'[^a-zA-Z0-9\s\-&]', '', section_key.strip())
            logging.info(f"Processing key: {section_key} (cleaned: {clean_key})")
            matched_section = next(
                (k for k, rx in SECTION_PATTERNS.items() if rx.search(clean_key)),
                None
            )
            if matched_section:
                sections[matched_section] = text_value.strip()
                matched_count += 1
            else:
                logging.warning(f"No regex match for key: {section_key}")
    
    except (json.JSONDecodeError, StopIteration, AttributeError) as e:
        logging.info(f"Strict JSON parse failed: {e}. Trying lenient JSON parse.")
        
        # --- Lenient JSON Parse (if looks like JSON but failed strict load) ---
        if text_content.startswith('{') and text_content.endswith('}'):
            try:
                # Improved pattern: Matches "key": "value" with escaped inner content
                pair_pattern = r'"([^"]+)":\s*"((?:\\.|[^"\\])*)"'
                pairs = re.findall(pair_pattern, text_content)
                
                for section_key, text_value in pairs:
                    clean_key = re.sub(r'[^a-zA-Z0-9\s\-&]', '', section_key.strip())
                    logging.info(f"Lenient processing key: {section_key} (cleaned: {clean_key})")
                    matched_section = next(
                        (k for k, rx in SECTION_PATTERNS.items() if rx.search(clean_key)),
                        None
                    )
                    if matched_section:
                        # Manual unescape for common cases (e.g., \")
                        text_value = text_value.replace('\\"', '"').replace('\\\\', '\\')
                        sections[matched_section] = text_value.strip()
                        matched_count += 1
            except Exception as le:
                logging.info(f"Lenient JSON parse failed: {le}. Falling back to Markdown.")
        
        # --- Markdown Regex Parsing ---
        current_section = None
        lines = text_content.splitlines()
        for i, line in enumerate(lines):
            clean_line = re.sub(r'[^a-zA-Z0-9\s\-&]', '', line.strip())
            matched_section = next(
                (k for k, rx in SECTION_PATTERNS.items() if rx.match(clean_line)),
                None
            )
            if matched_section:
                current_section = matched_section
                matched_count += 1
                # Improved: If value is on same line after :, append it
                if ':' in line:
                    value_start = line.find(':') + 1
                    sections[current_section] += line[value_start:].strip() + "\n"
            elif current_section:
                sections[current_section] += line + "\n"
    
    # --- Fuzzy Matching Fallback if No Matches ---
    if matched_count == 0:
        logging.warning("No sections matched via regex. Attempting fuzzy matching.")
        try:
            data = json.loads(text_content)
            summary_data = next(iter(data.values()))
            for section_key, text_value in summary_data.items():
                clean_key = re.sub(r'[^a-zA-Z0-9\s\-&]', '', section_key.strip().lower())
                best_match = difflib.get_close_matches(clean_key, [t.lower() for t in STANDARD_TITLES.values()], n=1, cutoff=0.8)
                if best_match:
                    matched_section = next(k for k, v in STANDARD_TITLES.items() if v.lower() == best_match[0])
                    sections[matched_section] = text_value.strip()
                    matched_count += 1
                    logging.info(f"Fuzzy matched {section_key} to {matched_section}")
        except (json.JSONDecodeError, StopIteration, AttributeError):
            # Fuzzy on lines for non-JSON
            current_section = None
            lines = text_content.splitlines()
            for i, line in enumerate(lines):
                clean_line = re.sub(r'[^a-zA-Z0-9\s\-&]', '', line.strip().lower())
                best_match = difflib.get_close_matches(clean_line, [t.lower() for t in STANDARD_TITLES.values()], n=1, cutoff=0.8)
                if best_match:
                    matched_section = next(k for k, v in STANDARD_TITLES.items() if v.lower() == best_match[0])
                    current_section = matched_section
                    matched_count += 1
                elif current_section:
                    sections[current_section] += line + "\n"
    
    # Final cleanup and logging
    for k in sections:
        sections[k] = sections[k].strip()
    
    logging.info("Parsed Sections Breakdown:")
    for k, v in sections.items():
        logging.info("  %-25s -> %4d chars", k, len(sections[k]))
    
    if matched_count == 0:
        logging.error("All sections empty after all attempts. Check summary format.")
    
    return sections

def get_embeddings(summary_gcs_path: str) -> dict:
    logging.info("Starting embedding generation for: %s", summary_gcs_path)
    failed_sections = []
    try:
        bucket, blob = summary_gcs_path.removeprefix("gs://").split("/", 1)
        text = storage_client.bucket(bucket).blob(blob).download_as_bytes().decode('utf-8', errors='replace')
        logging.info("Downloaded summary: %d chars, sample: %s", len(text), text[:500])
        
        sections = _parse_sections(text)
        results = {}

        for sec, txt in sections.items():
            txt = (txt or "").strip()
            key = f"{sec}_embedding"
            if not txt:
                logging.warning("Section '%s' is empty. Setting embedding to None.", sec)
                results[key] = None
                continue

            # Sanitize input: normalize whitespace, remove control characters
            txt = re.sub(r'\s+', ' ', txt)
            txt = ''.join(c for c in txt if ord(c) >= 32)
            logging.info("Section %s: text_len=%d, sample=%s", sec, len(txt), txt[:100])

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    emb = embedding_model.get_embeddings([txt], output_dimensionality=768)[0]
                    vec = list(emb.values)
                    if not vec:
                        raise ValueError("Empty embedding response")
                    vec_np = np.array(vec, dtype=float)
                    if len(vec_np) != 768:
                        raise ValueError(f"Unexpected embedding length: {len(vec_np)}, expected 768")
                    if not np.all(np.isfinite(vec_np)):
                        raise ValueError("Non-finite values in embedding")
                    norm = np.linalg.norm(vec_np)
                    if norm == 0:
                        raise ValueError("Zero-norm embedding")
                    vec_np = vec_np / norm
                    results[key] = vec_np.tolist()
                    logging.info("Embedded %s | len=%d | first5=%s | norm=%f", sec, len(vec_np), vec_np[:5], norm)
                    break
                except Exception as e:
                    logging.error("Embedding attempt %d/%d failed for %s: %s", attempt + 1, max_retries, sec, e)
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    else:
                        logging.error("Max retries exceeded for %s. Setting to None.", sec)
                        failed_sections.append(sec)
                        results[key] = None

        if failed_sections:
            logging.error("Sections with failed embeddings: %s", failed_sections)
            raise RuntimeError(f"Failed to generate embeddings for sections: {failed_sections}")
        return results
    except Exception as e:
        logging.error("get_embeddings failed: %s", e, exc_info=True)
        return {}

def get_earnings_call_date(transcript_gcs_path: str) -> dict:
    """Reads the original transcript JSON *only* to pull the 'date' field."""
    try:
        bucket, blob = transcript_gcs_path.removeprefix("gs://").split("/", 1)
        raw = storage_client.bucket(bucket).blob(blob).download_as_string()
        date_str = json.loads(raw).get("date")
        if date_str:
            date_str = date_str.split(" ")[0]
        return {"earnings_call_date": date_str}
    except Exception as e:
        logging.error("get_earnings_call_date: %s", e, exc_info=True)
        return {"earnings_call_date": None}

def get_stock_metadata(ticker: str) -> dict:
    try:
        table = os.getenv("METADATA_TABLE", "profitscout-lx6bb.profit_scout.stock_metadata")
        query = f"SELECT sector, industry FROM `{table}` WHERE ticker = @ticker LIMIT 1"
        df = bq_client.query(
            query,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
            ),
        ).to_dataframe()
        return df.iloc[0].to_dict() if not df.empty else {"sector": None, "industry": None}
    except Exception as e:
        logging.error("get_stock_metadata: %s", e)
        return {"sector": None, "industry": None}

def get_eps_surprise(api_key: Optional[str], ticker: str) -> dict:
    if not api_key:
        return {"eps_surprise": None, "eps_missing": True}
    url = (f"https://financialmodelingprep.com/api/v3/"
           f"earnings-surprises/{ticker}?apikey={api_key}")
    try:
        data = requests.get(url, timeout=10).json()
        if not data:
            raise ValueError("empty response")
        actual, estimate = data[0]["actualEarningResult"], data[0]["estimatedEarning"]
        surprise = actual - estimate
        surprise_pct = surprise / abs(estimate) if estimate else None
        return {"eps_surprise": float(surprise_pct) if surprise_pct is not None else None, "eps_missing": False}
    except Exception as e:
        logging.error("get_eps_surprise: %s", e)
        return {"eps_surprise": None, "eps_missing": True}

def get_price_technicals(ticker: str, call_date: str) -> dict:
    try:
        price_table = os.environ["PRICE_TABLE"]
        call_date = call_date.split(" ")[0]
        query = f"SELECT date, open, high, low, adj_close FROM `{price_table}` WHERE ticker = @ticker AND date <= @call_date ORDER BY date DESC LIMIT 100"
        df = (
            bq_client.query(
                query,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                        bigquery.ScalarQueryParameter("call_date", "DATE", call_date),
                    ]
                ),
            ).to_dataframe().set_index("date").sort_index()
            .rename(columns={"open": "Open", "high": "High", "low": "Low", "adj_close": "Close"})
        )
        if df.empty:
            return {}
        df["Adj Close"] = df["Close"]
        df.ta.sma(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.adx(length=14, append=True, high=df["High"], low=df["Low"], close=df["Close"])
        latest, prev = df.iloc[-1], df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        delta = lambda cur, pre: float(cur - pre) if pd.notna(cur) and pd.notna(pre) else None
        return {
            "adj_close_on_call_date": float(latest["Adj Close"]),
            "sma_20": float(latest.get("SMA_20")) if pd.notna(latest.get("SMA_20")) else None,
            "ema_50": float(latest.get("EMA_50")) if pd.notna(latest.get("EMA_50")) else None,
            "rsi_14": float(latest.get("RSI_14")) if pd.notna(latest.get("RSI_14")) else None,
            "adx_14": float(latest.get("ADX_14")) if pd.notna(latest.get("ADX_14")) else None,
            "sma_20_delta": delta(latest.get("SMA_20"), prev.get("SMA_20")),
            "ema_50_delta": delta(latest.get("EMA_50"), prev.get("EMA_50")),
            "rsi_14_delta": delta(latest.get("RSI_14"), prev.get("RSI_14")),
            "adx_14_delta": delta(latest.get("ADX_14"), prev.get("ADX_14")),
        }
    except Exception as e:
        logging.error("get_price_technicals: %s", e)
        return {}

def get_max_close_30d(ticker: str, call_date: str) -> dict:
    try:
        start_date_ts = pd.Timestamp(call_date.split(" ")[0])
        end_date_window_ts = start_date_ts + pd.Timedelta(days=30)
        today_ts = pd.Timestamp.now().floor('D')
        if today_ts < end_date_window_ts:
            return {"max_close_30d": None}
        price_table = os.environ["PRICE_TABLE"]
        max_q = f"SELECT MAX(adj_close) AS max_price FROM `{price_table}` WHERE ticker=@ticker AND date BETWEEN @start_date AND @end_date"
        max_df = bq_client.query(
            max_q,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                    bigquery.ScalarQueryParameter("start_date", "DATE", start_date_ts.strftime('%Y-%m-%d')),
                    bigquery.ScalarQueryParameter("end_date", "DATE", end_date_window_ts.strftime('%Y-%m-%d')),
                ]
            )
        ).to_dataframe()
        max_price = float(max_df.at[0, "max_price"]) if not max_df.empty and pd.notna(max_df.at[0, "max_price"]) else None
        return {"max_close_30d": max_price}
    except Exception as e:
        logging.error("get_max_close_30d failed for ticker %s: %s", ticker, e)
        return {"max_close_30d": None}

def get_lda_topics(summary_gcs_path: str) -> dict:
    stop_words = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    }
    try:
        bucket, blob = summary_gcs_path.removeprefix("gs://").split("/", 1)
        text = storage_client.bucket(bucket).blob(blob).download_as_text()
        if not text.strip():
            logging.warning("Empty summary for LDA at %s", summary_gcs_path)
            return {f"lda_topic_{i}": 0.0 for i in range(10)}
        full_text = text.strip()
        tokens = [word for word in simple_preprocess(full_text) if word not in stop_words and len(word) > 2]
        if not tokens:
            return {f"lda_topic_{i}": 0.0 for i in range(10)}
        dictionary = corpora.Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10, iterations=100, random_state=42)
        topic_probs = lda.get_document_topics(corpus[0], minimum_probability=0.0)
        probs = [0.0] * 10
        for topic_id, prob in topic_probs:
            probs[topic_id] = float(prob)
        return {f"lda_topic_{i}": probs[i] for i in range(10)}
    except Exception as e:
        logging.error("get_lda_topics: %s", e)
        return {f"lda_topic_{i}": None for i in range(10)}

def get_finbert_sentiments(summary_gcs_path: str) -> dict:
    """
    Run FinBERT sentiment over each parsed section of the summary.
    Returns dict like: {f"{section}_{cls}_prob": float} for cls in {pos, neg, neu}.
    Falls back to neutral on any failure.
    """
    import logging
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from google.cloud import storage

    # Initialize defaults
    results = {f"{sec}_{sent}_prob": 0.0 for sec in SECTION_PATTERNS for sent in ['pos', 'neg', 'neu']}

    try:
        # --- Load text from GCS ---
        bucket_name, blob_name = summary_gcs_path.removeprefix("gs://").split("/", 1)
        storage_client = storage.Client()
        text = storage_client.bucket(bucket_name).blob(blob_name).download_as_text()
        sections = _parse_sections(text)

        # --- Load FinBERT on CPU (no meta tricks) ---
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert",
            torch_dtype=torch.float32,   # explicit, avoids half/meta weirdness
            low_cpu_mem_usage=False      # ensure real weights are loaded
        )
        model.eval()  # stays on CPU by default

        def score_text(t: str):
            """Tokenize (chunk if needed) and return averaged probs."""
            if not t.strip():
                return [0.0, 0.0, 1.0]  # neutral

            # Tokenize once to see length
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=512)
            if enc["input_ids"].shape[1] <= 512:
                with torch.no_grad():
                    logits = model(**enc).logits
                return torch.softmax(logits, dim=-1)[0].tolist()

            # Chunk if longer than 512 tokens
            tokens = tokenizer.encode(t, add_special_tokens=True)
            chunk_size = 510  # keep room for [CLS] & [SEP]
            probs_list = []

            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i:i + chunk_size]
                # Re-add special tokens per chunk
                chunk_tokens = tokenizer.build_inputs_with_special_tokens(chunk_tokens)
                chunk_enc = {
                    "input_ids": torch.tensor([chunk_tokens]),
                    "attention_mask": torch.ones(1, len(chunk_tokens), dtype=torch.long),
                }
                with torch.no_grad():
                    logits = model(**chunk_enc).logits
                probs_list.append(torch.softmax(logits, dim=-1)[0])

            mean_probs = torch.stack(probs_list, dim=0).mean(dim=0).tolist()
            return mean_probs

        for sec, txt in sections.items():
            logging.info(f"Processing FinBERT for section {sec} ({len(txt)} chars)")
            probs = score_text(txt)
            results[f"{sec}_pos_prob"] = probs[0]
            results[f"{sec}_neg_prob"] = probs[1]
            results[f"{sec}_neu_prob"] = probs[2]

        return results

    except Exception as e:
        logging.error("get_finbert_sentiments failed: %s", e, exc_info=True)
        # Neutral fallback
        return {
            f"{sec}_{sent}_prob": 1.0 if sent == 'neu' else 0.0
            for sec in SECTION_PATTERNS for sent in ['pos', 'neg', 'neu']
        }

def create_features(message: dict) -> dict | None:
    """Orchestrates feature creation and returns a complete row for BigQuery."""
    required_keys = ["ticker", "quarter_end_date", "summary_gcs_path", "transcript_gcs_path"]
    if not all(key in message and message[key] is not None for key in required_keys):
        logging.error(f"FATAL: Message missing required keys: {message}")
        return None

    ticker = message.get("ticker")
    row = {"ticker": ticker, "quarter_end_date": message.get("quarter_end_date")}

    row.update(get_stock_metadata(ticker))
    row.update(get_embeddings(message.get("summary_gcs_path")))
    row.update(get_earnings_call_date(message.get("transcript_gcs_path")))
    row.update(get_eps_surprise(get_fmp_api_key(), ticker))

    # Compute sentiment_score from FinBERT results
    finbert_results = get_finbert_sentiments(message["summary_gcs_path"])
    pos_probs = [finbert_results[f"{sec}_pos_prob"] for sec in SECTION_PATTERNS]
    valid_probs = [p for p in pos_probs if p > 0]  # Exclude neutral-only (1.0 neu)
    row["sentiment_score"] = float(np.mean(valid_probs)) if valid_probs else 0.0
    row.update(finbert_results)

    if row.get("earnings_call_date"):
        row["earnings_call_date"] = row["earnings_call_date"].split(" ")[0]
        row.update(get_price_technicals(ticker, row["earnings_call_date"]))
        row.update(get_max_close_30d(ticker, row["earnings_call_date"]))
    else:
        logging.warning(f"No earnings_call_date; skipped price features for ticker {ticker}")

    # --- Engineered Features ---
    logging.info(f"Engineering features for {ticker}...")
    
    # Combined categorical feature
    sector = row.get("sector", "UNK_SEC") or "UNK_SEC"
    industry = row.get("industry", "UNK_IND") or "UNK_IND"
    row["industry_sector"] = f"{sector}_{industry}"

    # Embedding stats and similarities
    EMB_COLS_NAMES = [
        "key_financial_metrics_embedding", "key_discussion_points_embedding",
        "sentiment_tone_embedding", "short_term_outlook_embedding",
        "forward_looking_signals_embedding", "qa_summary_embedding",
    ]
    vecs = {c: np.array(row[c]) if row.get(c) is not None else np.array([]) for c in EMB_COLS_NAMES}
    
    for col_name, vec in vecs.items():
        base_name = col_name.replace("_embedding", "")
        if col_name == "forward_looking_signals_embedding":
            logging.info(f"Embedding: size={vec.size}, first5={vec[:5] if vec.size > 0 else '[]'}, isfinite={np.all(np.isfinite(vec)) if vec.size > 0 else False}")
        try:
            if vec.size != 768 or not np.all(np.isfinite(vec)):
                logging.warning(f"Invalid embedding for {col_name}: size={vec.size}. Setting stats to 0.0 (investigate API issue).")
                row[f"{base_name}_norm"] = 0.0
                row[f"{base_name}_mean"] = 0.0
                row[f"{base_name}_std"] = 0.0
            else:
                row[f"{base_name}_norm"] = float(norm(vec))
                row[f"{base_name}_mean"] = float(vec.mean())
                row[f"{base_name}_std"] = float(vec.std())
        except Exception as e:
            logging.error(f"Error computing stats for {col_name}: {e}")
            row[f"{base_name}_norm"] = 0.0
            row[f"{base_name}_mean"] = 0.0
            row[f"{base_name}_std"] = 0.0

    sim_pairs = [
        ("cos_fin_disc", "key_financial_metrics_embedding", "key_discussion_points_embedding"),
        ("cos_fin_tone", "key_financial_metrics_embedding", "sentiment_tone_embedding"),
        ("cos_disc_short", "key_discussion_points_embedding", "short_term_outlook_embedding"),
        ("cos_short_fwd", "short_term_outlook_embedding", "forward_looking_signals_embedding"),
    ]
    for key, col1, col2 in sim_pairs:
        row[key] = float(safe_cosine(vecs.get(col1, []), vecs.get(col2, [])))

    # Null indicator for EPS
    row["eps_surprise_isnull"] = 1 if row.get("eps_surprise") is None else 0

    # Price and technical ratios
    if row.get("adj_close_on_call_date") and row.get("sma_20") and row.get("sma_20") != 0:
        row["price_sma20_ratio"] = float(row["adj_close_on_call_date"] / row["sma_20"])
    if row.get("ema_50") and row.get("sma_20") and row.get("sma_20") != 0:
        row["ema50_sma20_ratio"] = float(row["ema_50"] / row["sma_20"])
    if row.get("rsi_14") is not None:
        row["rsi_centered"] = float(row["rsi_14"] - 50)
        if row.get("sentiment_score") is not None:
            row["sent_rsi"] = float(row["sentiment_score"] * row["rsi_centered"])
    if row.get("adx_14") is not None:
        row["adx_log1p"] = float(np.log1p(row["adx_14"]))

    row.update(get_lda_topics(message["summary_gcs_path"]))

    if not {"earnings_call_date", "sentiment_score"}.issubset(row):
        logging.error(f"Failed to generate critical features for {ticker}. Aborting row.")
        return None
        
    return row